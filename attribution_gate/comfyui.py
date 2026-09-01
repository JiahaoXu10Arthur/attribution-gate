"""Recover an arm's factors from a ComfyUI PNG, instead of being told them.

ComfyUI embeds the whole API workflow in every PNG it writes, so the output
file is a complete record of what produced it. Reading the config back out of
the artifact removes one way to be wrong: the arm that gets adjudicated is the
arm that actually ran, not the one someone remembered running.

No dependencies. PNG chunks are parsed directly -- Pillow is not needed to
read a ``tEXt`` chunk, and requiring it for that would be the largest cost in
the package.

The text encoder is not where the prompt is typed
-------------------------------------------------
Prompt-building workflows chain string nodes: concatenate a quality prefix,
switch between branches, then feed the result to ``CLIPTextEncode``. Reading
the node someone typed into can miss most of what the model actually received
-- in one measured workflow the operator's own text was about a tenth of the
final prompt.

So resolution starts at the sampler's ``positive`` input and walks back, and
where a switch chose a branch, only the branch that actually ran is followed,
because the graph records which one it was.

Some prompts are not in the graph at all, and those are refused
--------------------------------------------------------------
Walking back only works while every node on the path is string plumbing whose
output is determined by its inputs. Two common node kinds break that:

- **runtime injectors** -- a LoRA trigger-word toggle emits words it read out
  of the LoRA's own metadata; the graph records the toggle, not the words
- **generative upsamplers** -- a node that expands a short prompt with a
  language model produces text that exists only at run time

For those the arm genuinely cannot be recovered from the PNG. This module
raises instead of returning what it managed to collect, because a silently
incomplete arm is exactly the failure the gate exists to prevent: it would
compare two arms on a fraction of their real factors and then state a verdict
about it.

Traversal is therefore an **allowlist**. An unrecognised node on the path is a
refusal that names it, not a guess. Real workflows also leak model identifiers
and enum values into any walk that follows every input: before the allowlist
existed, a render's "prompt" came back as ``long``, ``short`` and
``KBlueLeaf/TIPO-500M-ft | TIPO-500M-ft-F16.gguf`` -- two enum values and a
model identifier, collected as though they were prompt factors.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Dict, List

__all__ = ["read_text_chunks", "read_workflow", "positive_prompt",
           "negative_prompt", "arm_of", "PngArm", "ComfyUIError"]

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

#: Node classes whose output is a pure function of their string inputs, so
#: walking back through them reconstructs the real text. Matched as
#: substrings of ``class_type``.
_TRAVERSABLE = (
    "CLIPTextEncode",
    "TextEncode",
    "StringConcatenate",
    "JoinStringMulti",
    "JoinString",
    "PrimitiveString",
    "StringLiteral",
    "CR Text",
    "Text Multiline",
    "ShowText",
    "ConditioningCombine",
    "ConditioningConcat",
)

#: Switch nodes that record which branch they took as a numeric ``select``
#: input, so the branch that actually ran is recoverable from the graph.
#: Deliberately an explicit list rather than "any class containing Switch":
#: an unrecognised node called ``FooSwitch`` may do something else entirely
#: with an input named ``select``, and traversing it would report an arm that
#: never reached the model. Extend this tuple if your switch is a plain
#: index selector.
#:
#: ``Any Switch (rgthree)`` is deliberately absent: it forwards the first
#: non-null input, and which input was non-null is a run-time fact the graph
#: does not record.
_SELECT_SWITCHES = ("ImpactSwitch",)

#: Nodes that discard the conditioning they are given. The text upstream of
#: one of these did *not* reach the model, so reporting it as the arm would be
#: a silently wrong answer -- the failure this module exists to avoid.
_DISCARDS_CONDITIONING = ("ConditioningZeroOut",)

#: Input keys that are separators, enums, paths or device names rather than
#: content. ``delimiter`` is usually ``", "``, which becomes an empty factor.
_SKIP_KEYS = frozenset({
    "delimiter", "separator", "type", "device", "weight_dtype", "precision",
    "unet_name", "clip_name", "ckpt_name", "vae_name", "lora_name",
    "model_name", "upscale_method", "segmentor", "sampler_name", "scheduler",
    "image", "system", "filename", "filename_prefix", "url", "keep_alive",
})

#: Inputs that are structurally not text: model plumbing, latents, images.
#: These are skipped rather than refused -- a ``CLIPTextEncode`` always has a
#: ``clip`` input wired to a loader, and that is not a reason to give up on
#: recovering its ``text``.
_NON_TEXT_KEYS = frozenset({
    "clip", "vae", "latent_image", "samples", "images", "pixels", "mask",
    "control_net", "guider", "sigmas", "noise", "model",
})

#: Nodes that consume conditioning and expose a ``positive`` input.
_SAMPLERS = ("KSampler", "SamplerCustom", "KSamplerAdvanced")

#: Scalars on the sampler node that must be held constant for a two-arm
#: comparison to isolate anything. They are literals -- no traversal, no
#: allowlist, no new refusal modes -- and the tool already tells users to fix
#: the seed, so being handed proof of whether they did and not looking was the
#: least defensible omission in this module.
_CONTROLLED = ("seed", "noise_seed", "steps", "cfg", "sampler_name",
               "scheduler", "denoise")

#: Model and geometry inputs, found by walking back from the sampler rather
#: than by sitting on it. A checkpoint swap is the largest uncontrolled
#: variable a workflow can carry, and it was passing green.
_CONTROLLED_UPSTREAM = ("ckpt_name", "unet_name", "vae_name", "clip_name",
                        "width", "height")

#: Substring that marks a node as carrying LoRA settings. Matched against
#: ``class_type``, not against input keys: ``lora_name`` matched zero of the
#: six committed real fixtures while two of them load LoRAs, because the
#: LoraManager family holds its stack in ``loras: {"__value__": [...]}`` or in
#: ``lora_syntax``. The corpus behind those fixtures has no native
#: ``LoraLoader`` at all, so the key-name check had never once fired on a real
#: graph.
_LORA_NODE = "lora"

_MAX_DEPTH = 24

#: Total node visits allowed for one arm. A diamond -- a concatenate whose two
#: inputs come from the same node -- legitimately doubles the text, and real
#: ComfyUI would too, so the duplication is faithful and the walk must not
#: dedupe it. But nested diamonds double per level, so a graph within the
#: depth limit can expand to millions of visits. Refusing beats hanging, and
#: hanging is what a CI gate must never do.
_MAX_VISITS = 20000


class ComfyUIError(Exception):
    """The PNG carried no usable workflow, or one that cannot be resolved."""


def read_text_chunks(png) -> Dict[str, str]:
    """Return every ``tEXt``/``zTXt``/``iTXt`` chunk in a PNG as ``{key: value}``.

    Chunks are read in order and later duplicates win, matching how decoders
    generally behave.
    """
    png = Path(png)
    data = png.read_bytes()
    if not data.startswith(_PNG_SIG):
        raise ComfyUIError("%s is not a PNG" % png.name)

    out: Dict[str, str] = {}
    pos = len(_PNG_SIG)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # 4 length + 4 type + body + 4 crc
        if ctype == b"IEND":
            break
        try:
            if ctype == b"tEXt":
                key, _, val = body.partition(b"\x00")
                out[key.decode("latin-1")] = val.decode("utf-8", "replace")
            elif ctype == b"zTXt":
                key, _, rest = body.partition(b"\x00")
                # rest[0] is the compression-method byte.
                out[key.decode("latin-1")] = zlib.decompress(
                    rest[1:]).decode("utf-8", "replace")
            elif ctype == b"iTXt":
                key, _, rest = body.partition(b"\x00")
                flag = rest[0] if rest else 0
                rest = rest[2:]  # compression flag + method
                _, _, rest = rest.partition(b"\x00")  # language tag
                _, _, rest = rest.partition(b"\x00")  # translated keyword
                out[key.decode("latin-1")] = (
                    zlib.decompress(rest) if flag else rest
                ).decode("utf-8", "replace")
        except Exception:
            # A malformed chunk is not a reason to abandon the whole file.
            continue
    return out


def read_workflow(png) -> dict:
    """Return the API workflow ComfyUI embedded in ``png``.

    ComfyUI writes two things: ``prompt`` (the API graph, what actually ran)
    and ``workflow`` (the editor graph, for reopening in the UI). Only the
    former is authoritative, so only the former is used.
    """
    chunks = read_text_chunks(png)
    raw = chunks.get("prompt")
    if not raw:
        have = ", ".join(sorted(chunks)) or "none"
        raise ComfyUIError(
            "%s has no embedded ComfyUI workflow (text chunks present: %s). "
            "Images re-saved by an editor, or assembled by a script, usually "
            "lose it." % (Path(png).name, have))
    try:
        api = json.loads(raw)
    except ValueError as e:
        raise ComfyUIError("%s has an unreadable workflow: %s"
                           % (Path(png).name, e))
    if not isinstance(api, dict) or not api:
        raise ComfyUIError("%s has an empty workflow" % Path(png).name)
    return api


def _class_of(api: dict, nid) -> str:
    return str((api.get(str(nid)) or {}).get("class_type") or "")


def _resolve(api: dict, value, depth: int = 0, seen=None,
             budget: List[int] = None, polarity: str = "positive") -> List[str]:
    """Follow one node input back to every string that feeds it."""
    if depth > _MAX_DEPTH:
        raise ComfyUIError(
            "the prompt chain is more than %d nodes deep, or contains a cycle"
            % _MAX_DEPTH)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and len(value) == 2 and \
            isinstance(value[0], (str, int)):
        # Do not add to depth here. ``_node_strings`` is about to charge for
        # this same graph hop, and charging twice halved the budget: the limit
        # said 24 nodes and refused at 13.
        return _node_strings(api, str(value[0]), depth, seen, budget, polarity)
    return []


def _node_strings(api: dict, nid: str, depth: int = 0, seen=None,
                  budget: List[int] = None, polarity: str = "positive") -> List[str]:
    if depth > _MAX_DEPTH:
        raise ComfyUIError(
            "the prompt chain is more than %d nodes deep, or contains a cycle"
            % _MAX_DEPTH)
    budget = [_MAX_VISITS] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        raise ComfyUIError(
            "this graph expands to more than %d node visits -- some node is "
            "reached along very many paths. Refusing rather than grinding; "
            "pass the arm as text." % _MAX_VISITS)
    seen = set() if seen is None else seen
    if nid in seen:
        raise ComfyUIError("the prompt chain loops back on node %s" % nid)
    seen = seen | {nid}

    node = api.get(str(nid))
    if not isinstance(node, dict):
        return []
    cls = str(node.get("class_type") or "")
    inputs = node.get("inputs") or {}

    if any(s in cls for s in _DISCARDS_CONDITIONING):
        # Zeroing the *negative* is a standard pattern, and its result is
        # perfectly determinate: an empty negative. Only on the positive path
        # does a zero-out mean the text upstream never reached the model.
        if polarity == "negative":
            return []
        raise ComfyUIError(
            "cannot recover the prompt: node %s is a %r, which discards the "
            "conditioning it receives. Whatever text is upstream of it did "
            "not reach the model, so reporting that text as the arm would be "
            "wrong." % (nid, cls))

    # A switch picks one branch by an index stored in the graph, so which
    # branch ran *is* recoverable -- but only that branch. Following all of
    # them would report factors that never reached the model.
    if any(s in cls for s in _SELECT_SWITCHES) and "select" in inputs:
        try:
            sel = int(inputs["select"])
        except (TypeError, ValueError):
            raise ComfyUIError(
                "node %s (%s) has a non-numeric select, so the branch that "
                "ran cannot be determined" % (nid, cls))
        key = "input%d" % sel
        if key not in inputs:
            raise ComfyUIError(
                "node %s (%s) selects %r, which is not present in the saved "
                "graph" % (nid, cls, key))
        return _resolve(api, inputs[key], depth + 1, seen, budget, polarity)

    if not any(s in cls for s in _TRAVERSABLE):
        raise ComfyUIError(
            "cannot recover the prompt: node %s is a %r, which is not a "
            "string node whose output follows from its inputs. If it injects "
            "text at run time (LoRA trigger words) or generates it (a prompt "
            "upsampler), the text is not in this file at all. Pass the arm as "
            "text instead of a PNG." % (nid, cls))

    out: List[str] = []
    # Dict order is insertion order, which for a ComfyUI API graph is the
    # node's declared input order -- so a concatenate node comes out in the
    # order it would actually concatenate.
    for key, val in inputs.items():
        if key in _SKIP_KEYS or key in _NON_TEXT_KEYS:
            continue
        if isinstance(val, (int, float, bool)):
            continue
        out.extend(_resolve(api, val, depth + 1, seen, budget, polarity))
    return out


def positive_prompt(api: dict) -> str:
    """Return the text that actually reached the positive conditioning.

    Raises :class:`ComfyUIError` if the workflow has no sampler, if the chain
    passes through a node whose output is not recoverable from the graph, or
    if several samplers disagree about the prompt.
    """
    samplers = [nid for nid in api
                if any(s in _class_of(api, nid) for s in _SAMPLERS)]
    if not samplers:
        raise ComfyUIError(
            "no sampler node found, so the positive conditioning cannot be "
            "located; pass the arm as text instead")

    found = []
    for nid in sorted(samplers):
        pos = (api[nid].get("inputs") or {}).get("positive")
        if pos is None:
            continue
        text = ", ".join(p for p in _resolve(api, pos) if p.strip())
        if text.strip():
            found.append(text)

    uniq = sorted(set(found))
    if not uniq:
        raise ComfyUIError(
            "the sampler's positive input did not resolve to any text; the "
            "prompt may come from an image-interrogation branch, which "
            "cannot be compared as a factor set")
    if len(uniq) > 1:
        raise ComfyUIError(
            "this workflow has %d samplers whose positive prompts differ, so "
            "there is no single arm to adjudicate. Pass the arm as text "
            "explicitly." % len(uniq))
    return uniq[0]


def negative_prompt(api: dict) -> str:
    """The text that reached the negative conditioning, by the same walk.

    Raises like :func:`positive_prompt` when it cannot be recovered. It is
    deliberately *not* rescued by comparing the two negative subgraphs
    structurally when the text will not resolve: identical machinery can still
    emit different text if the data it reads off disk changed, which is the
    same reason LoRA trigger injectors are refused in the first place.
    """
    samplers = [nid for nid in api
                if any(s in _class_of(api, nid) for s in _SAMPLERS)]
    if not samplers:
        raise ComfyUIError("no sampler node found, so the negative "
                           "conditioning cannot be located")
    found = []
    for nid in sorted(samplers):
        neg = (api[nid].get("inputs") or {}).get("negative")
        if neg is None:
            continue
        found.append(", ".join(
            p for p in _resolve(api, neg, polarity="negative") if p.strip()))
    if not found:
        # No sampler exposed a ``negative`` input at all. That is an
        # unreadable state, not an empty negative, and handing back "" would
        # let it compare equal to a genuinely empty one.
        raise ComfyUIError(
            "no sampler here exposes a negative input, so the negative "
            "conditioning cannot be read")
    uniq = sorted(set(found))
    if len(uniq) > 1:
        raise ComfyUIError(
            "this workflow has %d samplers whose negative prompts differ"
            % len(uniq))
    return uniq[0]


class PngArm(str):
    """The positive prompt, carrying what the tool saw and did not compare.

    A ``str`` subclass so every existing caller -- ``factors``, ``weights``,
    ``changes`` -- keeps working untouched, while ``adjudicate`` can pick up
    ``.residue`` and refuse a comparison whose seed or negative prompt moved.

    The distinction that earns this: in text mode the *caller* authors the
    arm, and whatever they left out is by construction outside the comparison.
    In PNG mode the *tool* authors it. A premise this package manufactured
    carries a duty a premise it was handed does not.
    """

    residue: dict

    def __new__(cls, text: str, residue: dict):
        self = super().__new__(cls, text)
        self.residue = residue
        return self


#: A value that is wired in from somewhere this reader cannot follow. It must
#: never compare equal to anything, including another unknown: "I could not
#: read either seed" is not evidence that the seeds matched.
class _Unknown(object):
    __slots__ = ("why",)

    def __init__(self, why: str):
        self.why = why

    def __repr__(self) -> str:
        return "<unknown: %s>" % self.why

    def __eq__(self, other) -> bool:
        return False

    def __ne__(self, other) -> bool:
        return True

    def __hash__(self) -> int:
        return id(self)


def _scalar(api: dict, value, depth: int = 0):
    """Read a sampler setting, following one link if it is wired in.

    Seeds are usually not literals. In one real corpus the sampler's ``seed``
    was a link in 770 of 803 workflows and a literal in 33, so a reader that
    only understood literals would have missed the single most important
    controlled variable almost every time -- and missed it silently.
    """
    if depth > 4:
        return _Unknown("seed chain too deep")
    if not isinstance(value, list):
        return value
    if len(value) != 2 or not isinstance(value[0], (str, int)):
        return _Unknown("unreadable link")
    node = api.get(str(value[0]))
    if not isinstance(node, dict):
        return _Unknown("dangling link")
    inputs = node.get("inputs") or {}

    # Numbers only. Taking "whatever single scalar is on the node" read a
    # SeedGenerator's mode as the seed -- both arms came back ``'randomize'``,
    # compared equal, and passed at exit 0 with different seeds. A setting
    # that arrives as a string is by definition not a number this can compare.
    numbers = {k: v for k, v in inputs.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    links = [v for v in inputs.values() if isinstance(v, list)]

    if len(numbers) == 1:
        return next(iter(numbers.values()))
    # Follow the wire before giving up: a generator node commonly holds its
    # mode as a literal and its value as a link.
    if len(links) == 1:
        return _scalar(api, links[0], depth + 1)
    if not numbers and not links:
        return _Unknown("%s exposes no number" % node.get("class_type"))
    return _Unknown("%s exposes %d numbers and %d links"
                    % (node.get("class_type"), len(numbers), len(links)))


def _controlled(api: dict) -> dict:
    """Sampler settings that must match for the comparison to isolate anything."""
    out = {}
    for nid in sorted(api):
        if not any(s in _class_of(api, nid) for s in _SAMPLERS):
            continue
        inputs = api[nid].get("inputs") or {}
        for key in _CONTROLLED:
            if key in inputs:
                out.setdefault(key, []).append(_scalar(api, inputs[key]))

    # Model and size live on loader nodes, not on the sampler. Collect them
    # graph-wide: which node holds them does not matter, only that the two
    # runs used the same ones.
    for nid in sorted(api):
        inputs = (api.get(nid) or {}).get("inputs") or {}
        for key in _CONTROLLED_UPSTREAM:
            v = inputs.get(key)
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                out.setdefault(key, []).append(v)

    # A gate does not need to parse the stack -- it needs to know whether both
    # arms carried the same one. So every literal input on a LoRA node goes in
    # as it stands, which compares a shape nobody has written a reader for yet
    # exactly as well as a familiar one. Sorted, so node ids and ordering --
    # which ComfyUI rewrites on every save -- cannot make identical stacks
    # look different.
    stack = []
    for nid in sorted(api):
        node = api.get(nid) or {}
        if _LORA_NODE not in str(node.get("class_type") or "").lower():
            continue
        inputs = node.get("inputs") or {}
        for key in sorted(inputs):
            value = inputs[key]
            if isinstance(value, list):
                continue  # a wire, not a setting
            stack.append("%s.%s=%s" % (node.get("class_type"), key,
                                       json.dumps(value, sort_keys=True)))
    if stack:
        out["lora settings"] = [tuple(sorted(stack))]

    return {k: (v[0] if len(v) == 1 else tuple(v)) for k, v in out.items()}


def arm_of(png) -> "PngArm":
    """Read one arm straight from a ComfyUI PNG, plus what was not compared."""
    api = read_workflow(png)
    # Resolve the positive first. "I cannot read your prompt" is the more
    # fundamental failure, and reporting the negative's problem ahead of it
    # would misfile the reason.
    positive = positive_prompt(api)
    residue = _controlled(api)
    residue["negative prompt"] = negative_prompt(api)
    return PngArm(positive, residue)
