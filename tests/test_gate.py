import json
import struct
import zlib

import pytest

from attribution_gate import (ABSENT, CONSTANT, IDENTICAL, MULTIVARIATE,
                              UNMODELLED,
                              VALID, PlanError, adjudicate, difference,
                              changes, factors, plan, weights)
from attribution_gate.comfyui import (ComfyUIError, arm_of, positive_prompt,
                                      read_text_chunks, read_workflow)


# --------------------------------------------------------------- normalising

def test_factors_splits_and_lowercases():
    assert factors("1girl, Solo,  BLUE hair ") == {"1girl", "solo", "blue hair"}


def test_factors_drops_empty_and_trailing_comma():
    assert factors("a, , b,") == {"a", "b"}


def test_factors_strips_attention_weights():
    assert factors("(blue hair:1.3), smile") == {"blue hair", "smile"}
    assert factors("(blue hair: 1.3)") == {"blue hair"}


def test_factors_accepts_an_iterable():
    assert factors(["A", "b "]) == {"a", "b"}


def test_weight_syntax_does_not_manufacture_a_difference():
    # The whole point of normalising: these two arms are the same experiment.
    only_a, only_b = difference("smile, blue hair", "smile, (blue hair:1.2)")
    assert not only_a and not only_b


# ------------------------------------------------------------------ verdicts

def test_valid_single_variable():
    v = adjudicate("a, b", "a, b, c", "c")
    assert v.code == VALID
    assert v.ok and bool(v) is True
    assert v.candidates == ("c",)


def test_constant_is_refused():
    """The real incident: the credited factor was in both arms."""
    a = "1girl, silhouetted figure, sunset"
    b = "1girl, silhouetted figure, sunset, rim lighting"
    v = adjudicate(a, b, "silhouetted figure")
    assert v.code == CONSTANT
    assert not v
    # And it says what it could have been instead.
    assert v.candidates == ("rim lighting",)


def test_absent_is_refused():
    v = adjudicate("a, b", "a, b, c", "zzz")
    assert v.code == ABSENT
    assert not v


def test_identical_arms_are_refused_before_anything_else():
    v = adjudicate("a, b", "a, b", "a")
    assert v.code == IDENTICAL
    assert v.candidates == ()
    # Same experiment run twice, or an instrument that sees nothing: either
    # way the fix is at the bench, not in the sentence.
    assert v.exit_code == 2 and v.needs_rerun


def test_multivariate_is_refused_even_though_the_claim_is_present():
    v = adjudicate("a", "a, b, c", "b")
    assert v.code == MULTIVARIATE
    assert not v
    assert v.candidates == ("b", "c")


def test_claim_must_name_exactly_one_factor():
    with pytest.raises(ValueError):
        adjudicate("a", "a, b", "b, c")
    with pytest.raises(ValueError):
        adjudicate("a", "a, b", "")


def test_direction_does_not_matter():
    """Removing a factor is as attributable as adding one."""
    assert adjudicate("a, b", "a", "b").code == VALID


# ---------------------------------------------------------------------- plan

def test_plan_add():
    assert plan("a, b", add="c") == "a, b, c"


def test_plan_add_refuses_a_factor_already_present():
    with pytest.raises(PlanError):
        plan("a, b", add="b")


def test_plan_add_refuses_a_case_variant_of_a_present_factor():
    # Silently produces two identical arms if not caught.
    with pytest.raises(PlanError):
        plan("a, Blue Hair", add="blue hair")


def test_plan_swap():
    assert plan("a, b, c", swap="b=z") == "a, z, c"
    only_a, only_b = difference("a, b, c", plan("a, b, c", swap="b=z"))
    assert set(only_a) | set(only_b) == {"b", "z"}


def test_plan_swap_refuses_absent_old():
    with pytest.raises(PlanError):
        plan("a, b", swap="zzz=c")


def test_plan_swap_refuses_when_new_already_present():
    """Swapping b->a on "a, b" only deletes b; calling it a swap misdescribes it."""
    with pytest.raises(PlanError):
        plan("a, b", swap="b=a")


def test_plan_requires_exactly_one_operation():
    with pytest.raises(PlanError):
        plan("a, b")
    with pytest.raises(PlanError):
        plan("a, b", add="c", swap="a=z")


def test_plan_output_is_adjudicable():
    base = "1girl, solo, outdoors"
    arm_b = plan(base, add="rim lighting")
    assert adjudicate(base, arm_b, "rim lighting").code == VALID


# ------------------------------------------------------------------- ComfyUI

def _png(chunks):
    """Build a minimal PNG carrying the given (type, payload) chunks."""
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    for ctype, payload in [(b"IHDR", ihdr)] + list(chunks) + [(b"IEND", b"")]:
        out += struct.pack(">I", len(payload))
        out += ctype + payload
        out += struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
    return bytes(out)


def _text_chunk(key, value):
    return (b"tEXt", key.encode() + b"\x00" + value.encode())


WORKFLOW = {
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "some_model.safetensors"}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "1girl, solo, rim lighting", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "worst quality", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler",
          "inputs": {"seed": 1, "sampler_name": "euler",
                     "positive": ["6", 0], "negative": ["7", 0]}},
}


def test_reads_text_chunks(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text_chunk("prompt", '{"x": 1}')]))
    assert read_text_chunks(p)["prompt"] == '{"x": 1}'


def test_reads_compressed_text_chunks(tmp_path):
    p = tmp_path / "z.png"
    payload = b"prompt\x00\x00" + zlib.compress(b'{"x": 2}')
    p.write_bytes(_png([(b"zTXt", payload)]))
    assert read_text_chunks(p)["prompt"] == '{"x": 2}'


def test_positive_prompt_ignores_the_negative_branch(tmp_path):
    p = tmp_path / "w.png"
    p.write_bytes(_png([_text_chunk("prompt", json.dumps(WORKFLOW))]))
    assert arm_of(p) == "1girl, solo, rim lighting"


def test_positive_prompt_follows_a_concatenate_chain():
    """The prompt is not always typed into the encoder.

    A quality prefix and injected trigger words reach the model through
    intermediate string nodes. Reading only the node a human typed into misses
    them -- which is how 90% of a real prompt once went unnoticed.
    """
    api = {
        "10": {"class_type": "PrimitiveString",
               "inputs": {"value": "masterpiece, best quality"}},
        "11": {"class_type": "PrimitiveString",
               "inputs": {"value": "1girl, solo"}},
        "12": {"class_type": "StringConcatenate",
               "inputs": {"string_a": ["10", 0], "string_b": ["11", 0],
                          "delimiter": ", "}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    assert positive_prompt(api) == "masterpiece, best quality, 1girl, solo"


def test_loader_filenames_do_not_become_factors():
    """A ``clip`` wire into a loader is plumbing, not a reason to give up.

    Following it would index ``style_v1.safetensors`` as though it were a
    prompt factor.
    """
    api = {
        "4": {"class_type": "LoraLoader",
              "inputs": {"lora_name": "style_v1.safetensors"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "1girl", "clip": ["4", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    assert positive_prompt(api) == "1girl"


def test_runtime_injected_text_is_refused_not_partially_reported():
    """A LoRA trigger toggle emits words that are not in the graph.

    Returning just the half that *is* in the graph would compare two arms on a
    fraction of their real factors -- the exact failure this package exists to
    prevent. In the workflow this was taken from, the missing half was 90% of
    the prompt.
    """
    api = {
        "1086": {"class_type": "Lora Stacker (LoraManager)",
                 "inputs": {"text": "<lora:some_style:1.00>"}},
        "1078": {"class_type": "TriggerWord Toggle (LoraManager)",
                 "inputs": {"group_mode": ["1086", 0]}},
        "1043": {"class_type": "StringConcatenate",
                 "inputs": {"string_a": "masterpiece, best quality",
                            "string_b": ["1078", 0], "delimiter": ", "}},
        "1044": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1043", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1044", 0]}},
    }
    with pytest.raises(ComfyUIError) as e:
        positive_prompt(api)
    assert "TriggerWord Toggle" in str(e.value)


def test_generative_upsampler_is_refused():
    """A node that writes the prompt at run time leaves nothing to recover.

    Following its inputs collected a model identifier and an enum value as
    though they were prompt factors, which is how this case was found.
    """
    api = {
        "22": {"class_type": "TIPO",
               "inputs": {"nl_prompt": "n", "device": "cpu",
                          "tipo_model": "KBlueLeaf/TIPO-500M-ft | x.gguf",
                          "tag_prompt": "1girl"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["22", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    with pytest.raises(ComfyUIError) as e:
        positive_prompt(api)
    assert "TIPO" in str(e.value)


def test_cycles_do_not_hang():
    api = {
        "1": {"class_type": "StringConcatenate", "inputs": {"a": ["2", 0]}},
        "2": {"class_type": "StringConcatenate", "inputs": {"a": ["1", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    with pytest.raises(ComfyUIError):
        positive_prompt(api)


def test_ambiguous_workflow_is_refused_not_guessed():
    api = dict(WORKFLOW)
    api["8"] = {"class_type": "CLIPTextEncode",
                "inputs": {"text": "a completely different prompt"}}
    api["9"] = {"class_type": "KSampler", "inputs": {"positive": ["8", 0]}}
    with pytest.raises(ComfyUIError):
        positive_prompt(api)


def test_missing_workflow_is_an_error(tmp_path):
    p = tmp_path / "plain.png"
    p.write_bytes(_png([]))
    with pytest.raises(ComfyUIError):
        read_workflow(p)


def test_not_a_png(tmp_path):
    p = tmp_path / "nope.png"
    p.write_bytes(b"hello")
    with pytest.raises(ComfyUIError):
        read_text_chunks(p)


def test_end_to_end_two_pngs(tmp_path):
    """The case the tool exists for, run through the artifacts themselves."""
    a = dict(WORKFLOW)
    b = json.loads(json.dumps(WORKFLOW))
    b["6"]["inputs"]["text"] = "1girl, solo, rim lighting, sunset"

    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    pa.write_bytes(_png([_text_chunk("prompt", json.dumps(a))]))
    pb.write_bytes(_png([_text_chunk("prompt", json.dumps(b))]))

    assert adjudicate(arm_of(pa), arm_of(pb), "sunset").code == VALID
    assert adjudicate(arm_of(pa), arm_of(pb), "rim lighting").code == CONSTANT


def test_switch_follows_only_the_branch_that_ran():
    """A switch records which input it selected, so that branch is recoverable.

    Following every branch would report factors that never reached the model.
    """
    api = {
        "10": {"class_type": "CR Text", "inputs": {"text": "branch one"}},
        "11": {"class_type": "CR Text", "inputs": {"text": "branch two"}},
        "12": {"class_type": "ImpactSwitch",
               "inputs": {"select": 2, "sel_mode": True,
                          "input1": ["10", 0], "input2": ["11", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    assert positive_prompt(api) == "branch two"


def test_switch_selecting_a_missing_branch_is_refused():
    api = {
        "10": {"class_type": "CR Text", "inputs": {"text": "branch one"}},
        "12": {"class_type": "ImpactSwitch",
               "inputs": {"select": 3, "input1": ["10", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    with pytest.raises(ComfyUIError):
        positive_prompt(api)


# ------------------------------------------------------------------- weights

def test_weights_parses_names_and_values():
    assert weights("a, (b:1.5)") == {"a": None, "b": 1.5}


def test_weights_handles_negative():
    assert weights("(a:-1)") == {"a": -1.0}


def test_a_weight_change_is_one_change_not_zero():
    """The bug this guards: normalising weights away hid a real experiment.

    Dropping the weight made both arms look identical, so a comparison that
    varied exactly one thing was told its arms were the same and to go look at
    the seed.
    """
    v = adjudicate("a, b", "a, (b:1.5)", "b")
    assert v.code == VALID
    assert "weight" in v.reason


def test_a_weight_change_is_one_change_not_two():
    """And it is not an unrelated remove-plus-add, which would read as
    multivariate and reject a perfectly good single-variable comparison."""
    added, removed, reweighted = changes("a, b", "a, (b:1.5)")
    assert (added, removed, reweighted) == ((), (), ("b",))


def test_reweighted_plus_added_is_multivariate():
    assert adjudicate("a, b", "a, (b:1.5), d", "b").code == MULTIVARIATE


def test_same_weight_on_both_sides_is_identical():
    assert adjudicate("a, (b:1.2)", "a, (b:1.2)", "b").code == IDENTICAL


def test_a_respelt_weight_is_not_a_reordering():
    """(b:1.5) and (b:1.50) are the same level, so this is truly identical."""
    assert adjudicate("a, (b:1.5)", "a, (b:1.50)", "b").code == IDENTICAL


def test_constant_requires_unchanged_not_merely_present():
    """A factor present in both arms but reweighted is still attributable."""
    assert adjudicate("(b:1.0), a", "(b:1.5), a", "b").code == VALID
    assert adjudicate("b, a", "b, a, c", "b").code == CONSTANT


def test_conditioning_zero_out_is_refused():
    """The text upstream of a zero-out never reached the model.

    Reporting it as the arm would be a silently wrong answer of exactly the
    kind this module refuses to give. It was on the traversal allowlist once.
    """
    api = {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, sunset"}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
    }
    with pytest.raises(ComfyUIError) as e:
        positive_prompt(api)
    assert "discards" in str(e.value)


def test_unknown_switch_class_does_not_bypass_the_allowlist():
    """Switch handling used to fire on any class containing 'Switch'.

    An unrecognised node may do something else entirely with an input named
    'select', so matching on the word alone reopened the hole the allowlist
    was added to close.
    """
    api = {
        "10": {"class_type": "CR Text", "inputs": {"text": "branch one"}},
        "12": {"class_type": "TotallyUndocumentedSwitchThing",
               "inputs": {"select": 1, "input1": ["10", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    with pytest.raises(ComfyUIError) as e:
        positive_prompt(api)
    assert "TotallyUndocumentedSwitchThing" in str(e.value)


def test_first_non_null_switch_is_refused():
    """rgthree's Any Switch forwards the first non-null input.

    Which input was non-null is a run-time fact; the graph does not record it,
    so the branch that ran is not recoverable. Observed 100 times in a real
    corpus, which is why it is called out by name.
    """
    api = {
        "10": {"class_type": "CR Text", "inputs": {"text": "x"}},
        "12": {"class_type": "Any Switch (rgthree)", "inputs": {"any_01": ["10", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    with pytest.raises(ComfyUIError):
        positive_prompt(api)


# ----------------------------------------------------------------------- CLI

def test_missing_png_is_an_error_not_a_prompt(tmp_path, monkeypatch):
    """A mistyped image path must not be adjudicated as though it were a tag.

    The earlier fallback made `check arm_a.png arm_b.png "sunset"` return a
    confident ABSENT verdict whose "factors" were the two filenames.
    """
    from attribution_gate.__main__ import main
    monkeypatch.chdir(tmp_path)
    assert main(["check", "nope_a.png", "nope_b.png", "sunset"]) == 3


def test_png_arm_is_read_when_it_exists(tmp_path, monkeypatch):
    from attribution_gate.__main__ import main
    a = tmp_path / "arm_a.png"
    b = tmp_path / "arm_b.png"
    wf_b = json.loads(json.dumps(WORKFLOW))
    wf_b["6"]["inputs"]["text"] = "1girl, solo, rim lighting, sunset"
    a.write_bytes(_png([_text_chunk("prompt", json.dumps(WORKFLOW))]))
    b.write_bytes(_png([_text_chunk("prompt", json.dumps(wf_b))]))
    monkeypatch.chdir(tmp_path)
    assert main(["check", "arm_a.png", "arm_b.png", "sunset"]) == 0
    assert main(["check", "arm_a.png", "arm_b.png", "rim lighting"]) == 1


# --------------------------------------------------- precedence and exit codes

def test_reordering_is_not_reported_as_no_change():
    """The arms are not the same text. Saying IDENTICAL would send the user
    off to investigate the seed for a change they made themselves."""
    v = adjudicate("a, b", "b, a", "a")
    assert v.code == UNMODELLED
    assert v.exit_code == 2


def test_repetition_is_not_reported_as_no_change():
    v = adjudicate("a, a, b", "a, b", "a")
    assert v.code == UNMODELLED


def test_multivariate_outranks_a_constant_claim():
    """Otherwise the user re-credits to another factor and is refused again.

    Two round trips to learn what one should have said, from the codes whose
    whole job is to say what to do next.
    """
    v = adjudicate("a, b", "a, b, c, d", "a")
    assert v.code == MULTIVARIATE
    assert v.exit_code == 2
    # The constant diagnosis is not lost, just demoted to a note.
    assert "both arms" in v.note


def test_multivariate_outranks_an_absent_claim():
    v = adjudicate("a, b", "a, b, c, d", "zzz")
    assert v.code == MULTIVARIATE
    assert "neither arm" in v.note


def test_exit_codes_split_bad_claim_from_bad_experiment():
    bad_claim = [adjudicate("a, b", "a, b, c", "a"),      # CONSTANT
                 adjudicate("a, b", "a, b, c", "zzz")]    # ABSENT
    bad_experiment = [adjudicate("a, b", "a, b", "a"),        # IDENTICAL
                      adjudicate("a", "a, b, c", "b"),        # MULTIVARIATE
                      adjudicate("a, b", "b, a", "a")]        # UNMODELLED
    assert all(v.exit_code == 1 and not v.needs_rerun for v in bad_claim)
    assert all(v.exit_code == 2 and v.needs_rerun for v in bad_experiment)
    assert adjudicate("a", "a, b", "b").exit_code == 0


def test_unknown_verdict_code_fails_toward_rerun():
    """Fail safe: an unmapped code must not read as 'just fix your wording'."""
    from attribution_gate import Verdict
    assert Verdict("SOMETHING_NEW", "", "x").exit_code == 2


def test_constant_message_points_at_the_2x2():
    v = adjudicate("a, b", "a, b, c", "a")
    assert "2x2" in v.reason


# ------------------------------------------------------------- plan is honest

def test_plan_swap_builds_a_two_factor_design_and_says_so():
    """A swap is two factors. The gate refuses it, by design.

    `sunset` and `sunrise` share no key; calling them two levels of one factor
    is domain knowledge this package refuses to have. This test exists because
    plan() and adjudicate() once disagreed about it silently.
    """
    base = "1girl, sunset"
    arm_b = plan(base, swap="sunset=sunrise")
    assert arm_b == "1girl, sunrise"
    v = adjudicate(base, arm_b, "sunrise")
    assert v.code == MULTIVARIATE
    assert v.exit_code == 2


def test_plan_swap_refuses_a_multi_factor_side():
    """`swap="a, b=c"` used to pick whichever of a/b set iteration yielded."""
    with pytest.raises(PlanError):
        plan("a, b", swap="a, b=c")
    with pytest.raises(PlanError):
        plan("a, b", swap="a=c, d")


# ------------------------------------------- holes found in code review, high

def test_a_duplicate_at_conflicting_weights_is_refused():
    """weights() is a dict, so it could only keep one of them.

    Last-wins made *removing the duplicate* look like a weight change and
    passed it as VALID -- the tool inventing an edit that never happened.
    """
    v = adjudicate("x, (a:1.2), (a:1.5)", "x, (a:1.2)", "a")
    assert v.code == UNMODELLED
    assert "more than once" in v.reason


def test_a_duplicate_at_the_same_weight_is_fine():
    """Repetition is only a problem when it hides a level."""
    assert adjudicate("a, a, b", "a, a, b, c", "c").code == VALID


def test_a_reordering_riding_along_with_a_real_change_is_caught():
    """Checking order only when nothing else changed let it slip past on the
    pass path -- the one path nothing downstream re-checks."""
    v = adjudicate("a, b", "b, a, c", "c")
    assert v.code == UNMODELLED
    assert v.exit_code == 2


def test_a_clean_single_change_still_passes():
    for a, b, claim in [("a, b", "a, b, c", "c"),
                        ("a, b, c", "a, b", "c"),
                        ("a, (b:1.0)", "a, (b:1.5)", "b")]:
        assert adjudicate(a, b, claim).code == VALID


def test_depth_limit_matches_the_number_it_reports():
    """_resolve and _node_strings each charged for the same hop, so the limit
    said 24 nodes and refused at 13."""
    from attribution_gate.comfyui import _MAX_DEPTH

    def chain(n):
        api = {str(i): {"class_type": "StringConcatenate",
                        "inputs": {"a": [str(i + 1), 0]}} for i in range(n)}
        api[str(n)] = {"class_type": "CR Text", "inputs": {"text": "leaf"}}
        api["enc"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ["0", 0]}}
        api["s"] = {"class_type": "KSampler", "inputs": {"positive": ["enc", 0]}}
        return api

    assert positive_prompt(chain(_MAX_DEPTH - 4)) == "leaf"
    with pytest.raises(ComfyUIError):
        positive_prompt(chain(_MAX_DEPTH + 4))


def test_exploding_fan_in_is_refused_not_ground_through():
    """A concatenate fed twice from one node doubles the text, and nesting
    that doubles per level. Faithful, but a CI gate must not hang."""
    n = 16
    api = {str(i): {"class_type": "StringConcatenate",
                    "inputs": {"a": [str(i + 1), 0], "b": [str(i + 1), 0]}}
           for i in range(n)}
    api[str(n)] = {"class_type": "CR Text", "inputs": {"text": "leaf"}}
    api["enc"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ["0", 0]}}
    api["s"] = {"class_type": "KSampler", "inputs": {"positive": ["enc", 0]}}
    with pytest.raises(ComfyUIError) as e:
        positive_prompt(api)
    assert "node visits" in str(e.value)


def test_a_small_diamond_still_resolves_faithfully():
    """The duplication is real -- ComfyUI would produce it too."""
    api = {
        "10": {"class_type": "CR Text", "inputs": {"text": "x"}},
        "12": {"class_type": "StringConcatenate",
               "inputs": {"a": ["10", 0], "b": ["10", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["12", 0]}},
    }
    assert positive_prompt(api) == "x, x"


def test_internal_errors_are_not_reported_as_usage_errors(monkeypatch):
    """A defect in the gate must not look like the caller's mistake.

    Exit 3 means "you used it wrong"; a CI job would keep going, green,
    against a gate that had stopped evaluating anything.
    """
    import attribution_gate.__main__ as cli

    def boom(*a, **k):
        raise RuntimeError("internal defect")

    monkeypatch.setattr(cli, "adjudicate", boom)
    with pytest.raises(RuntimeError):
        cli.main(["check", "a", "a, b", "b"])


# ------------------------------------------- residue: the tool-authored arm

def _png_pair(tmp_path, pos_a, pos_b, neg_a="worst quality",
              neg_b="worst quality", seed_a=1, seed_b=1, cfg_a=5.0, cfg_b=5.0):
    def wf(pos, neg, seed, cfg):
        return {
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg}},
            "99": {"class_type": "Seed (rgthree)", "inputs": {"seed": seed}},
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": ["99", 0], "cfg": cfg, "steps": 30,
                             "positive": ["6", 0], "negative": ["7", 0]}},
        }
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(_png([_text_chunk("prompt", json.dumps(wf(pos_a, neg_a, seed_a, cfg_a)))]))
    b.write_bytes(_png([_text_chunk("prompt", json.dumps(wf(pos_b, neg_b, seed_b, cfg_b)))]))
    return a, b


def test_seed_wired_in_by_a_link_is_still_read(tmp_path):
    """In one real corpus the sampler's seed was a link in 770 of 803
    workflows and a literal in 33. A literal-only reader would have missed the
    most important controlled variable almost every time, and missed it
    silently."""
    a, _ = _png_pair(tmp_path, "1girl", "1girl", seed_a=777000)
    assert arm_of(a).residue["seed"] == 777000


def test_a_different_seed_blocks_the_verdict(tmp_path):
    a, b = _png_pair(tmp_path, "1girl", "1girl, sunset", seed_a=1, seed_b=2)
    v = adjudicate(arm_of(a), arm_of(b), "sunset")
    assert v.code == UNMODELLED and v.exit_code == 2
    assert "seed" in v.reason


def test_a_different_negative_prompt_blocks_the_verdict(tmp_path):
    a, b = _png_pair(tmp_path, "1girl", "1girl, sunset",
                     neg_a="worst quality", neg_b="worst quality, blurry")
    v = adjudicate(arm_of(a), arm_of(b), "sunset")
    assert v.code == UNMODELLED
    assert "negative prompt" in v.reason


def test_a_different_cfg_blocks_the_verdict(tmp_path):
    a, b = _png_pair(tmp_path, "1girl", "1girl, sunset", cfg_a=5.0, cfg_b=7.0)
    assert adjudicate(arm_of(a), arm_of(b), "sunset").code == UNMODELLED


def test_a_properly_controlled_png_pair_still_passes(tmp_path):
    a, b = _png_pair(tmp_path, "1girl", "1girl, sunset")
    assert adjudicate(arm_of(a), arm_of(b), "sunset").code == VALID


def test_text_arms_are_untouched_by_the_residue_check():
    """Text callers author their own arm; nothing is checked behind them."""
    assert adjudicate("a, b", "a, b, c", "c").code == VALID
    assert not hasattr("a, b", "residue")


def test_a_png_arm_is_a_plain_string_everywhere_else(tmp_path):
    a, _ = _png_pair(tmp_path, "1girl, sunset", "1girl")
    arm = arm_of(a)
    assert isinstance(arm, str)
    assert factors(arm) == {"1girl", "sunset"}


def test_an_unreadable_setting_never_compares_equal(tmp_path):
    """"I could not read either seed" is not evidence that the seeds matched."""
    from attribution_gate.comfyui import _Unknown
    u = _Unknown("test")
    assert u != u and u != 1 and not (u == u)


def test_zero_out_on_the_negative_branch_is_an_empty_negative():
    """Zeroing the negative is a standard pattern and is perfectly
    determinate. Only on the positive path does it mean the text never
    reached the model."""
    from attribution_gate.comfyui import negative_prompt
    api = {
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl"}},
        "3": {"class_type": "KSampler",
              "inputs": {"positive": ["6", 0], "negative": ["9", 0]}},
    }
    assert negative_prompt(api) == ""
    assert positive_prompt(api) == "1girl"


def test_mixing_a_png_arm_with_a_text_arm_says_what_was_not_checked(tmp_path):
    """One arm read from a workflow, one typed: nothing to compare against.

    The tool cannot say the runs were controlled. It can say that it could
    not say, and that must not be silent on the pass path.
    """
    a, _ = _png_pair(tmp_path, "1girl", "1girl")
    arm_a = arm_of(a)
    v = adjudicate(arm_a, str(arm_a) + ", sunset", "sunset")
    assert v.code == VALID
    assert "only arm A carried workflow settings" in v.note


def test_the_mixed_note_reaches_the_cli_on_a_pass(tmp_path, monkeypatch, capsys):
    a, _ = _png_pair(tmp_path, "1girl", "1girl")
    from attribution_gate.__main__ import main
    monkeypatch.chdir(tmp_path)
    assert main(["check", "a.png", "1girl, sunset", "sunset"]) == 0
    assert "were not compared" in capsys.readouterr().out


def test_two_text_arms_carry_no_mixed_note():
    assert adjudicate("a, b", "a, b, c", "c").note == ""


# --------------------------------------- holes found in final verification

def test_a_non_numeric_scalar_cannot_impersonate_the_seed():
    """Taking "whatever single scalar is on the node" read a generator's mode
    as the seed: both arms came back 'randomize', compared equal, and passed
    at exit 0 with genuinely different seeds."""
    from attribution_gate.comfyui import _scalar, _Unknown
    api = {"98": {"class_type": "SeedGenerator",
                  "inputs": {"mode": "randomize", "seed": ["97", 0]}},
           "97": {"class_type": "Seed (rgthree)", "inputs": {"seed": 11111111}}}
    assert _scalar(api, ["98", 0]) == 11111111

    only_text = {"98": {"class_type": "IntConstant",
                        "inputs": {"label": "the seed knob"}}}
    assert isinstance(_scalar(only_text, ["98", 0]), _Unknown)


def test_a_checkpoint_swap_does_not_pass(tmp_path):
    """The largest uncontrolled variable a workflow can carry was unread."""
    def wf(ckpt, pos):
        return {"4": {"class_type": "CheckpointLoaderSimple",
                      "inputs": {"ckpt_name": ckpt}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad"}},
                "3": {"class_type": "KSampler",
                      "inputs": {"seed": 7, "cfg": 5.0, "positive": ["6", 0],
                                 "negative": ["7", 0]}}}
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(_png([_text_chunk("prompt", json.dumps(wf("modelA.safetensors", "1girl")))]))
    b.write_bytes(_png([_text_chunk("prompt", json.dumps(wf("modelB.safetensors", "1girl, sunset")))]))
    v = adjudicate(arm_of(a), arm_of(b), "sunset")
    assert v.code == UNMODELLED and "ckpt_name" in v.reason


def test_a_missing_negative_input_is_not_an_empty_negative():
    """Handing back "" would let an unreadable state compare equal to a
    genuinely empty negative."""
    from attribution_gate.comfyui import negative_prompt
    api = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl"}},
           "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}}}
    with pytest.raises(ComfyUIError):
        negative_prompt(api)


def test_residue_refusal_still_reports_the_factor_problem(tmp_path):
    """Otherwise the user fixes the seed, reruns the render, and is refused a
    second time -- the two-round-trip failure the ordering exists to prevent."""
    a, b = _png_pair(tmp_path, "1girl", "1girl, sunset, night, rain", seed_a=1, seed_b=2)
    v = adjudicate(arm_of(a), arm_of(b), "sunset")
    assert v.code == UNMODELLED
    assert v.candidates == ("night", "rain", "sunset")
    assert "differ by 3 factors" in v.note


def test_a_generator_arm_is_not_consumed_into_nothing():
    def g(xs):
        for x in xs:
            yield x
    assert adjudicate(g(["a", "b"]), g(["a", "b", "c"]), "c").code == VALID


def test_long_values_are_clipped_in_the_middle_not_the_end():
    """Tail truncation printed two different negatives as the same prefix,
    and lopped off the closing quote so the cut was invisible."""
    from attribution_gate import _clip
    a = "worst quality, low quality, bad anatomy, jpeg artifacts, watermark, aaa"
    b = "worst quality, low quality, bad anatomy, jpeg artifacts, watermark, bbb"
    assert _clip(a) != _clip(b)
    assert _clip(a).endswith("'") and _clip(b).endswith("'")
