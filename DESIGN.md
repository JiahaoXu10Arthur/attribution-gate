# Design notes

Why the verdicts are ordered the way they are, what was rejected, and what to
know before changing it. The README says what the package does; this says why.

## The rule the whole thing follows

> **Read a factorization out of the data. Never invent one.**

This is the sentence that settles most arguments here. Where the dialect gives
you a key, use it: `b` and `(b:1.5)` share the name `b`, so a weight change is
one factor at two levels and the parser can *read* that. Where it does not,
merging two things is an **assertion** — and assertions belong to whoever is
making the claim, not to this tool.

## Decision: experiment-level verdicts outrank claim-level ones

`MULTIVARIATE` / `IDENTICAL` / `UNMODELLED` are decided before `CONSTANT` /
`ABSENT`.

An exit code encodes *what to do next*. If the comparison is multivariate then
no single-factor sentence about it holds, so telling the user "you credited the
wrong factor" first sends them off to re-credit and be refused a second time
for the design. Two round trips to learn something one message should have
said.

The demoted diagnosis is not lost — it rides along in `Verdict.note`.

## Decision: exit 1 and exit 2 mean different things

- `1` — the experiment is fine; the sentence about it is not. Rewrite the claim.
- `2` — go back to the bench. No sentence about this comparison is safe.
- `3` — the tool could not run. Not a verdict; nothing was adjudicated.

`IDENTICAL` used to be 1, while its own message told the user to go look at the
seed — which is the exit-2 meaning. The code contradicted itself, so it moved.

`Verdict.exit_code` and `.needs_rerun` live in the **library**, not the CLI. The
mapping was in `__main__.py` once, which meant a library caller could not reach
a distinction the command line already knew — and an agent loop is a library
caller.

An unknown verdict code maps to 2 on purpose: "your experiment is broken" is
the safer wrong answer.

## Decision: a change is one of three things

Added, removed, or **reweighted**. This is the most useful idea in the package
and it came from noticing that both obvious designs fail, in opposite
directions:

- Normalise weights away, and an experiment that varied only a weight is told
  its arms are identical and sent to investigate the seed.
- Treat a weight change as an unrelated removal plus addition, and the same
  experiment is rejected as multivariate.

Both destroy a perfectly good single-variable comparison. So factors carry a
name *and* a value.

## Rejected: inferring substitution

`sunset` → `sunrise` does **not** count as one change, and `plan --swap`
therefore builds a two-factor design that `check` refuses. Two reasons:

1. A reweight is a **parse** — the key is shared, and the dialect's own syntax
   supplies it. A substitution is an **assertion**: calling `sunset` and
   `sunrise` two levels of one factor means naming a factor ("time of day")
   that appears in neither arm.
2. A swap changes the counterfactual. `VALID` on `sunrise` would bless "sunrise
   did it", but the experiment compared sunrise *against sunset*, not against
   its absence. Written down as a finding about `sunrise` alone it fails the
   first time it meets a prompt with no sunset in it.

A tool that inferred a substitution from "one out, one in" would pass
`blue hair, sunset` against `red dress, sunset` as single-variable — and the
hair colour may be doing all the work.

## Decision: PNG mode reads the residue; text mode does not

**Authorship, not availability.** In text mode the caller asserts the arm, and
whatever they left out is by construction outside the comparison. In PNG mode
the *tool* asserts the arm — so it also reads seed, steps, CFG, sampler,
scheduler, denoise, checkpoint, LoRA, resolution and the negative prompt, and
refuses the comparison if any of them moved.

A premise this package manufactured carries a duty a premise it was handed does
not. The same reasoning is why `_read_arm` refuses to treat a missing `.png`
path as prompt text.

Measured justification: across one corpus, of the pairs that differ by exactly
one positive factor — what a clean single-variable comparison looks like from
outside — only one in five was actually controlled. The rest would have passed
at exit 0.

## Rejected: an `ignore=` escape hatch for known-inert factors

The abuse surface is precisely the stated primary use case. In an agent loop
the entity writing the report is the entity writing the ignore list, which is
self-certification rather than an override. And the argument that PNG mode is
unusable without it is weakened by the measurement: PNG mode resolves a small
fraction of a real corpus for other reasons entirely.

## Known weaknesses, all stated in the README

- It enforces [OFAT][ofat]. A two-arm contrast cannot identify an interaction,
  and as an interlock this does not merely tolerate one-factor-at-a-time — it
  converts a workflow to it by sending every two-factor comparison back.
- "A constant cannot explain a variable" is **false** as stated: a factor held
  constant can be load-bearing through an interaction. The verdict text retreats
  to identifiability — this comparison has no contrast on it, so it cannot
  identify its effect — which licenses the same refusal without the false claim.
- In an agent loop the gate is satisfiable by construction: an agent can run
  `diff` first and name whichever factor changed. What survives that is exit 2,
  which is a property of the experiment and cannot be reworded away.

## Before you change anything

**Claims in the README are runnable.** Every console example matches real
output byte for byte. Change the output and you must re-run them. This has been
broken before: a debug print's truncated tail (`repr(t[-160:])` turning
`red eyes` into `yes`) once ended up documented as a literal string the tool
had collected.

**Numbers are censuses, not samples.** An earlier version sampled 300 files and
reported 89/6/4 where the census said 85.6/8.0/4.9 — and the census took
seconds.

**Two fixture kinds.** Synthetic graphs prove the walker handles what it was
written for. `tests/fixtures/real_workflows.json` — six real graphs, one per
outcome the corpus produces, structure untouched and names scrubbed — proves it
still refuses what it used to refuse and still resolves what it used to
resolve. Putting `TriggerWord Toggle` back on the traversal allowlist turns
those red; the synthetic ones do not notice.

**The chunk layer is tested separately.** Graph fixtures start from parsed JSON
and never exercise the PNG walker. `_realistic_png()` mirrors the layout real
files have — IHDR, one or two `tEXt` chunks keyed `prompt` and `workflow`,
several IDAT, IEND — because the older synthetic PNGs had no IDAT to step over.

**Depth is charged once per graph hop.** `_resolve` and `_node_strings` both
incremented it once, which halved the budget: the limit said 24 and refused at
13. If you touch the walk, re-check the boundary against the number the error
message states.

**Zero third-party dependencies is a hard constraint.** PNG chunk parsing is
hand-written to avoid Pillow.

```console
python -I -c "
import sys; sys.path.insert(0, '.')
before = set(sys.modules)
import attribution_gate, attribution_gate.comfyui
new = [m for m in set(sys.modules) - before
       if 'site-packages' in str(getattr(sys.modules[m], '__file__', '') or '')]
print('third-party:', new or 'none')"
```

[ofat]: https://en.wikipedia.org/wiki/One-factor-at-a-time_method
