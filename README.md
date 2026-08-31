# attribution-gate

Refuse a causal claim that the two compared arms cannot support.

You ran a two-arm comparison, looked at the results, and wrote down *why*.
This takes that "why" as **input** and tries to reject it. The exit code is the
product.

```console
$ attribution-gate check "1girl, silhouetted figure, sunset" \
                        "1girl, silhouetted figure, sunset, rim lighting" \
                        "silhouetted figure"
claim: silhouetted figure
CONSTANT -- 'silhouetted figure' is at the same level in both arms, so this comparison carries no information about it -- there is no contrast to read an effect from. If you meant it was a necessary background condition for the factor that did change, that is an interaction claim, and a two-arm design cannot test it: you need a 2x2.

factors that actually changed (1): rim lighting
$ echo $?
1
```

Arms are comma-separated factors, or any iterable of strings — prompts are one
instance, not the domain. Config keys, feature flags and hyperparameters work
the same way.

```python
from attribution_gate import adjudicate

v = adjudicate(arm_a, arm_b, "rim lighting")
if v.needs_rerun:                 # the experiment cannot answer this at all
    raise SystemExit(v.reason)
if not v:                         # the experiment is fine, the claim is not
    print("re-credit to one of:", ", ".join(v.candidates))
```

No dependencies, stdlib only. 82 tests, CI on Python 3.9, 3.11 and 3.13.

## Why this exists

A comparison was run to find out what had removed a stray extra figure from an
image. The effect was credited to a `silhouetted figure` tag.

That tag was in **both** arms. It could not have been the cause.

The finding was one step from being written down permanently, and every later
experiment would have been built on it. What makes that worth a tool is not
that the mistake was subtle — it is that catching it needs no second opinion.
It is a set difference. It went uncaught because checking it was nobody's job.

That is not the same as "no judgement anywhere". The judgement moved into the
factorization: that an arm splits on commas, that case and surrounding
whitespace do not matter, that `(tag:1.3)` is one factor at a level, and that
everything outside the list — seed, sampler, negative prompt — is out of
scope. Those are assumptions, they are this package's assumptions, and they
are wrong for some dialects. What the *rule* then does with them needs no
judgement at all, and that is the part worth mechanising.

## In CI, and in agent loops

That is where a gate earns its keep. Anyone who installs one interactively
already knows to fix the seed and change one thing. The case that needs a
machine is the one where a machine writes the experiment report and something
has to decide whether the report may be believed.

```yaml
# .github/workflows/findings.yml
- run: |
    while IFS=$'\t' read -r arm_a arm_b claim; do
      attribution-gate check "$arm_a" "$arm_b" "$claim" || exit $?
    done < findings.tsv
```

One thing to be clear about, since it undercuts the pitch: **an agent that
writes both the report and the claim can satisfy this gate trivially.** It can
run `diff` first, name whichever factor actually changed, and pass every time,
regardless of what caused anything. The gate constrains a claim's *form*, and
form is the cheapest thing for a language model to fix.

What survives that is the half the claimant cannot talk its way out of: exit 2.
A multivariate or unmodelled comparison is a property of the experiment, not of
the sentence, and no amount of rewording clears it. Deploy this to catch the
honest error and to block indefensible designs — not as an adversarial control
over a claimant that also picks the claim.

Exit 1 and exit 2 mean different things and deserve different handling:

| verdict | meaning | exit |
|---|---|---|
| `VALID` | arms differ by exactly one factor, and it is the claimed one | 0 |
| `CONSTANT` | the claimed factor is at the same level in both arms | 1 |
| `ABSENT` | the claimed factor is in neither arm | 1 |
| `IDENTICAL` | nothing in the modelled space distinguishes the arms | 2 |
| `MULTIVARIATE` | arms differ by more than one factor | 2 |
| `UNMODELLED` | a difference, or an arm, that this tool does not model | 2 |

**1 — the experiment is fine, the sentence about it is not.** Rewrite the
claim; the refusal prints the factors that did change, which is the honest
answer to "then what could it have been?"

**2 — go back to the bench. No sentence about this comparison is safe.**

**3 — the tool could not run.** A malformed claim, a missing file, a workflow
it will not guess at. Not a verdict: nothing was adjudicated, so the `|| exit
$?` above will propagate it and your job stops, which is the intended
behaviour. An internal defect is *not* mapped to 3 — it raises, so a broken
gate cannot pass itself off as your mistake.

Experiment-level verdicts outrank claim-level ones, so a multivariate
comparison reports as multivariate even when the claimed factor is also
constant — the constant diagnosis is carried along as a note. Ordering it the
other way sent you off to re-credit the effect, only to be refused again for
the design: two round trips to learn what one should have said.

`Verdict.exit_code` and `Verdict.needs_rerun` expose the same split to a
library caller. `bool(v)` alone is a safe gate — every refusal is falsy — but
it collapses the two.

## A change is one of three things

Added, removed, or **reweighted**. The third matters more than it looks.

Attention weights (`(blue hair:1.3)`) are easy to normalise away so that
`blue hair` and `(blue hair:1.3)` count as one factor. Do that and an
experiment which varied only a weight is told its arms are identical, and sent
off to investigate the seed. Treat the weight change as an unrelated removal
plus addition instead, and the same experiment is rejected as multivariate.
Both readings destroy a good single-variable comparison, in opposite
directions.

```console
$ attribution-gate diff "a, b" "a, (b:1.5), d"
added in B  (1): d
removed in B(0): -
reweighted  (1): b
changed factors: 2  (not a single-variable comparison)
```

### The line: read a factorization, never invent one

`b` and `(b:1.5)` share a key, and that key comes from the prompt dialect's own
syntax. The parser *reads* that they are one factor at two levels. Nothing is
inferred.

`sunset` and `sunrise` share no key. Calling them two levels of one factor
means asserting a factor — "time of day" — that appears in neither arm. So
`plan --swap` builds a **two-factor design and says so**, and `check` refuses
it. That is deliberate, and it is why:

- One out, one in is usually two independent edits. A tool that inferred a
  substitution from that shape would pass `blue hair, sunset` against
  `red dress, sunset` as single-variable — and the hair colour may be doing all
  the work. This one reports `MULTIVARIATE`.
- A swap changes the counterfactual. `VALID` on `sunrise` would bless "sunrise
  did it", but the experiment compared sunrise *against sunset*, not against
  its absence. Written down as a finding about `sunrise` alone it fails the
  first time it meets a prompt with no sunset in it.

## What it does not do

**It is not a statistics package.** It inspects the *structure* of a
comparison, never the *strength* of a result. One sample per arm proves nothing
regardless of what this returns. `VALID` means your claim is well-formed; it
never means your claim is true, and the tool says so on every pass.

**In text mode it sees the factor list you hand it and nothing else** — not
the seed, the sampler, the step count, the CFG, or the negative prompt. A
`VALID` verdict asserts nothing about any of them. You authored the arm, so
whatever you left out is by construction outside the comparison.

**In PNG mode the tool authors the arm, and then it owes you more.** It reads
the seed, steps, CFG, sampler, scheduler, denoise and the negative prompt out
of the same workflow, and refuses the comparison if any of them moved. A
premise this package manufactured carries a duty a premise it was handed does
not. See below for what that is worth in practice.

Token **order** and **repetition** are not modelled either, but they are not
silently ignored: arms that differ only that way return `UNMODELLED` rather
than a confident `IDENTICAL` pointing you at the seed. A change count is the
whole product, and there is no honest count for "moved one token past four
others".

## Prior art

What is unoccupied is narrow: **taking the causal claim itself as input and
refusing it**. Every comparison tool I have looked at is a renderer — A1111's
X/Y/Z plot, ComfyUI's XY Plot, promptfoo's matrix, W&B run comparison — it
produces a grid and hands interpretation to a human. This is the interlock for
the moment the human writes the interpretation down.

Almost everything else already exists:

- The rule is [Mill's Method of Difference][mill], 1843. This is its crudest
  special case, and Mill's precondition — every circumstance in common save one
  — is exactly what the tool cannot verify.
- **A1111's Prompt S/R** already refuses one of these cases: `xyz_grid.py`
  raises when the search token is not in the prompt, which is `ABSENT`. It does
  not check duplicates, case collisions, or how many factors differ.
- **[Crystools][crystools]** diffs two ComfyUI metadata blobs — the
  set-difference step without a gate on top.
- **Reading config out of embedded PNG metadata** is done by a dozen tools.
  Table stakes, not a feature.
- **Sample Ratio Mismatch** is the established mechanical "this experiment is
  invalid" check in A/B platforms. It validates *assignment*; this validates
  *what differs*. Nothing here competes with it.

## Known limitation: this enforces OFAT

Requiring exactly one factor to differ is [one-factor-at-a-time][ofat], which
design-of-experiments literature has argued against for decades. A single
two-arm contrast cannot identify an interaction at all, and this gate will
happily call a main-effect claim `VALID` under a model that may be pure
interaction. Worse, as an interlock it does not merely tolerate OFAT — it
converts a workflow to it, by sending every two-factor comparison back for a
rerun.

Use a factorial design when you need interactions. This is for the case I kept
hitting: two variants were run and a reason was written down.

## Reading arms from ComfyUI PNGs

Pass `.png` paths and the embedded API workflow is read instead, so the arm
adjudicated is the arm that ran. Resolution starts at the sampler's `positive`
input and walks back, because the prompt is often not typed into the encoder;
where a switch chose a branch, only the branch that ran is followed.

**Most of the time it refuses.** Traversal is an allowlist of string nodes
whose output follows from their inputs; anything else stops it, naming the
node. Two common node kinds assemble the prompt at run time and leave nothing
in the file — LoRA managers injecting trigger words read from LoRA metadata,
and prompt upsamplers generating text with a language model. Returning the
fraction that *is* in the graph would compare two arms on part of their factors
and then state a verdict about it. An early version did exactly that, reporting
one render's "prompt" as `long`, `short` and
`KBlueLeaf/TIPO-500M-ft | TIPO-500M-ft-F16.gguf`.

Every PNG in one heavily-scripted setup, 1,608 renders, no sampling:

| outcome | count | share |
|---|---|---|
| refused — LoRA trigger words injected at run time | 1377 | 85.6% |
| resolved | 129 | 8.0% |
| refused — no sampler node in the workflow at all | 47 | 2.9% |
| refused — prompt upsampler | 27 | 1.7% |
| refused — img2img: the arm is an image, not text | 23 | 1.4% |
| refused — no embedded workflow | 5 | 0.3% |

That is one author's worst case, not a general rate — this setup drives
everything through a LoRA manager, and a workflow that types its prompt into
the encoder resolves fine. But the shape generalises: if your prompts are
assembled at run time, pass the arms as text. You have them, and the tool will
not pretend to.

### What the workflow check is worth

Of the 129 renders that do resolve, **every one** also yields its negative
prompt and its seed. The seed is worth reading carefully: across the 129, the
sampler's `seed` is a link to a seed node 83 times and a plain literal 46, so a
reader that only understood literals would miss the single most important
controlled variable on two thirds of the comparisons it can make — silently.
(Counting every sampler node in the corpus, including the workflows this tool
refuses, it is 1,916 links to 99 literals.)

Then the number that decided this feature. Across all pairs of those 129
renders, exactly **5** differ by precisely one positive factor, which is what a
clean single-variable comparison looks like from the outside. Of those five,
**one** is actually controlled. The other four differ in CFG or in the negative
prompt as well, and every one of them would have been stamped `VALID`.

An 80% false-pass rate, on the only path that ends in a green exit code.

## Install

Not on PyPI. From a clone:

```console
pip install .          # or -e ".[test]" to run the suite
pytest -q
```

Installs as `attribution-gate` and the shorter `attrgate`. Pure stdlib, so
there is nothing else to resolve.

## License

MIT

[mill]: https://en.wikipedia.org/wiki/Mill%27s_methods
[ofat]: https://en.wikipedia.org/wiki/One-factor-at-a-time_method
[crystools]: https://github.com/crystian/ComfyUI-Crystools
