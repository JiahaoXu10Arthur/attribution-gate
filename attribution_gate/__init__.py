"""Refuse a causal claim that the two compared arms cannot support.

You ran a two-arm comparison -- two prompts, two configs, two anything --
looked at the results, and wrote down *why*: "the ``rim lighting`` tag caused
it." This library takes that claim as **input** and tries to reject it.

Four refusals, all decidable by set arithmetic alone:

- the claimed factor is in **both** arms, unchanged -- a constant cannot
  explain a variable
- the claimed factor is in **neither** arm
- the arms differ by **more than one** factor -- the comparison cannot isolate
  anything, so no single factor may be credited
- the arms carry the same factors at the same weights, so nothing this tool
  models distinguishes them at all

Why this exists
---------------
This is a real bug that produced a real false conclusion. A comparison was run
to find out what removed a stray extra figure from an image, and the effect
was credited to a ``silhouetted figure`` tag. That tag was present in **both**
arms. It could not have been the cause. The rule it violated needs no
judgement and no domain knowledge to catch -- only a set difference -- and it
had gone uncaught long enough to nearly be written down as a permanent
finding.

That is the shape this library targets: not conclusions that are hard to
check, but conclusions that are *trivially* checkable and never checked,
because checking them is nobody's job.

What this is not
----------------
**Not a statistics package.** It inspects the *structure* of a comparison,
never the *strength* of a result. It knows nothing about sample size,
variance, or effect size -- one sample per arm proves nothing regardless of
what this returns. A ``VALID`` verdict means *your claim is well-formed*. It
never means *your claim is true*.

The rule is not new. It is the crudest special case of Mill's Method of
Difference, published in 1843. The contribution here is that it exits nonzero.

Mill's precondition -- every circumstance in common save one -- is precisely
what this cannot verify. It sees the factor list it is handed and nothing
else: not the seed, not the sampler, not the step count, not the negative
prompt. A ``VALID`` verdict asserts nothing about any of them.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Set, Tuple, Union

__all__ = [
    "factors", "weights", "difference", "changes", "adjudicate", "plan",
    "Verdict", "PlanError",
    "VALID", "CONSTANT", "ABSENT", "IDENTICAL", "MULTIVARIATE", "UNMODELLED",
]
__version__ = "0.1.0"

#: The claim is well-formed: the arms differ by exactly one factor, and it is
#: the claimed one. Says nothing about whether the claim is *true*.
VALID = "VALID"

#: The claimed factor is at the same level in both arms, so the comparison has
#: zero contrast on it and cannot identify its effect. Note the careful
#: wording: *this experiment is silent about it*, not *it is causally inert*.
#: A factor held constant can still be load-bearing through an interaction --
#: an INUS condition -- and saying otherwise would be false.
CONSTANT = "CONSTANT"

#: The claimed factor appears in neither arm.
ABSENT = "ABSENT"

#: Same factor names, same weights, same order. Whatever was observed came
#: from outside the modelled space -- negative prompt, seed, sampler, steps --
#: or the two arms are the same run. Not "nothing differs": *nothing this tool
#: can see* differs, which is a smaller claim and the only one it may make.
IDENTICAL = "IDENTICAL"

#: The arms differ by more than one factor. The claim may well be right, but
#: this comparison cannot show it: the effect is not attributable to any
#: single factor.
MULTIVARIATE = "MULTIVARIATE"

#: The arms are not the same text, but they differ only in ways this tool does
#: not model -- token order, or a repeated factor. Refusing is the honest
#: answer: a change count is the whole product, and there is no defensible
#: number for "moved one token past four others".
UNMODELLED = "UNMODELLED"

#: What the user must do next, which is what an exit code is for.
#: ``1`` -- the experiment is fine, the sentence about it is not.
#: ``2`` -- go back to the bench; no sentence about this comparison is safe.
_EXIT_CODES = {
    VALID: 0,
    CONSTANT: 1,
    ABSENT: 1,
    IDENTICAL: 2,
    MULTIVARIATE: 2,
    UNMODELLED: 2,
}

Arm = Union[str, Iterable[str]]

#: ``(tag:1.3)`` and ``(tag: 1.3)`` are attention-weight syntax in several
#: prompt dialects. Left in place they would make ``tag`` and ``(tag:1.3)``
#: two distinct factors, and the set difference would report a difference that
#: is not one.
_WEIGHTED = re.compile(r"^\((.*):\s*(-?[\d.]+)\s*\)$")


class PlanError(ValueError):
    """A proposed second arm would not be the comparison that was asked for."""


class Verdict:
    """The result of adjudicating one claim. Falsy unless the claim is valid.

    Two questions, and they are different::

        v = adjudicate(arm_a, arm_b, "rim lighting")
        if v.needs_rerun:            # the *experiment* cannot answer this
            raise SystemExit(v.reason)
        if not v:                    # the experiment is fine, the claim is not
            print("re-credit to one of:", ", ".join(v.candidates))

    ``bool(v)`` alone is a safe gate -- every refusal is falsy -- but it
    collapses "rewrite the sentence" into "rerun the experiment", and for an
    automated caller those imply completely different next actions.
    """

    __slots__ = ("code", "reason", "claim", "candidates", "note")

    def __init__(self, code: str, reason: str, claim: str,
                 candidates: Sequence[str] = (), note: str = ""):
        self.code = code
        self.reason = reason
        self.claim = claim
        #: The factors that actually do differ between the arms. On a
        #: rejection this is the honest answer to "then what could it have
        #: been?"
        self.candidates = tuple(candidates)
        #: A second thing that is also true and worth knowing. A multivariate
        #: comparison whose claimed factor is *also* constant deserves to be
        #: told both, so the next attempt does not just re-credit to another
        #: factor and come back for a second refusal.
        self.note = note

    @property
    def ok(self) -> bool:
        return self.code == VALID

    @property
    def exit_code(self) -> int:
        """The process exit code for this verdict.

        Lives here, not in the CLI, so a library caller can reach the same
        distinction the command line makes. Unknown codes fail toward 2 --
        "your experiment is broken" is the safer wrong answer.
        """
        return _EXIT_CODES.get(self.code, 2)

    @property
    def needs_rerun(self) -> bool:
        """True when no sentence about this comparison would be safe.

        The question an automated caller actually asks. ``not verdict`` says
        the claim failed; this says the *experiment* did.
        """
        return self.exit_code == 2

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return "Verdict(%s, claim=%r, candidates=%r)" % (
            self.code, self.claim, self.candidates)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Verdict):
            return NotImplemented
        return (self.code, self.claim, self.candidates) == (
            other.code, other.claim, other.candidates)

    def __hash__(self) -> int:
        return hash((self.code, self.claim, self.candidates))


def _items(arm: Arm):
    """Split an arm and normalise case and whitespace. One parser, not two.

    Case and stray whitespace are invisible when two arms are read side by
    side, but they turn two spellings of one factor into two factors -- and
    then the arms "differ by two things" and every claim about them is
    rejected for the wrong reason.
    """
    raw = arm.split(",") if isinstance(arm, str) else list(arm)
    for item in raw:
        t = str(item).strip().lower()
        if t:
            yield t


def weights(arm: Arm) -> "dict":
    """Map each factor name to its attention weight, or ``None`` if unweighted.

    ``"a, (b:1.5)"`` becomes ``{"a": None, "b": 1.5}``.
    """
    out = {}
    for t in _items(arm):
        m = _WEIGHTED.match(t)
        if not m:
            out[t] = None
            continue
        name = m.group(1).strip()
        try:
            out[name] = float(m.group(2))
        except ValueError:
            # Something shaped like a weight but not a number. Keep the
            # factor; drop the value rather than the whole item.
            out[name] = None
    return out


def _conflicts(arm: Arm) -> Tuple[str, ...]:
    """Factor names that appear more than once in one arm at different weights.

    ``weights()`` is a dict, so it can only keep one of them, and "last wins"
    would make removing a duplicate look like a weight change. The package
    refuses to discard a change silently; that has to include this one.
    """
    seen = {}
    bad = set()
    for name, w in _sequence(arm):
        if name in seen and seen[name] != w:
            bad.add(name)
        seen[name] = w
    return tuple(sorted(bad))


def _clip(value, width: int = 60) -> str:
    """``repr`` shortened in the middle, never at the end.

    Truncating the tail printed two *different* negative prompts as the same
    60-character prefix -- and lopped off the closing quote, so the reader
    could not even tell it had been cut.
    """
    text = repr(value)
    if len(text) <= width:
        return text
    keep = (width - 5) // 2
    return text[:keep] + " ... " + text[-keep:]


def _sequence(arm: Arm):
    """The arm as an ordered list of ``(name, weight)``, duplicates kept.

    Used only to tell "the arms are genuinely identical" apart from "they
    differ in a way the set model throws away". ``(b:1.5)`` and ``(b:1.50)``
    normalise to the same pair, so a harmless respelling is not mistaken for
    a reordering.
    """
    out = []
    for t in _items(arm):
        m = _WEIGHTED.match(t)
        if not m:
            out.append((t, None))
            continue
        try:
            out.append((m.group(1).strip(), float(m.group(2))))
        except ValueError:
            out.append((m.group(1).strip(), None))
    return out


def factors(arm: Arm, strip_weights: bool = True) -> Set[str]:
    """The set of factor names in one arm.

    Accepts a comma-separated string, or any iterable of strings if the
    factors do not live in a prompt. With ``strip_weights=False`` the items
    are returned verbatim, so ``(b:1.5)`` stays distinct from ``b`` -- rarely
    what you want, since :func:`changes` models weights properly.
    """
    if strip_weights:
        return set(weights(arm))
    return set(_items(arm))


def difference(a: Arm, b: Arm) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Which factor *names* each arm has that the other does not.

    This is name-level only. A factor present in both arms at different
    weights shows up in neither tuple -- see :func:`changes`.
    """
    fa, fb = factors(a), factors(b)
    return tuple(sorted(fa - fb)), tuple(sorted(fb - fa))


def changes(a: Arm, b: Arm) -> Tuple[Tuple[str, ...], Tuple[str, ...],
                                     Tuple[str, ...]]:
    """Every way the arms differ, as ``(added, removed, reweighted)``.

    Three kinds, not two. Treating a weight change as an unrelated
    remove-plus-add would report one edit as two differing factors, and every
    claim about it would then be rejected as multivariate. Treating it as no
    change at all -- which is what dropping weights during normalisation does
    -- is worse: an experiment that varied only a weight gets told its arms
    are identical and to go looking at the seed.
    """
    wa, wb = weights(a), weights(b)
    added = tuple(sorted(set(wb) - set(wa)))
    removed = tuple(sorted(set(wa) - set(wb)))
    reweighted = tuple(sorted(k for k in set(wa) & set(wb)
                              if wa[k] != wb[k]))
    return added, removed, reweighted


def adjudicate(a: Arm, b: Arm, claim: str) -> Verdict:
    """Decide whether ``claim`` may be credited for the difference between arms.

    Returns a :class:`Verdict`. Never raises on an invalid claim -- an invalid
    claim is an ordinary result, not an error.
    """
    # A generator is consumed by the first pass over it, and every later pass
    # would then see an empty arm and report IDENTICAL. Materialise once.
    if not isinstance(a, str) and not hasattr(a, "__len__"):
        a = list(a)
    if not isinstance(b, str) and not hasattr(b, "__len__"):
        b = list(b)

    fa, fb = factors(a), factors(b)
    claim_set = factors(claim)
    if len(claim_set) != 1:
        raise ValueError(
            "a claim must name exactly one factor, got %d from %r"
            % (len(claim_set), claim))
    c = next(iter(claim_set))

    # When the arms were authored by a reader rather than by the caller, that
    # reader also saw what it did not compare. Anything it discarded has to
    # match, or "the arms differ by exactly one factor" is false. This outranks
    # every other verdict: if the two runs were not controlled, nothing about
    # their factors is worth saying.
    _added, _removed, _reweighted = changes(a, b)
    _changed = tuple(sorted(set(_added) | set(_removed) | set(_reweighted)))

    res_a, res_b = getattr(a, "residue", None), getattr(b, "residue", None)
    mixed_note = ""
    if (res_a is None) != (res_b is None):
        # One arm was read from a workflow and the other was typed. The tool
        # knows one run's seed and negative prompt and has nothing to compare
        # them against, so it cannot say the runs were controlled. It can say
        # that it could not say, which is the part that must not be silent.
        which = "A" if res_a is not None else "B"
        mixed_note = ("also: only arm %s carried workflow settings, so the "
                      "seed, CFG and negative prompt were not compared at "
                      "all. Pass both arms as images to have them checked."
                      % which)
    if res_a is not None and res_b is not None:
        off = sorted(k for k in set(res_a) | set(res_b)
                     if res_a.get(k) != res_b.get(k))
        if off:
            detail = "; ".join(
                "%s: %s vs %s" % (k, _clip(res_a.get(k)), _clip(res_b.get(k)))
                for k in off)
            extra = ("" if len(_changed) <= 1 else
                     "also: the arms differ by %d factors (%s), so this "
                     "comparison would not be attributable even once the runs "
                     "are controlled -- fix both before rerunning"
                     % (len(_changed), ", ".join(_changed)))
            return Verdict(
                UNMODELLED,
                "the two runs were not controlled -- they differ in %s, which "
                "this tool read from the workflows but does not model as a "
                "factor. No single factor is isolated by this comparison. "
                "(%s)" % (", ".join(off), detail),
                c, _changed, extra)

    for label, arm in (("A", a), ("B", b)):
        clash = _conflicts(arm)
        if clash:
            return Verdict(
                UNMODELLED,
                "arm %s lists %s more than once at different weights, so it "
                "has no single level for %s and this tool cannot say what "
                "changed. Write each factor once."
                % (label, ", ".join(repr(x) for x in clash),
                   "them" if len(clash) > 1 else "it"),
                c, ())

    added, removed, reweighted = changes(a, b)
    changed = tuple(sorted(set(added) | set(removed) | set(reweighted)))

    # Why the claim failed, for use as a secondary note when the experiment
    # itself is the bigger problem.
    if c in changed:
        claim_note = ""
    elif c in fa and c in fb:
        claim_note = ("also: %r is present in both arms, unchanged -- it "
                      "could not be the cause even in a clean comparison" % c)
    else:
        claim_note = ("also: %r is present in neither arm" % c)

    if mixed_note:
        claim_note = claim_note + "\n" + mixed_note if claim_note else mixed_note

    # Experiment-level verdicts come first. Exit 2 means no sentence about
    # this comparison is safe, which subsumes any complaint about the
    # particular sentence that was offered. Ordering them the other way sends
    # the user off to re-credit the effect, only to be refused again for the
    # design -- two round trips to learn what one should have said.
    if not changed:
        if _sequence(a) != _sequence(b):
            return Verdict(
                UNMODELLED,
                "the arms are not the same text, but they differ only in "
                "token order or in a repeated factor -- things this tool does "
                "not model. It counts changes, and there is no honest count "
                "for those, so it will not adjudicate this comparison",
                c, changed, claim_note)
        return Verdict(
            IDENTICAL,
            "nothing this tool can see distinguishes the arms: same factor "
            "names, same weights. Either the difference lies outside the "
            "modelled space -- negative prompt, seed, sampler, steps, CFG, "
            "token order, repetition -- or the two arms really are the same "
            "run. This tool models the positive prompt's factor names and "
            "their weights, and nothing else",
            c, changed, claim_note)
    if len(changed) > 1:
        return Verdict(
            MULTIVARIATE,
            "the arms differ by %d factors, so the effect is not attributable "
            "to any single one of them" % len(changed),
            c, changed, claim_note)

    # Everything the change model did *not* account for has to be identical,
    # or "the arms differ by exactly one factor" is a false statement. Checking
    # this only when nothing changed would let a reordering ride along
    # unmentioned on the pass path, which is the one path nothing downstream
    # re-checks.
    rest_a = [x for x in _sequence(a) if x[0] not in changed]
    rest_b = [x for x in _sequence(b) if x[0] not in changed]
    if rest_a != rest_b:
        return Verdict(
            UNMODELLED,
            "apart from %r the arms still differ -- in token order or in a "
            "repeated factor, which this tool does not model. It counts "
            "changes and there is no honest count for those, so it will not "
            "credit %r for the difference" % (c, c),
            c, changed, claim_note)

    if c not in changed:
        if c in fa and c in fb:
            return Verdict(
                CONSTANT,
                "%r is at the same level in both arms, so this comparison "
                "carries no information about it -- there is no contrast to "
                "read an effect from. If you meant it was a necessary "
                "background condition for the factor that did change, that is "
                "an interaction claim, and a two-arm design cannot test it: "
                "you need a 2x2." % c,
                c, changed, mixed_note)
        return Verdict(
            ABSENT,
            "%r is not among the factors of either arm, so this comparison "
            "gives no grounds to credit it" % c,
            c, changed, mixed_note)

    how = ("its weight changed" if c in reweighted else
           "it was added" if c in added else "it was removed")
    return Verdict(
        VALID,
        "the arms differ by exactly one factor, %r (%s); the claim is "
        "well-formed (this says nothing about whether it is true)" % (c, how),
        c, changed, mixed_note)


def plan(base: str, add: str = None, swap: str = None) -> str:
    """Build a second arm from ``base`` by making one edit.

    ``add="smile"`` appends a factor. The result differs by one factor and
    :func:`adjudicate` can attribute to it.

    ``swap="old=new"`` replaces one factor with another. **The result differs
    by two factors and this package will not adjudicate it**, by design. That
    is not an oversight; see below. Use ``swap`` to construct the comparison,
    and score it with something that models factorial designs.

    Either way :class:`PlanError` is raised when the edit would not produce
    the comparison that was asked for -- a base that already contains the
    addition, a duplicated factor, or two spellings of one factor. Those
    silently produce the wrong experiment.

    Why a swap is two factors and not one
    -------------------------------------
    A weight change *is* one factor: ``b`` and ``(b:1.5)`` share a key, and
    that key comes from the prompt dialect's own syntax. :func:`weights` reads
    it. Nothing is inferred.

    ``sunset`` and ``sunrise`` share no key. Calling them two levels of one
    factor means asserting a factor -- "time of day" -- that appears in
    neither arm. That assertion is exactly the domain knowledge this package
    refuses to have, and a tool that guessed it would silently convert a
    two-edit comparison into an attributable one. Consider ``blue hair,
    sunset`` against ``red dress, sunset``: one factor out, one in, and the
    hair colour may be doing all the work.

    There is a second reason, and it is the stronger one. A swap changes the
    counterfactual that was tested. ``VALID`` on ``sunrise`` would bless the
    claim "sunrise did it", but the experiment compared sunrise *against
    sunset*, not against its absence. Written down as a finding about
    ``sunrise`` alone, it would fail the first time it met a prompt with no
    sunset in it -- which is precisely the kind of false finding this package
    exists to stop.

    The line, stated once: **this tool may read a factorization out of the
    data, and it will never invent one.**
    """
    if (add is None) == (swap is None):
        raise PlanError("give exactly one of add= or swap=")

    base_factors = factors(base)
    parts = [p.strip() for p in base.split(",") if p.strip()]

    if add is not None:
        new = factors(add)
        if len(new) != 1:
            raise PlanError("add= must name exactly one factor, got %r" % add)
        if next(iter(new)) in base_factors:
            raise PlanError(
                "base already contains %r; adding it would produce two "
                "identical arms" % add)
        arm_b = ", ".join(parts + [add.strip()])
        expected = 1
    else:
        if "=" not in swap:
            raise PlanError("swap= must be written old=new, got %r" % swap)
        old, new = [x.strip() for x in swap.split("=", 1)]
        if not old or not new:
            raise PlanError("swap= must be written old=new, got %r" % swap)
        # Each side must name exactly one factor. Taking the first element of
        # a set would pick by iteration order and return a confident result
        # for `swap="a, b=c"` -- silently swapping whichever of a/b came out
        # first.
        for label, side in (("left", old), ("right", new)):
            n = len(factors(side))
            if n != 1:
                raise PlanError(
                    "the %s side of swap= must name exactly one factor, got "
                    "%d from %r" % (label, n, side))
        old_n = next(iter(factors(old)))
        new_n = next(iter(factors(new)))
        if old_n not in base_factors:
            raise PlanError(
                "base does not contain %r, so it cannot be swapped" % old)
        if new_n in base_factors:
            raise PlanError(
                "base already contains %r; the swap would only remove %r, "
                "which is a one-factor change described as a two-factor one"
                % (new, old))
        arm_b = ", ".join(new.strip() if factors(p) == {old_n} else p
                          for p in parts)
        expected = 2

    only_a, only_b = difference(base, arm_b)
    changed = set(only_a) | set(only_b)
    if len(changed) != expected:
        raise PlanError(
            "this is not a single-variable change: the arms would differ by "
            "%d factors (%s). A duplicated factor in the base, or two "
            "spellings of one factor, is the usual cause."
            % (len(changed), ", ".join(sorted(changed))))
    return arm_b
