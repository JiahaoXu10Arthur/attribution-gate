"""Command line entry point.

Exit codes are the product, not the printed text::

    0   the claim is well-formed
    1   the experiment is fine; the sentence about it is not
    2   go back to the bench -- no sentence about this comparison is safe
    3   usage or input error

That split exists so a caller can tell "you credited the wrong thing" (1)
apart from "your experiment cannot answer this" (2). The first is a mistake in
the write-up; the second is a mistake in the design, and needs a rerun. The
mapping lives in the library (``Verdict.exit_code``), not here, so a Python
caller can reach the same distinction.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import PlanError, adjudicate, changes, difference, plan

USAGE = """\
attribution-gate -- refuse a causal claim two arms cannot support

  attribution-gate check ARM_A ARM_B CLAIM
      Adjudicate CLAIM. ARM_A/ARM_B are comma-separated factors, or paths to
      ComfyUI PNGs whose embedded workflow is read instead.

  attribution-gate diff ARM_A ARM_B
      Show what actually differs, without adjudicating anything.

  attribution-gate plan BASE --add FACTOR
  attribution-gate plan BASE --swap OLD=NEW
      Build a second arm that differs by exactly one factor, or refuse.

Exit codes: 0 valid, 1 bad claim, 2 bad experiment, 3 usage error.
Installed as both `attribution-gate` and the shorter `attrgate`.
"""

def _fail(msg: str, *extra: str) -> "int":
    print("error: %s" % msg, file=sys.stderr)
    for line in extra:
        print("       %s" % line, file=sys.stderr)
    return 3


def _read_arm(text: str) -> str:
    """Treat an argument ending in ``.png`` as a file, never as a prompt.

    An earlier version fell back to treating it as prompt text when the file
    was missing, so a mistyped path was silently adjudicated as though
    ``arm_b.png`` were a tag. Anything that looks like a filename and is not
    one is an error, not an arm.
    """
    p = Path(text)
    if p.suffix.lower() != ".png":
        return text
    if not p.exists():
        raise FileNotFoundError(
            "%s does not exist. Arguments ending in .png are read as images, "
            "never as prompt text." % text)
    from .comfyui import arm_of
    return arm_of(p)


def _cmd_check(args) -> int:
    if len(args) < 3:
        return _fail("check needs ARM_A ARM_B CLAIM")
    a, b, claim = _read_arm(args[0]), _read_arm(args[1]), " ".join(args[2:])
    v = adjudicate(a, b, claim)

    print("claim: %s" % v.claim)
    print("%s -- %s" % (v.code, v.reason))
    if v.ok:
        print()
        print("Well-formed is not true. Before this becomes a finding, it "
              "still needs")
        print("a result that a human checked, and enough samples that the "
              "difference")
        print("is not noise. This tool measured neither.")
        # The caveat belongs *here* most of all. Every other verdict already
        # stops the user; this is the one path that ends in a green exit code
        # and a finding getting written down, and it was the only one not
        # saying what it had not looked at.
        print()
        print("Compared: the factor names and weights in the two arms.")
        print("Not compared: anything else -- and for text arms this tool "
              "cannot even")
        print("see what else there is.")
        if v.note:
            print(v.note)
        return v.exit_code

    if v.candidates:
        print()
        print("factors that actually changed (%d): %s"
              % (len(v.candidates), ", ".join(v.candidates)))
    if v.note:
        print(v.note)
    return v.exit_code


def _cmd_diff(args) -> int:
    if len(args) < 2:
        return _fail("diff needs ARM_A ARM_B")
    a, b = _read_arm(args[0]), _read_arm(args[1])
    added, removed, reweighted = changes(a, b)
    print("added in B  (%d): %s" % (len(added), ", ".join(added) or "-"))
    print("removed in B(%d): %s" % (len(removed), ", ".join(removed) or "-"))
    print("reweighted  (%d): %s"
          % (len(reweighted), ", ".join(reweighted) or "-"))
    n = len(set(added) | set(removed) | set(reweighted))
    print("changed factors: %d%s"
          % (n, "" if n == 1 else "  (not a single-variable comparison)"))
    return 0


def _cmd_plan(args) -> int:
    if not args:
        return _fail("plan needs BASE and one of --add / --swap")
    base, add, swap = args[0], None, None
    i = 1
    while i < len(args):
        if args[i] == "--add" and i + 1 < len(args):
            add = args[i + 1]
            i += 2
        elif args[i] == "--swap" and i + 1 < len(args):
            swap = args[i + 1]
            i += 2
        else:
            return _fail("unrecognised argument %r" % args[i])
    try:
        arm_b = plan(base, add=add, swap=swap)
    except PlanError as e:
        return _fail(str(e))
    only_a, only_b = difference(base, arm_b)
    changed = sorted(set(only_a) | set(only_b))
    print(arm_b)
    print()
    print("differs by: %s" % ", ".join(changed), file=sys.stderr)
    if add is not None:
        print("run both arms with the same seed, then come back with:",
              file=sys.stderr)
        print("  attribution-gate check A.png B.png %r" % add, file=sys.stderr)
    else:
        print("this is a TWO-factor design, and `check` will refuse it "
              "(exit 2).", file=sys.stderr)
        print("A swap replaces one factor with another, so the comparison "
              "tests", file=sys.stderr)
        print("  %r against %r -- not either one against its absence."
              % tuple(swap.split("=", 1)), file=sys.stderr)
        print("Calling that one factor would mean asserting they are two "
              "levels of", file=sys.stderr)
        print("one thing, which is domain knowledge this tool does not have. "
              "Score", file=sys.stderr)
        print("it with something that models factorial designs.",
              file=sys.stderr)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, args = argv[0], argv[1:]
    handlers = {"check": _cmd_check, "diff": _cmd_diff, "plan": _cmd_plan}
    if cmd not in handlers:
        return _fail("unknown command %r" % cmd, "try --help")

    from .comfyui import ComfyUIError
    try:
        return handlers[cmd](args)
    except (ValueError, OSError, ComfyUIError) as e:
        # Everything the tool knows how to be told: a malformed claim, a
        # missing file, a workflow it will not guess at.
        return _fail(str(e))
    # Anything else is a defect in this program, and it must not be dressed up
    # as exit 3 "usage error". A CI job would keep going, green, against a
    # gate that had stopped evaluating anything. Let it raise a traceback.


if __name__ == "__main__":
    sys.exit(main())
