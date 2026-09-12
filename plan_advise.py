"""Plan-time advisor CLI: feed a new plan/spec, get the planner appendPrompt.

Usage:
  python plan_advise.py "we will use a shared global registry mutated by handlers"
  echo "spec text..." | python plan_advise.py --repo my/repo
"""

from __future__ import annotations

import sys

from flyretro.nwf_ingest import DEFAULT_DB
from flyretro.pipeline import build
from flyretro.plan_advice import advise_plan


def main() -> None:
    args = [a for a in sys.argv[1:]]
    repo = "(new)"
    db = DEFAULT_DB
    if "--repo" in args:
        i = args.index("--repo")
        repo = args[i + 1]
        del args[i:i + 2]
    text = " ".join(args).strip() or sys.stdin.read().strip()
    if not text:
        print("provide plan/spec text as an argument or on stdin"); return

    pipe = build(db)
    advice = advise_plan(pipe, text, repo=repo)
    print("=" * 72)
    print(advice.append_prompt())
    print("=" * 72)
    print(f"[matched={advice.matched} overlap={advice.overlap:.2f} source={advice.source}]")


if __name__ == "__main__":
    main()
