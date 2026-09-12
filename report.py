"""Generate the synthesized architectural-patterns report from real retros.

Usage: python report.py [path/to/app.db]
Writes reports/architectural-patterns.md and prints a summary.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from flyretro.distill import distill_family, synthesize_report
from flyretro.llm import llm_configured
from flyretro.nwf_ingest import DEFAULT_DB
from flyretro.pipeline import build

OUT = os.path.join(os.path.dirname(__file__), "reports", "architectural-patterns.md")


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    pipe = build(db)
    n_repos = len({r.repo for r in pipe.records})
    small = "local small LLM" if llm_configured() else "heuristic fallback (no small LLM)"
    frontier = "frontier LLM" if llm_configured("frontier") else "off"

    lines: list[str] = []
    lines.append("# Architectural patterns across our repos")
    lines.append("")
    lines.append(f"_Synthesized by the male-CNS mushroom body (FlyHash) from "
                 f"{len(pipe.records)} retros across {n_repos} repos. "
                 f"Per-family guidance via {small}; deep synthesis: {frontier}. "
                 f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}._")
    lines.append("")
    if not pipe.families:
        lines.append("No multi-member pattern families yet — the corpus is still "
                     "too small or too diverse to cluster. More retros will resolve them.")

    guidelines: list[str] = []
    family_blocks: list[str] = []
    for fam in pipe.families:
        guidance, source = distill_family(fam)
        guidelines.append(guidance)
        scope = "cross-repo" if fam.is_cross_repo() else "single-repo"
        family_blocks.append(f"## {fam.label} ({scope})")
        family_blocks.append("")
        family_blocks.append(f"- **Repos:** {', '.join(fam.repos)}")
        family_blocks.append(f"- **Epics:** {', '.join(m['epic'] for m in fam.members)}")
        family_blocks.append(f"- **Themes:** {', '.join(fam.keywords[:8])}")
        family_blocks.append(f"- **Guidance ({source}):** {guidance}")
        family_blocks.append("")
        family_blocks.append("<details><summary>Member lessons</summary>")
        family_blocks.append("")
        for m in fam.members:
            for l in (m.get("learnings") or [])[:3]:
                family_blocks.append(f"- `{m['repo']}{m['epic']}` — {l}")
        family_blocks.append("")
        family_blocks.append("</details>")
        family_blocks.append("")

    # Frontier-tier escalation: a cross-cutting synthesis over all families.
    meta = synthesize_report(pipe.families, guidelines)
    if meta is not None:
        narrative, _ = meta
        lines.append("## Cross-cutting synthesis (frontier)")
        lines.append("")
        lines.append(narrative)
        lines.append("")

    lines.extend(family_blocks)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    print(f"  {len(pipe.records)} retros, {n_repos} repos, {len(pipe.families)} families; "
          f"per-family via {small}; frontier synthesis: {frontier}")


if __name__ == "__main__":
    main()
