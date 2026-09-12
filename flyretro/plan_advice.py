"""Plan-time prophylaxis: match a new plan/spec to a known pattern family and
emit guidance formatted for the senior planner's appendPrompt.

This is the forward-feeding step -- the same seam app/retro.ts already uses to
map digests onto an agent's appendPrompt, but pointed at the *planning* stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distill import phrase_plan_advice
from .pipeline import Pipeline
from .synthesize import retrieve


@dataclass
class PlanAdvice:
    matched: bool
    overlap: float
    repos: list[str]
    keywords: list[str]
    guidance: str
    source: str  # 'llm' | 'fallback' | 'none'

    def append_prompt(self) -> str:
        if not self.matched:
            return (
                "## Architectural prior (fly-retro-nose)\n"
                "No prior architectural pattern resembles this plan — it looks "
                "structurally novel. Proceed, and the retro will teach the fly.\n"
            )
        return (
            "## Architectural prior (fly-retro-nose)\n"
            f"This plan resembles a pattern seen across {', '.join(self.repos)} "
            f"(match {self.overlap:.0%}; themes: {', '.join(self.keywords[:5])}).\n\n"
            f"**Structure the work to honour this up front:** {self.guidance}\n"
        )


def advise_plan(pipe: Pipeline, text: str, repo: str = "(new)", min_overlap: float = 0.12) -> PlanAdvice:
    tag = pipe.encode_text(text, repo=repo)
    fam, ov = retrieve(pipe.families, tag)
    if fam is None or ov < min_overlap:
        return PlanAdvice(False, ov, [], [], "", "none")
    guidance, source = phrase_plan_advice(fam, text)
    return PlanAdvice(True, ov, fam.repos, fam.keywords, guidance, source)
