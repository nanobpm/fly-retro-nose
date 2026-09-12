"""Distil a pattern family's raw retro lessons into one structural guideline.

Prefers a local LLM (llama.cpp) when configured; otherwise degrades to a
deterministic heuristic (an existing retro summary, or a keyword template).
Every result is tagged with its source so callers can be transparent.
"""

from __future__ import annotations

from .llm import complete, llm_configured
from .synthesize import PatternFamily

_SYSTEM = (
    "You are a pragmatic software architect. Given recurring engineering retro "
    "lessons seen across multiple repositories, output exactly ONE concise, "
    "imperative structural guideline (max 30 words) that would prevent the "
    "pattern if applied when structuring a new project. No preamble, no list."
)


def _fallback(family: PatternFamily) -> str:
    for g in family.guidance:
        if g and g.strip():
            return g.strip()
    if family.keywords:
        kws = ", ".join(family.keywords[:5])
        return (
            f"Recurring pattern around: {kws}. Address these areas explicitly "
            f"when structuring similar work."
        )
    return "Recurring pattern with no distilled guidance yet."


def distill_family(family: PatternFamily) -> tuple[str, str]:
    """Return (guideline, source) where source is 'llm' or 'fallback'."""
    if not llm_configured():
        return _fallback(family), "fallback"
    learnings = [l for m in family.members for l in (m.get("learnings") or [])]
    if not learnings:
        return _fallback(family), "fallback"
    prompt = (
        f"Repositories affected: {', '.join(family.repos)}\n"
        f"Recurring lessons:\n"
        + "\n".join(f"- {l}" for l in learnings[:12])
        + "\n\nOne structural guideline:"
    )
    out = complete(prompt, system=_SYSTEM)
    if out:
        return out, "llm"
    return _fallback(family), "fallback"


_PLAN_SYSTEM = (
    "You are a pragmatic software architect advising a project planner BEFORE "
    "work starts. Given a NEW plan and recurring retro lessons from structurally "
    "similar past work, output 2-4 short imperative bullets telling the planner "
    "how to structure THIS plan to avoid repeating those lessons. Be specific to "
    "the new plan. No preamble."
)


def phrase_plan_advice(family: PatternFamily, plan_text: str) -> tuple[str, str]:
    """Tailor a matched family's lessons to a specific new plan.

    Small tier does the phrasing; falls back to the family's static guideline.
    Returns (advice_text, source) where source is 'llm' or 'fallback'.
    """
    if not llm_configured():
        return _fallback(family), "fallback"
    learnings = [l for m in family.members for l in (m.get("learnings") or [])]
    if not learnings:
        return _fallback(family), "fallback"
    prompt = (
        f"NEW PLAN:\n{plan_text.strip()}\n\n"
        f"Similar past work spanned: {', '.join(family.repos)}\n"
        f"Recurring lessons from that work:\n"
        + "\n".join(f"- {l}" for l in learnings[:12])
        + "\n\nHow to structure the new plan up front:"
    )
    out = complete(prompt, system=_PLAN_SYSTEM, max_tokens=260)
    if out:
        return out, "llm"
    return _fallback(family), "fallback"


_META_SYSTEM = (
    "You are a principal engineer writing a cross-repository architecture review. "
    "Given several already-clustered pattern families (each a recurring class of "
    "retro lessons across repos), write a concise synthesis (max ~200 words) that "
    "names the deeper structural forces at work, how the families relate, and the "
    "single highest-leverage change to how projects are structured. Markdown prose."
)


def synthesize_report(families: list[PatternFamily],
                      guidelines: list[str]) -> tuple[str, str] | None:
    """Optional frontier-tier meta-synthesis across families.

    Returns (narrative, 'frontier') when the frontier tier is configured and
    responds; otherwise None so the report simply omits the section.
    """
    if not llm_configured("frontier") or not families:
        return None
    blocks = []
    for fam, g in zip(families, guidelines):
        themes = ", ".join(fam.keywords[:6])
        blocks.append(
            f"Family {fam.label} (repos: {', '.join(fam.repos)}; themes: {themes})\n"
            f"  distilled guideline: {g}"
        )
    prompt = "Pattern families:\n\n" + "\n\n".join(blocks) + "\n\nSynthesis:"
    out = complete(prompt, system=_META_SYSTEM, max_tokens=2000, tier="frontier")
    if out:
        return out, "frontier"
    return None
