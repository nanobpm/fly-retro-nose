"""Associative memory over Kenyon-cell tags: the mushroom body's MBON layer.

Each remembered retro is a sparse KC tag plus provenance (repo, epic, a
human-facing note). Recognition is nearest-tag retrieval by Jaccard overlap;
novelty is one minus the best overlap against everything seen so far. Writes
are online and cheap, so the memory "gets better over time" as retros land.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flyhash import overlap


@dataclass
class Trace:
    tag: np.ndarray  # active KC indices
    meta: dict  # {"repo":..., "epic":..., "note":..., "smell":...}


@dataclass
class Match:
    overlap: float
    meta: dict


@dataclass
class KCMemory:
    traces: list[Trace] = field(default_factory=list)

    def add(self, tag: np.ndarray, meta: dict) -> None:
        self.traces.append(Trace(tag=np.asarray(tag), meta=dict(meta)))

    def query(self, tag: np.ndarray, top_k: int = 5) -> list[Match]:
        tag = np.asarray(tag)
        scored = [Match(overlap(tag, t.tag), t.meta) for t in self.traces]
        scored.sort(key=lambda m: m.overlap, reverse=True)
        return scored[:top_k]

    def novelty(self, tag: np.ndarray) -> float:
        """1.0 = never smelled anything like this; 0.0 = seen it exactly."""
        if not self.traces:
            return 1.0
        best = max(overlap(np.asarray(tag), t.tag) for t in self.traces)
        return 1.0 - best

    def __len__(self) -> int:
        return len(self.traces)
