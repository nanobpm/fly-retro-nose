"""Turn a retro into PN activity: the sensory transduction layer.

A retro's free-text learnings/deviations are embedded with a small
sentence-transformer and fused with a handful of structured signals
(decomposition deltas, round counts). The fused vector is projected through a
fixed, seeded random projection onto the connectome's 296 PN channels
(a Johnson-Lindenstrauss map that preserves relative distances), giving the
"odor" the mushroom body then hashes.

Text embeddings make smells transfer across repos: two epics that describe the
same architectural problem in different words land near each other in PN space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass
class RetroRecord:
    """The material nano-workforce already emits per epic (see app/retro.ts)."""

    repo: str
    epic: str
    learnings: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    summary: str = ""
    # decomposition deltas from the RetroDigest
    tasks_total: int = 0
    tasks_skipped: int = 0
    tasks_blocked: int = 0
    rounds: int = 0

    def text(self) -> str:
        parts = self.learnings + self.deviations
        if self.summary:
            parts.append(self.summary)
        return "\n".join(p for p in parts if p.strip())

    def structured(self) -> np.ndarray:
        total = max(self.tasks_total, 1)
        return np.array(
            [
                self.tasks_skipped / total,
                self.tasks_blocked / total,
                np.tanh(len(self.learnings) / 5.0),
                np.tanh(len(self.deviations) / 5.0),
                np.tanh(self.rounds / 4.0),
            ],
            dtype=np.float32,
        )


class RetroFeaturizer:
    def __init__(
        self,
        n_pn: int,
        model_name: str = "all-MiniLM-L6-v2",
        structured_gain: float = 2.0,
        seed: int = 1234,
    ) -> None:
        self.n_pn = n_pn
        self.model_name = model_name
        self.structured_gain = structured_gain
        self.seed = seed

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    @cached_property
    def _projection(self) -> np.ndarray:
        # feature dim = embedding dim + structured dim, known after first embed
        feat_dim = self._model.get_sentence_embedding_dimension() + 5
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, 1.0, size=(self.n_pn, feat_dim)).astype(np.float32)

    def _feature_vector(self, record: RetroRecord) -> np.ndarray:
        emb = self._model.encode([record.text()], normalize_embeddings=True)[0]
        struct = record.structured() * self.structured_gain
        return np.concatenate([emb.astype(np.float32), struct])

    def pn_activity(self, record: RetroRecord) -> np.ndarray:
        return self._projection @ self._feature_vector(record)

    def pn_activity_batch(self, records: list[RetroRecord]) -> np.ndarray:
        return np.stack([self.pn_activity(r) for r in records])
