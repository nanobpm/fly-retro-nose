"""Synthesize accumulated retros into cross-repo architectural patterns.

The mushroom-body memory holds one sparse KC tag per retro. Clustering those
tags (by tag overlap, the fly's own similarity metric) makes recurring patterns
emerge as families that span repos and epics -- the "bigger architectural
patterns" a single retro can't reveal. Each family is distilled into forward
guidance that can prime how a *new* project is structured.

Naming is left to humans (unsupervised, per the design): families come out as
`pattern-N` with auto-extracted keyword hints and their member learnings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .flyhash import overlap
from .memory import KCMemory


@dataclass
class PatternFamily:
    label: str
    members: list[dict] = field(default_factory=list)  # trace metas
    keywords: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    tags: list[np.ndarray] = field(default_factory=list)

    @property
    def repos(self) -> list[str]:
        return sorted({m.get("repo", "?") for m in self.members})

    def is_cross_repo(self) -> bool:
        return len(self.repos) > 1

    def consensus_tag(self) -> np.ndarray:
        """KC indices active in a plurality of the family's tags -- the family
        centroid in the fly's own code space, used for plan-time retrieval."""
        c = Counter()
        for t in self.tags:
            c.update(int(i) for i in t)
        keep = max(1, len(self.tags) // 2)
        return np.array(sorted(i for i, n in c.items() if n >= keep))


def _distance_matrix(tags: list[np.ndarray]) -> np.ndarray:
    n = len(tags)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = 1.0 - overlap(tags[i], tags[j])
    return d


def _keywords(texts: list[str], top: int = 6) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    if not any(t.strip() for t in texts):
        return []
    # project proper-nouns/tooling that would otherwise dominate over concepts
    domain_stop = {
        "nano", "nanobpm", "workforce", "urban", "npm", "ci", "pr", "epic",
        "test", "tests", "run", "repo", "branch", "commit", "file", "files",
    }
    stop = list(ENGLISH_STOP_WORDS | domain_stop)
    vec = TfidfVectorizer(
        stop_words=stop, max_features=400, ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b",  # letters only; drop bare numbers/ids
    )
    x = vec.fit_transform(texts)
    scores = np.asarray(x.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    return vocab[np.argsort(scores)[::-1][:top]].tolist()


def synthesize(
    mem: KCMemory,
    distance_threshold: float = 0.82,
    min_family: int = 2,
) -> list[PatternFamily]:
    """Cluster memory traces into emergent architectural pattern families."""
    from sklearn.cluster import AgglomerativeClustering

    if len(mem) < 2:
        return []
    tags = [t.tag for t in mem.traces]
    d = _distance_matrix(tags)
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(d)

    families: list[PatternFamily] = []
    for cid in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == cid]
        if len(idx) < min_family:
            continue
        fam = PatternFamily(label=f"pattern-{len(families)}")
        for i in idx:
            tr = mem.traces[i]
            fam.members.append(tr.meta)
            fam.tags.append(tr.tag)
            note = tr.meta.get("note") or tr.meta.get("summary")
            if note:
                fam.guidance.append(note)
        texts = [n for m in fam.members for n in (m.get("learnings") or [])]
        fam.keywords = _keywords(texts) if texts else []
        families.append(fam)
    # biggest, most cross-repo families first
    families.sort(key=lambda f: (f.is_cross_repo(), len(f.members)), reverse=True)
    return families


def retrieve(families: list[PatternFamily], plan_tag: np.ndarray) -> tuple[PatternFamily | None, float]:
    """Plan-time lookup: which family holds the nearest exemplar to a new plan?

    Scored by the best overlap against any member tag (nearest-exemplar), which
    is robust for sparse KC codes where a multi-member consensus tag collapses.
    """
    best, best_ov = None, 0.0
    for fam in families:
        ov = max((overlap(plan_tag, t) for t in fam.tags), default=0.0)
        if ov > best_ov:
            best, best_ov = fam, ov
    return best, best_ov
