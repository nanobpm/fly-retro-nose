"""FlyHash: the fly olfactory circuit as a locality-sensitive hash.

Implements the PN -> Kenyon-cell -> winner-take-all pipeline from
Dasgupta, Stevens & Navlakha, "A neural algorithm for a fundamental
computing problem", Science 2017. The sparse expansive random projection
followed by a k-winner-take-all step yields a similarity-preserving
(locality-sensitive) sparse binary tag, which we use for novelty detection
and nearest-smell retrieval over engineering retros.

The projection matrix is pluggable: `synthetic_projection` builds a random
sparse binary matrix with the fly's connection statistics, and the real
male-CNS PN->KC adjacency can be dropped in via `FlyHash(projection=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def synthetic_projection(
    n_pn: int,
    n_kc: int,
    samples_per_kc: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """A random sparse binary PN->KC matrix (n_kc x n_pn).

    Each Kenyon cell samples `samples_per_kc` projection neurons at random,
    matching the ~6-input sparsity the fly uses. This stands in for the real
    connectome adjacency until it is loaded from neuPrint.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((n_kc, n_pn), dtype=np.float32)
    for k in range(n_kc):
        idx = rng.choice(n_pn, size=min(samples_per_kc, n_pn), replace=False)
        m[k, idx] = 1.0
    return m


@dataclass
class FlyHash:
    """Sparse expansive projection + k-winner-take-all -> sparse binary tag."""

    projection: np.ndarray  # (n_kc, n_pn)
    hash_length: int = 32  # number of active Kenyon cells kept (the WTA k)

    @property
    def n_pn(self) -> int:
        return self.projection.shape[1]

    @property
    def n_kc(self) -> int:
        return self.projection.shape[0]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Divisive normalization the antennal lobe performs: centre each odor
        so absolute concentration is discarded and only the pattern remains."""
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        return x - mean

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return the indices of the active Kenyon cells (the sparse tag)."""
        xn = self._normalize(x)
        activity = self.projection @ xn
        k = min(self.hash_length, self.n_kc)
        winners = np.argpartition(activity, -k)[-k:]
        return np.sort(winners)

    def encode_dense(self, x: np.ndarray) -> np.ndarray:
        """Return the tag as a dense 0/1 vector over all Kenyon cells."""
        tag = np.zeros(self.n_kc, dtype=np.uint8)
        tag[self.encode(x)] = 1
        return tag


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two active-index sets: the similarity of two smells."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
