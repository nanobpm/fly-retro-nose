"""Load the real male-CNS PN->KC connectivity as a FlyHash projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flyhash import FlyHash


@dataclass
class Connectome:
    flyhash: FlyHash
    pn_bodyids: np.ndarray
    kc_bodyids: np.ndarray
    pn_types: np.ndarray
    kc_types: np.ndarray

    @property
    def n_pn(self) -> int:
        return self.flyhash.n_pn


def load_connectome(
    npz_path: str,
    hash_length: int = 32,
    binary: bool = True,
) -> Connectome:
    """Build a FlyHash from the saved male-CNS PN->KC matrix.

    `binary=True` uses the presence/absence of each PN->KC connection (the
    classic FlyHash), discarding synapse counts; `False` keeps the weights.
    """
    d = np.load(npz_path, allow_pickle=True)
    m = d["M"].astype(np.float32)
    if binary:
        m = (m > 0).astype(np.float32)
    return Connectome(
        flyhash=FlyHash(projection=m, hash_length=hash_length),
        pn_bodyids=d["pn_bodyids"],
        kc_bodyids=d["kc_bodyids"],
        pn_types=d["pn_types"],
        kc_types=d["kc_types"],
    )
