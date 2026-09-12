"""Shared pipeline: ingest real retros -> hash -> synthesize families.

Loaded once and reused by the report, the plan-time advisor, and the dashboard
so they all see the same synthesized memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .connectome import Connectome, load_connectome
from .featurizer import RetroFeaturizer, RetroRecord
from .memory import KCMemory
from .nwf_ingest import DEFAULT_DB, load_retros
from .synthesize import PatternFamily, synthesize

NPZ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "male_cns_pn_kc.npz")


@dataclass
class Pipeline:
    con: Connectome
    fz: RetroFeaturizer
    mem: KCMemory
    families: list[PatternFamily]
    records: list[RetroRecord]

    def encode_text(self, text: str, repo: str = "(new)"):
        rec = RetroRecord(repo=repo, epic="(new)", learnings=[text], summary=text)
        return self.con.flyhash.encode(self.fz.pn_activity(rec))


def build(
    db: str = DEFAULT_DB,
    hash_length: int = int(os.environ.get("FLY_HASH_LENGTH", "96")),
    threshold: float = float(os.environ.get("FLY_THRESHOLD", "0.90")),
    min_family: int = int(os.environ.get("FLY_MIN_FAMILY", "2")),
) -> Pipeline:
    records = load_retros(db)
    con = load_connectome(NPZ, hash_length=hash_length, binary=True)
    fz = RetroFeaturizer(n_pn=con.flyhash.n_pn)
    mem = KCMemory()
    for rec in records:
        tag = con.flyhash.encode(fz.pn_activity(rec))
        mem.add(tag, {"repo": rec.repo, "epic": rec.epic,
                      "learnings": rec.learnings, "note": rec.summary})
    families = synthesize(mem, distance_threshold=threshold, min_family=min_family)
    return Pipeline(con=con, fz=fz, mem=mem, families=families, records=records)
