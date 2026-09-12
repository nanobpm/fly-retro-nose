"""Feed the fly your real nano-workforce retros.

Ingests every filed retro from app.db, hashes each through the male-CNS
mushroom body, then reports which epics smell alike (leave-one-out nearest
neighbour) and any emergent cross-repo pattern families. Honest about a small
corpus: with only a handful of retros, expect few or no multi-member families.
"""

from __future__ import annotations

import os
import sys

from flyretro.connectome import load_connectome
from flyretro.featurizer import RetroFeaturizer
from flyretro.flyhash import overlap
from flyretro.memory import KCMemory
from flyretro.nwf_ingest import DEFAULT_DB, load_retros
from flyretro.synthesize import synthesize

NPZ = os.path.join(os.path.dirname(__file__), "data", "male_cns_pn_kc.npz")


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    records = load_retros(db)
    print(f"ingested {len(records)} filed retros from {db}\n")
    if not records:
        print("no filed retros found."); return

    con = load_connectome(NPZ, hash_length=96, binary=True)
    fh = con.flyhash
    fz = RetroFeaturizer(n_pn=fh.n_pn)

    mem = KCMemory()
    tags = []
    for rec in records:
        tag = fh.encode(fz.pn_activity(rec))
        tags.append(tag)
        mem.add(tag, {"repo": rec.repo, "epic": rec.epic,
                      "learnings": rec.learnings, "note": rec.summary})

    print("Which real epics smell alike (leave-one-out nearest neighbour):")
    for i, rec in enumerate(records):
        best_j, best_ov = -1, -1.0
        for j in range(len(records)):
            if j == i:
                continue
            ov = overlap(tags[i], tags[j])
            if ov > best_ov:
                best_j, best_ov = j, ov
        nn = records[best_j]
        print(f"  {rec.repo:26s}{rec.epic:7s} ~ {nn.repo}{nn.epic:7s} overlap={best_ov:.2f}")

    print("\nEmergent pattern families (min 2 members):")
    fams = synthesize(mem, distance_threshold=0.90, min_family=2)
    if not fams:
        print("  none yet — corpus too small / too diverse. Needs more retros to cluster.")
    for fam in fams:
        cross = "CROSS-REPO" if fam.is_cross_repo() else "single-repo"
        print(f"  {fam.label} [{cross}] repos={fam.repos}")
        print(f"    keywords: {', '.join(fam.keywords[:6])}")


if __name__ == "__main__":
    main()
