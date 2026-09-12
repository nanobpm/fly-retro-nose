"""Self-test: prove the real male-CNS connectome separates familiar smells
from novel ones, before any real retro data is wired in.

We fabricate a handful of archetypal "architectural smells" as PN-activity
centroids, draw noisy variants, teach the memory some of them, then check:
 - held-out variants of taught smells are recognised (high overlap, low novelty)
 - variants of an untaught smell read as novel (low overlap, high novelty)
"""

from __future__ import annotations

import os

import numpy as np

from flyretro.connectome import load_connectome
from flyretro.memory import KCMemory

HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, "data", "male_cns_pn_kc.npz")


def make_smell(rng: np.random.Generator, n_pn: int, active: int) -> np.ndarray:
    v = np.zeros(n_pn, dtype=np.float32)
    idx = rng.choice(n_pn, size=active, replace=False)
    v[idx] = rng.uniform(0.6, 1.0, size=active)
    return v


def variant(rng: np.random.Generator, centroid: np.ndarray, noise: float) -> np.ndarray:
    return centroid + rng.normal(0.0, noise, size=centroid.shape).astype(np.float32)


def main() -> None:
    con = load_connectome(NPZ, hash_length=32, binary=True)
    fh = con.flyhash
    rng = np.random.default_rng(7)
    print(f"connectome: {fh.n_kc} KCs x {fh.n_pn} PNs, tag length {fh.hash_length}")

    n_smells = 5
    centroids = [make_smell(rng, fh.n_pn, active=15) for _ in range(n_smells)]
    names = [f"smell-{i}" for i in range(n_smells)]

    # Teach the memory noisy variants of smells 0..3; hold out smell 4 entirely.
    mem = KCMemory()
    for s in range(4):
        for j in range(6):
            tag = fh.encode(variant(rng, centroids[s], noise=0.15))
            mem.add(tag, {"smell": names[s], "repo": f"repo-{s}", "epic": f"#{100 + s * 10 + j}"})
    print(f"memory holds {len(mem)} traces across smells 0..3\n")

    def probe(label, centroid):
        tag = fh.encode(variant(rng, centroid, noise=0.15))
        top = mem.query(tag, top_k=1)[0]
        nov = mem.novelty(tag)
        print(f"{label:16s} -> nearest={top.meta['smell']:8s} overlap={top.overlap:.2f} novelty={nov:.2f}")
        return top, nov

    print("Held-out variants of TAUGHT smells (expect recognised, low novelty):")
    fam_nov = []
    for s in range(4):
        top, nov = probe(f"taught {names[s]}", centroids[s])
        fam_nov.append(nov)
        assert top.meta["smell"] == names[s], "misrecognised a taught smell"

    print("\nVariants of the UNTAUGHT smell (expect novel, high novelty):")
    _, nov4 = probe("untaught smell-4", centroids[4])

    print("\nresult:")
    print(f"  mean novelty, taught smells   = {np.mean(fam_nov):.2f}")
    print(f"  novelty, untaught smell-4     = {nov4:.2f}")
    assert nov4 > max(fam_nov) + 0.2, "novel smell not clearly separated"
    print("  PASS: familiar smells recognised, novel smell flagged.")


if __name__ == "__main__":
    main()
