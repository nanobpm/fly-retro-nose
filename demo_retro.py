"""Cross-repo smell transfer on the real male-CNS connectome.

Each smell is described with DIFFERENT wording in DIFFERENT repos. We teach the
memory a subset, then probe with a held-out retro of the same smell from a repo
the fly never saw for that smell. If the embedding + FlyHash generalise, the fly
recognises the smell across the repo boundary -- and flags a genuinely new one.
"""

from __future__ import annotations

import os

from flyretro.connectome import load_connectome
from flyretro.featurizer import RetroFeaturizer, RetroRecord
from flyretro.memory import KCMemory

NPZ = os.path.join(os.path.dirname(__file__), "data", "male_cns_pn_kc.npz")

# smell -> list of (repo, epic, learning texts). Same problem, different words/repos.
TAUGHT = {
    "shared-mutable-singleton": [
        ("billing-svc", "#101", [
            "The global Config singleton is mutated at request time; two handlers raced on it and corrupted the tax rate.",
            "Everything reaches into AppState directly instead of receiving what it needs; impossible to test in isolation.",
        ]),
        ("auth-gateway", "#212", [
            "A process-wide mutable registry is written from three call sites; ordering bugs appear under load.",
            "Shared global state makes handlers non-reentrant and unit tests leak into each other.",
        ]),
    ],
    "serde-schema-drift": [
        ("engine-core", "#330", [
            "Adding a field to the persisted snapshot broke loading old snapshots because it lacked a serde default.",
            "Schema evolution wasn't additive; upgrading in place failed to deserialize prior state.",
        ]),
        ("ledger-store", "#455", [
            "A new enum variant made previously written records unreadable after deploy; no backward-compatible default.",
            "Persisted format changed non-additively and old data could not be migrated on read.",
        ]),
    ],
    "chatty-sync-fanout": [
        ("orders-api", "#501", [
            "The endpoint makes N synchronous downstream calls in a loop; p99 latency explodes and retries storm.",
            "Tight synchronous coupling between services causes cascading timeouts under partial failure.",
        ]),
        ("search-svc", "#622", [
            "Per-item blocking RPCs (an N+1 over the network) saturate the pool and trigger retry amplification.",
            "Synchronous fan-out to many services with no backpressure melts down when one dependency is slow.",
        ]),
    ],
}

# Held-out: SAME smells, NEW repos, fresh wording (should be recognised across repos).
HELD_OUT = {
    "shared-mutable-singleton": ("notify-worker", "#900", [
        "A singleton holds mutable connection state that request handlers overwrite; flaky under concurrency.",
        "Global shared object is poked from everywhere, so nothing can be tested without the whole world.",
    ]),
    "serde-schema-drift": ("audit-log", "#901", [
        "We appended a column to the stored event shape and old events failed to decode after rollout.",
        "Non-additive change to the serialized schema; in-place upgrade could not read historical rows.",
    ]),
    "chatty-sync-fanout": ("pricing-svc", "#902", [
        "A request triggers a blocking loop of remote calls; when one is slow everything backs up and retries pile on.",
        "Synchronous chatty coupling with no async boundary causes timeout cascades across the fleet.",
    ]),
}

# A genuinely NEW smell the fly has never encountered (should read as novel).
NOVEL = ("mobile-app", "#999", [
    "Business logic lives in the UI view controllers, so the same rules are re-implemented per screen and drift.",
    "No domain layer: validation is copy-pasted into each view and the variants have already diverged.",
])


def rec(repo, epic, learnings, **kw):
    return RetroRecord(repo=repo, epic=epic, learnings=learnings, **kw)


def main() -> None:
    con = load_connectome(NPZ, hash_length=40, binary=True)
    fh = con.flyhash
    fz = RetroFeaturizer(n_pn=fh.n_pn)
    print(f"connectome {fh.n_kc} KCs x {fh.n_pn} PNs; embedding + FlyHash ready\n")

    mem = KCMemory()
    for smell, items in TAUGHT.items():
        for repo, epic, learnings in items:
            tag = fh.encode(fz.pn_activity(rec(repo, epic, learnings)))
            mem.add(tag, {"smell": smell, "repo": repo, "epic": epic})
    print(f"taught {len(mem)} retros across {len(TAUGHT)} smells\n")

    print("CROSS-REPO probes (same smell, repo the fly never saw for it):")
    ok = 0
    for smell, (repo, epic, learnings) in HELD_OUT.items():
        tag = fh.encode(fz.pn_activity(rec(repo, epic, learnings)))
        top = mem.query(tag, top_k=1)[0]
        nov = mem.novelty(tag)
        hit = "OK " if top.meta["smell"] == smell else "MISS"
        ok += top.meta["smell"] == smell
        print(f"  [{hit}] {repo:12s}{epic:6s} -> {top.meta['smell']:26s} "
              f"(learned in {top.meta['repo']}) overlap={top.overlap:.2f} novelty={nov:.2f}")

    print("\nNOVEL probe (a smell never taught):")
    repo, epic, learnings = NOVEL
    tag = fh.encode(fz.pn_activity(rec(repo, epic, learnings)))
    top = mem.query(tag, top_k=1)[0]
    nov = mem.novelty(tag)
    print(f"  {repo:12s}{epic:6s} -> nearest={top.meta['smell']:26s} overlap={top.overlap:.2f} novelty={nov:.2f}")

    print(f"\nresult: {ok}/{len(HELD_OUT)} cross-repo smells recognised; novel smell novelty={nov:.2f}")


if __name__ == "__main__":
    main()
