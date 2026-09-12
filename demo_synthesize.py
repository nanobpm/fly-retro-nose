"""Synthesize a retro corpus into cross-repo architectural patterns, then use
them prophylactically: given a NEW plan/spec, surface the structural guidance
distilled from the family it most resembles -- before implementation starts.
"""

from __future__ import annotations

import os

from flyretro.connectome import load_connectome
from flyretro.featurizer import RetroFeaturizer, RetroRecord
from flyretro.memory import KCMemory
from flyretro.synthesize import retrieve, synthesize

NPZ = os.path.join(os.path.dirname(__file__), "data", "male_cns_pn_kc.npz")

# smell -> (guidance, [(repo, epic, [learnings])...]) across several repos.
CORPUS = {
    "shared-mutable-state": (
        "Introduce a domain layer and inject dependencies; forbid process-global mutable state.",
        [
            ("billing-svc", "#101", ["Global Config singleton mutated at request time; two handlers raced and corrupted the tax rate.",
                                     "Handlers reach into AppState directly, impossible to test in isolation."]),
            ("auth-gateway", "#212", ["A process-wide mutable registry is written from three call sites; ordering bugs under load.",
                                      "Shared global state makes handlers non-reentrant and tests leak into each other."]),
            ("notify-worker", "#318", ["Singleton holds mutable connection state handlers overwrite; flaky under concurrency.",
                                       "A global shared object poked from everywhere; nothing testable without the whole world."]),
            ("cart-svc", "#402", ["Module-level mutable cache shared across requests produced stale, racy reads.",
                                  "State stored in a global instead of passed explicitly; concurrency bugs everywhere."]),
        ],
    ),
    "serde-schema-drift": (
        "Make every schema change additive with serde defaults; add a golden-old-snapshot load test.",
        [
            ("engine-core", "#330", ["Adding a snapshot field broke loading old snapshots -- missing serde default.",
                                     "Schema evolution wasn't additive; in-place upgrade failed to deserialize prior state."]),
            ("ledger-store", "#455", ["A new enum variant made prior records unreadable after deploy; no backward-compatible default.",
                                      "Persisted format changed non-additively and old data could not be migrated on read."]),
            ("audit-log", "#521", ["Appended a column to the stored event shape; old events failed to decode after rollout.",
                                   "Non-additive serialized schema change; upgrade could not read historical rows."]),
            ("state-svc", "#588", ["Renamed a persisted field without a default; snapshot restore threw on old data.",
                                   "No compatibility shim for the serialized state across versions."]),
        ],
    ),
    "chatty-sync-fanout": (
        "Add an async boundary with backpressure; batch or precompute the fan-out instead of per-item RPCs.",
        [
            ("orders-api", "#501", ["Endpoint makes N synchronous downstream calls in a loop; p99 explodes and retries storm.",
                                    "Tight synchronous coupling causes cascading timeouts under partial failure."]),
            ("search-svc", "#622", ["Per-item blocking RPCs (an N+1 over the network) saturate the pool; retry amplification.",
                                    "Synchronous fan-out to many services with no backpressure melts down when one is slow."]),
            ("pricing-svc", "#655", ["A request triggers a blocking loop of remote calls; one slow dep backs everything up.",
                                     "Chatty synchronous coupling with no async boundary causes timeout cascades."]),
            ("feed-svc", "#690", ["Sequential remote calls per element create latency proportional to fan-out size.",
                                  "No batching or async pipeline; blocking network N+1 dominates the request."]),
        ],
    ),
}

# A NEW plan/spec (pre-implementation). Prose describes intended design.
NEW_PLAN = RetroRecord(
    repo="inventory-svc",
    epic="#1200",
    learnings=[
        "Plan: a service-wide in-memory registry that request handlers read and update to track live stock.",
        "Multiple endpoints will share and mutate this global object to stay in sync.",
    ],
    summary="Design a stock-tracking service around a shared mutable in-memory registry updated by handlers.",
)


def main() -> None:
    con = load_connectome(NPZ, hash_length=96, binary=True)
    fh = con.flyhash
    fz = RetroFeaturizer(n_pn=fh.n_pn)

    mem = KCMemory()
    for smell, (guidance, items) in CORPUS.items():
        for repo, epic, learnings in items:
            tag = fh.encode(fz.pn_activity(RetroRecord(repo=repo, epic=epic, learnings=learnings)))
            mem.add(tag, {"true_smell": smell, "repo": repo, "epic": epic,
                          "learnings": learnings, "note": guidance})
    print(f"ingested {len(mem)} retros across {len(CORPUS)} true smells\n")

    families = synthesize(mem, distance_threshold=0.88, min_family=2)
    print(f"the fly synthesized {len(families)} emergent pattern families:\n")
    for fam in families:
        truth = {m["true_smell"] for m in fam.members}
        tag = "CROSS-REPO" if fam.is_cross_repo() else "single-repo"
        print(f"  {fam.label} [{tag}] {len(fam.members)} retros, repos={fam.repos}")
        print(f"    keywords: {', '.join(fam.keywords[:6])}")
        print(f"    guidance: {fam.guidance[0] if fam.guidance else '(none)'}")
        print(f"    (ground-truth smells in family: {truth})\n")

    print("=" * 72)
    print(f"NEW PLAN {NEW_PLAN.repo}{NEW_PLAN.epic} -- prophylactic lookup before implementation:")
    plan_tag = fh.encode(fz.pn_activity(NEW_PLAN))
    fam, ov = retrieve(families, plan_tag)
    if fam:
        print(f"  closest pattern: {fam.label} (overlap {ov:.2f}), seen in {fam.repos}")
        print(f"  -> STRUCTURAL GUIDANCE: {fam.guidance[0]}")
    else:
        print("  no matching pattern -- looks structurally novel.")


if __name__ == "__main__":
    main()
