# fly-retro-nose

Synthesize nano-workforce post-implementation **retros** into cross-repo
**architectural patterns**, then feed that knowledge forward to
**prophylactically structure new projects** at plan time.

The similarity engine is the *Drosophila* mushroom body. The real
**male-CNS `v1.0` connectome** (Janelia, Sept 2026) supplies the PN→KC
adjacency matrix, which acts as a sparse, expansive random projection —
a FlyHash locality-sensitive hash (Dasgupta, Stevens & Navlakha,
*Science* 2017). Retros are "smelled" by the same nose, so structurally
similar retros land on overlapping Kenyon-cell tags regardless of repo.

## How it works

```
retro text ─▶ MiniLM embedding ⊕ 5 structured signals
           ─▶ 296 PN channels ─▶ [male-CNS PN→KC] ─▶ k-WTA ─▶ sparse KC tag
           ─▶ agglomerative clustering ─▶ emergent pattern families
```

- `flyretro/flyhash.py` — FlyHash core (sparse projection + k-WTA, Jaccard `overlap`).
- `flyretro/connectome.py` — loads `data/male_cns_pn_kc.npz` (3785 KC × 296 PN, ~5.7 inputs/KC).
- `flyretro/featurizer.py` — `RetroRecord` + embedding→PN transduction.
- `flyretro/memory.py` — associative KC store (`add`/`query`/`novelty`).
- `flyretro/synthesize.py` — clusters KC tags into `PatternFamily`s; `retrieve()` (nearest exemplar).
- `flyretro/nwf_ingest.py` — reads real retros from nano-workforce `app.db` (read-only/immutable).
- `flyretro/llm.py` + `distill.py` — **optional** llama.cpp guideline distillation, graceful fallback.
- `flyretro/pipeline.py` — shared `build()` → `Pipeline`.
- `flyretro/plan_advice.py` — `advise_plan()` for plan-time injection.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy sentence-transformers neuprint-python
# neuPrint token for regenerating the connectome (not needed if data/ exists):
echo 'NEUPRINT_TOKEN=...' > .env
```

`data/male_cns_pn_kc.npz` is the extracted connectome and is required.
It is regenerable via neuPrint but slow, so it is kept out of git.

## The four entry points

```bash
. .venv/bin/activate && set -a && . ./.env && set +a

python report.py            # writes reports/architectural-patterns.md
python plan_advise.py "..." # plan-time advisory for a new issue/plan
python dashboard.py         # live KC-raster "nose" at http://127.0.0.1:8765
                            #   GET /api/sniff?text=... -> {active,novelty,nearest,family}
```

Distillation of family guidelines (report + dashboard) and plan-time advice
use a **small always-on LLM** if configured, and the report can optionally
escalate a cross-cutting synthesis to a **frontier LLM**. Both degrade
gracefully — if a tier is unset or unreachable, that path falls back
(extractive guideline / omitted synthesis) and nothing crashes.

```bash
# small tier — always-on per-family distillation + plan-time phrasing
export FLY_LLM_URL=http://localhost:11434   # unset = disabled (extractive fallback)
export FLY_LLM_MODEL=qwen2.5:1.5b           # a "less capable" local model is fine
export FLY_LLM_KEY=...                       # optional bearer token
export FLY_LLM_TIMEOUT=60

# frontier tier — optional periodic deep synthesis in the report only
export FLY_FRONTIER_URL=https://api.openai.com   # unset = no synthesis section
export FLY_FRONTIER_MODEL=gpt-4o
export FLY_FRONTIER_KEY=$OPENAI_API_KEY
export FLY_FRONTIER_TIMEOUT=60
```

The division of labour is deliberate: the **fly** (deterministic, O(1),
auditable) does retrieval, clustering and novelty gating; the **small LLM**
does local wordsmithing; the **frontier LLM** is reserved for the low-frequency,
high-value cross-family narrative.

## Demos / self-tests

```bash
python demo_synthetic.py    # familiar vs novel cleanly separated
python demo_retro.py        # cross-repo transfer: 3/3 smells recognized across repos
python demo_synthesize.py   # emergent pattern families
python ingest_real.py       # real retros from nano-workforce app.db
```

## Honest limitation

On the current real corpus (7 filed retros, mostly the nano* ecosystem,
jargon-heavy) everything collapses into one broad family and plan-time
matching on generic prose scores near noise. The mechanism is proven on
synthetic and cross-repo demos; it sharpens as more diverse retros (and
real issue-body text) accumulate.
