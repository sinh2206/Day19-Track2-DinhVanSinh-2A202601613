#!/usr/bin/env bash
# ============================================================
# run_all_tests.sh — Chạy toàn bộ NB1–NB8 + pytest + verify
# ============================================================
set -e
cd "$(dirname "$0")"

source .venv/bin/activate

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Day 19 — Run All Tests & Notebooks      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

NB_OPTS="--to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3"

# ── Pre-download fastembed model (prevents kernel timeout on first run) ──
echo "▶ Pre-downloading fastembed model BAAI/bge-small-en-v1.5 ..."
python -c "
from fastembed import TextEmbedding
m = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
r = list(m.embed(['warmup']))
print(f'  Model ready — dim={len(r[0])}')
"
echo "  ✅ fastembed model cached"

run_nb() {
    local nb="$1"
    local label="$2"
    echo "▶ Running $label ..."
    if jupyter nbconvert $NB_OPTS "$nb" 2>&1 | tail -5; then
        echo "  ✅ $label DONE"
    else
        echo "  ❌ $label FAILED"
        exit 1
    fi
}

# ── Pre-generate data needed by NB6 and NB8 ──────────────────────────
echo "▶ Pre-generating agent_queries.jsonl (NB6 cần) ..."
python scripts/gen_agent_queries.py 2>&1 | tail -3
echo "  ✅ agent_queries.jsonl OK"

# ── Core (bắt buộc) ─────────────────────────────────────────────────
echo ""
echo "═══ CORE (NB1–NB4) ════════════════════════"
run_nb "notebooks/01_embeddings_index.ipynb"    "NB1 — Embeddings & Index"
run_nb "notebooks/02_hybrid_search_rrf.ipynb"   "NB2 — Hybrid Search RRF"
run_nb "notebooks/03_search_api_benchmark.ipynb" "NB3 — FastAPI Benchmark"
run_nb "notebooks/04_feast_feature_store.ipynb"  "NB4 — Feast Feature Store"

# ── Advanced (nâng cao) ──────────────────────────────────────────────
echo ""
echo "═══ ADVANCED (NB5–NB8) ════════════════════"
run_nb "notebooks/05_filtered_search.ipynb"      "NB5 — Filtered Search"
run_nb "notebooks/06_agent_retrieval.ipynb"      "NB6 — Agentic Retrieval"
run_nb "notebooks/07_semantic_cache.ipynb"       "NB7 — Semantic Cache"
run_nb "notebooks/08_feature_engineering.ipynb"  "NB8 — Feature Engineering"

# ── pytest ───────────────────────────────────────────────────────────
echo ""
echo "═══ pytest (34 tests) ══════════════════════"
pytest --tb=short -q 2>&1
echo "  ✅ pytest PASSED"

# ── Smoke test ───────────────────────────────────────────────────────
echo ""
echo "═══ Smoke test (verify-lite) ═══════════════"
python scripts/verify_lite.py 2>&1
echo "  ✅ verify-lite PASSED"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  🎉 ALL DONE — NB1–NB8 + tests PASSED   ║"
echo "╚══════════════════════════════════════════╝"
