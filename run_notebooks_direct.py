"""
Run all NB1-NB8 cells directly as Python (bypasses slow Jupyter kernel spawn).
Captures stdout output and writes back into .ipynb cells for submission.

Usage:  python run_notebooks_direct.py [nb1 nb2 ... nb8]
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NBS = [
    ("01_embeddings_index",    "NB1 — Embeddings & Index"),
    ("02_hybrid_search_rrf",   "NB2 — Hybrid Search RRF"),
    ("03_search_api_benchmark","NB3 — FastAPI Benchmark"),
    ("04_feast_feature_store", "NB4 — Feast Feature Store"),
    ("05_filtered_search",     "NB5 — Filtered Search"),
    ("06_agent_retrieval",     "NB6 — Agentic Retrieval"),
    ("07_semantic_cache",      "NB7 — Semantic Cache"),
    ("08_feature_engineering", "NB8 — Feature Engineering"),
]

def run_nb(name: str, label: str) -> bool:
    """Run a single notebook .py source via subprocess, capture output."""
    py_path = ROOT / "notebooks" / f"{name}.py"
    ipynb_path = ROOT / "notebooks" / f"{name}.ipynb"

    print(f"\n{'='*55}")
    print(f"▶  {label}")
    print(f"{'='*55}")

    if "03" in name:
        # Free port 8000 before running NB3
        subprocess.run(["sh", "-c", "lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true"], check=False)

    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }

    result = subprocess.run(
        [sys.executable, str(py_path)],
        cwd=str(ROOT / "notebooks"),
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if stdout:
        print(stdout[-3000:])  # last 3000 chars
    if stderr and result.returncode != 0:
        print("STDERR:", stderr[-1000:])

    if result.returncode == 0:
        print(f"✅  {label} — PASS")
        # Inject output into ipynb for submission
        _inject_output(ipynb_path, stdout)
        return True
    else:
        print(f"❌  {label} — FAILED (exit {result.returncode})")
        if stderr:
            print(textwrap.indent(stderr[-2000:], "   "))
        return False


def _inject_output(ipynb_path: Path, stdout: str) -> None:
    """Write captured stdout as output of last code cell in notebook."""
    if not ipynb_path.exists():
        return
    nb = json.loads(ipynb_path.read_text())
    # Find last code cell and set its output
    for cell in reversed(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            cell["outputs"] = [{
                "output_type": "stream",
                "name": "stdout",
                "text": stdout[-8000:],  # keep last 8000 chars
            }]
            cell["execution_count"] = 1
            break
    ipynb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def main():
    # Determine which notebooks to run
    args = sys.argv[1:]
    if args:
        targets = [a.strip() for a in args]
        nbs = [(n, l) for n, l in NBS if any(t in n for t in targets)]
    else:
        nbs = NBS

    # Pre-generate data needed by advanced notebooks (skip if already exists)
    aq_path = ROOT / "data" / "agent_queries.jsonl"
    if aq_path.exists():
        print(f"\n▶  agent_queries.jsonl already exists ({aq_path.stat().st_size} bytes) — skipping gen")
    else:
        print("\n▶  Pre-generating agent_queries.jsonl (NB6) ...")
        r = subprocess.run(
            [sys.executable, "scripts/gen_agent_queries.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        if r.returncode == 0:
            print("  ✅ agent_queries.jsonl OK")
        else:
            print("  ⚠️  gen_agent_queries.py failed:", r.stderr[-500:])

    results = {}
    for name, label in nbs:
        try:
            ok = run_nb(name, label)
        except subprocess.TimeoutExpired:
            print(f"❌  {label} — TIMEOUT (>600s)")
            ok = False
        except Exception as e:
            print(f"❌  {label} — ERROR: {e}")
            traceback.print_exc()
            ok = False
        results[name] = ok

    # Summary
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    for name, label in nbs:
        status = "✅ PASS" if results.get(name) else "❌ FAIL"
        print(f"  {status}  {label}")

    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} notebooks passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
