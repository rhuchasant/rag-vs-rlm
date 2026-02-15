"""
Phase 4: Hybrid RLM + RAG comparison on mixed tasks.
- Broad tasks (extract everything) -> RLM
- Targeted tasks (specific lookup) -> RAG
Compares: Hybrid vs RLM-only vs RAG-only.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

_rlm_log = logging.getLogger("rlm")
_rlm_log.setLevel(logging.WARNING)  # Less verbose for hybrid run

from src.rag import RAGPipeline
from src.rlm import RLMClient
from src.evaluation import compute_coverage





def _extract_from_tab_content(content: str) -> dict:
    """Extract filename and first column from tab content for ground truth."""
    out = {}
    m = re.search(r"File:\s*(\S+)", content, re.I)
    if m:
        out["filename"] = m.group(1).strip().lower()
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "columns:" in line.lower() and i + 1 < len(lines):
            # Next lines are like "- col_name (TYPE)"
            for j in range(i + 1, min(i + 10, len(lines))):
                mm = re.match(r"-\s*(\w+)\s*\(", lines[j])
                if mm:
                    out["first_column"] = mm.group(1).lower()
                    break
            break
    return out


def _check_targeted_match(response: str, expected: str | list) -> bool:
    """Check if expected value(s) appear in response."""
    r = response.lower()
    if isinstance(expected, str):
        return expected.lower() in r
    return any(e.lower() in r for e in expected)


def _build_tab_info(tabs: dict) -> dict:
    """Build tab metadata for ground truth (filename, first_column, filename->tab map)."""
    info = {}
    filename_to_tab = {}
    for tab_key, content in tabs.items():
        meta = _extract_from_tab_content(content)
        info[tab_key] = meta
        if meta.get("filename"):
            filename_to_tab[meta["filename"]] = tab_key
    return {"by_tab": info, "filename_to_tab": filename_to_tab}


def run_one_seed(args, benchmark_dir: Path, seed: int) -> dict:
    """Run hybrid comparison for one seed. Returns results dict."""
    if args.generate:
        from benchmarks.generate_data_layout_benchmark import generate_benchmark
        data = generate_benchmark(args.tabs, benchmark_dir, target_total_chars=args.size, seed=seed)
        tabs = data["tabs"]
        text = data["text"]
    else:
        json_path = benchmark_dir / "data_layout_benchmark.json"
        txt_path = benchmark_dir / "data_layout_benchmark.txt"
        if not json_path.exists() or not txt_path.exists():
            print("Run with --generate first")
            sys.exit(1)
        with open(json_path) as f:
            tabs = json.load(f)
        text = txt_path.read_text()

    expected_tabs = list(tabs.keys())
    num_tabs = len(expected_tabs)

    # Mixed tasks: 1 broad + 5 targeted (extended for Phase 4 high)
    keys = list(tabs.keys())
    tab_5_key = keys[4] if len(keys) > 4 else keys[0]
    tab_1_key = keys[0]
    tab_10_key = keys[9] if len(keys) > 9 else keys[0]

    tab_info = _build_tab_info(tabs)
    info_5 = tab_info["by_tab"].get(tab_5_key, {})
    info_1 = tab_info["by_tab"].get(tab_1_key, {})
    info_10 = tab_info["by_tab"].get(tab_10_key, {})
    f2t = tab_info["filename_to_tab"]

    tasks = [
        {
            "id": "broad_list_tabs",
            "query": "List all tab names in order. Include every tab from the document.",
            "type": "broad",
            "expected_tabs": expected_tabs,
        },
        {
            "id": "targeted_filename",
            "query": f"What is the filename for {tab_5_key}?",
            "type": "targeted",
            "expected": info_5.get("filename", "shipping.csv"),
        },
        {
            "id": "targeted_first_col",
            "query": f"What is the first column in {tab_1_key}?",
            "type": "targeted",
            "expected": info_1.get("first_column", "customer_id"),
        },
        {
            "id": "targeted_primary_key",
            "query": f"What is the primary key column for {tab_10_key}?",
            "type": "targeted",
            "expected": info_10.get("first_column", "supplier_id"),
        },
        {
            "id": "targeted_which_tab",
            "query": "Which tab uses the file shipping.csv?",
            "type": "targeted",
            "expected": [f2t.get("shipping.csv", tab_5_key), "Shipping_Info", "Tab_5_Shipping_Info"],
        },
        {
            "id": "targeted_file_for_tab1",
            "query": f"What file does {tab_1_key} export to?",
            "type": "targeted",
            "expected": info_1.get("filename", "customer_export.csv"),
        },
    ]

    results = {
        "num_tabs": num_tabs,
        "doc_size_chars": len(text),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "tasks": [],
    }

    rag = RAGPipeline(top_k=args.top_k)
    rlm = RLMClient()

    # Initialize Hybrid Engine
    from src.hybrid import HybridQueryEngine
    hybrid_engine = HybridQueryEngine(rag, rlm)

    for task in tasks:
        q = task["query"]
        tid = task["id"]
        ttype = task["type"]
        print(f"\n--- Task: {tid} ({ttype}) ---")

        ex = {"task_id": tid, "type": ttype, "query": q[:80] + "..."}

        # RAG-only
        try:
            r_rag = rag.query(text, q, collection_name=f"hybrid_rag_{tid}_s{seed}")
            if ttype == "broad":
                cov = compute_coverage(task["expected_tabs"], r_rag)
                ex["rag_match"] = cov >= 1.0
                ex["rag_coverage"] = cov
            else:
                ex["rag_match"] = _check_targeted_match(r_rag, task["expected"])
            ex["rag_response"] = (r_rag[:200] + "...") if len(r_rag) > 200 else r_rag
        except Exception as e:
            ex["rag_match"] = False
            ex["rag_error"] = str(e)[:150]
            ex["rag_response"] = None

        # RLM-only
        try:
            r_rlm = rlm.completion(tabs, q)
            if ttype == "broad":
                cov = compute_coverage(task["expected_tabs"], r_rlm)
                ex["rlm_match"] = cov >= 1.0
                ex["rlm_coverage"] = cov
            else:
                ex["rlm_match"] = _check_targeted_match(r_rlm, task["expected"])
            ex["rlm_response"] = (r_rlm[:200] + "...") if len(r_rlm) > 200 else r_rlm
        except Exception as e:
            ex["rlm_match"] = False
            ex["rlm_error"] = str(e)[:150]
            ex["rlm_response"] = None

        # Hybrid: using the new Engine
        try:
            h_out = hybrid_engine.query(q, tabs, text, task_id=f"{tid}_s{seed}")
            r_hyb = h_out["response"] or ""
            
            if ttype == "broad":
                cov = compute_coverage(task["expected_tabs"], r_hyb)
                ex["hybrid_match"] = cov >= 1.0
                ex["hybrid_coverage"] = cov
            else:
                ex["hybrid_match"] = _check_targeted_match(r_hyb, task["expected"])
            ex["hybrid_response"] = (r_hyb[:200] + "...") if len(r_hyb) > 200 else r_hyb
            ex["hybrid_route"] = h_out.get("route")
        except Exception as e:
            ex["hybrid_match"] = False
            ex["hybrid_error"] = str(e)[:150]
            ex["hybrid_response"] = None

        results["tasks"].append(ex)
        print(f"  RAG: {'OK' if ex['rag_match'] else 'FAIL'} | RLM: {'OK' if ex['rlm_match'] else 'FAIL'} | Hybrid: {'OK' if ex['hybrid_match'] else 'FAIL'}")

    # Summary
    rag_ok = sum(1 for t in results["tasks"] if t.get("rag_match"))
    rlm_ok = sum(1 for t in results["tasks"] if t.get("rlm_match"))
    hyb_ok = sum(1 for t in results["tasks"] if t.get("hybrid_match"))
    n = len(results["tasks"])
    results["summary"] = {
        "rag_correct": rag_ok,
        "rlm_correct": rlm_ok,
        "hybrid_correct": hyb_ok,
        "total_tasks": n,
        "rag_rate": rag_ok / n if n else 0,
        "rlm_rate": rlm_ok / n if n else 0,
        "hybrid_rate": hyb_ok / n if n else 0,
    }

    print("\n" + "=" * 50)
    print(f"SUMMARY (seed={seed}): RAG {rag_ok}/{n} | RLM {rlm_ok}/{n} | Hybrid {hyb_ok}/{n}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--tabs", type=int, default=30)
    parser.add_argument("--size", type=int, default=750000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run (e.g. 42,43,44)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    if len(seeds) == 1:
        results = run_one_seed(args, benchmark_dir, seeds[0])
        out_path = Path(__file__).parent.parent / "results" / f"hybrid_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")
        return

    # Multi-seed: run each, aggregate
    all_runs = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}\n[Hybrid {i}/{len(seeds)}] seed={seed}\n{'='*60}")
        args.generate = True  # Always generate for multi-seed
        run_result = run_one_seed(args, benchmark_dir, seed)
        all_runs.append(run_result)

    # Aggregate
    rag_rates = [r["summary"]["rag_rate"] for r in all_runs]
    rlm_rates = [r["summary"]["rlm_rate"] for r in all_runs]
    hyb_rates = [r["summary"]["hybrid_rate"] for r in all_runs]
    import statistics
    agg = {
        "seeds": seeds,
        "n_runs": len(seeds),
        "rag_rate_mean": statistics.mean(rag_rates),
        "rag_rate_std": statistics.stdev(rag_rates) if len(rag_rates) > 1 else 0,
        "rlm_rate_mean": statistics.mean(rlm_rates),
        "rlm_rate_std": statistics.stdev(rlm_rates) if len(rlm_rates) > 1 else 0,
        "hybrid_rate_mean": statistics.mean(hyb_rates),
        "hybrid_rate_std": statistics.stdev(hyb_rates) if len(hyb_rates) > 1 else 0,
    }
    agg["rag_rate_ci95"] = 1.96 * agg["rag_rate_std"] / (len(seeds) ** 0.5) if len(seeds) > 1 else 0
    agg["rlm_rate_ci95"] = 1.96 * agg["rlm_rate_std"] / (len(seeds) ** 0.5) if len(seeds) > 1 else 0
    agg["hybrid_rate_ci95"] = 1.96 * agg["hybrid_rate_std"] / (len(seeds) ** 0.5) if len(seeds) > 1 else 0

    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {"tabs": args.tabs, "size": args.size, "seeds": seeds},
        "aggregate": agg,
        "runs": all_runs,
    }

    out_path = Path(__file__).parent.parent / "results" / f"hybrid_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\nAGGREGATE ({len(seeds)} seeds)")
    print(f"  RAG:    {agg['rag_rate_mean']:.2%} +/- {agg['rag_rate_ci95']:.2%}")
    print(f"  RLM:    {agg['rlm_rate_mean']:.2%} +/- {agg['rlm_rate_ci95']:.2%}")
    print(f"  Hybrid: {agg['hybrid_rate_mean']:.2%} +/- {agg['hybrid_rate_ci95']:.2%}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
