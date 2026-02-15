"""
Run RLM vs RAG comparison on the data layout benchmark.
Reproduces the real-world failure case: multi-tab extraction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable RLM progress logging (shows iterations, code blocks, sub-LLM calls)
_rlm_log = logging.getLogger("rlm")
_rlm_log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
_rlm_log.addHandler(_h)

from src.rag import RAGPipeline
from src.rlm import RLMClient
from src.evaluation import compute_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data_layout", choices=["data_layout"])
    parser.add_argument("--generate", action="store_true", help="Generate benchmark first")
    parser.add_argument("--tabs", type=int, default=30, help="Number of tabs (default 30, max 34)")
    parser.add_argument("--size", type=int, default=750000, help="Target doc size in chars (default 750k)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for benchmark generation (when --generate)")
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--rlm-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="RAG retrieval top-k (semantic configs)")
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    if args.generate:
        from benchmarks.generate_data_layout_benchmark import generate_benchmark
        data = generate_benchmark(args.tabs, benchmark_dir, target_total_chars=args.size, seed=args.seed)
        tabs = data["tabs"]
        text = data["text"]
    else:
        json_path = benchmark_dir / "data_layout_benchmark.json"
        txt_path = benchmark_dir / "data_layout_benchmark.txt"
        if not json_path.exists() or not txt_path.exists():
            print("Run with --generate first to create the benchmark")
            sys.exit(1)
        with open(json_path) as f:
            tabs = json.load(f)
        text = txt_path.read_text()

    expected_tabs = list(tabs.keys())
    num_tabs = len(expected_tabs)
    query = "List all tab names in order. Include every tab from the document."

    results = {
        "query": query,
        "num_tabs": num_tabs,
        "expected_tabs": expected_tabs,
        "doc_size_chars": len(text),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    def run_rag_config(name: str, rag: RAGPipeline, collection: str, tab_ctx=None):
        """Run RAG query; on context_length_exceeded, log error and return failure."""
        try:
            r = rag.query(text, query, collection_name=collection, tab_contents=tab_ctx)
            c = compute_coverage(expected_tabs, r)
            return {"coverage": c, "response_preview": (r[:400] + "...") if len(r) > 400 else r}
        except Exception as e:
            err_msg = str(e)
            is_context_error = "context_length" in err_msg.lower() or "context length" in err_msg.lower()
            if is_context_error:
                print(f"  RAG {name}: ERROR - context length exceeded (model limit)")
            else:
                print(f"  RAG {name}: ERROR - {err_msg[:200]}")
            return {"error": err_msg, "coverage": None, "response_preview": None}

    # Run RAG with multiple configs (fairer comparison)
    if not args.rlm_only:
        results["rag_configs"] = {}

        # 1. RAG default: top_k=5, semantic chunking (original - may underperform)
        print("Running RAG (top_k=5, semantic chunking)...")
        rag1 = RAGPipeline(top_k=5)
        cfg1 = run_rag_config("top_k=5", rag1, "rag_top5_semantic")
        results["rag_configs"]["top_k=5_semantic"] = cfg1
        if cfg1.get("coverage") is not None:
            print(f"  RAG top_k=5: {cfg1['coverage']:.1%} ({cfg1['coverage'] * num_tabs:.0f}/{num_tabs} tabs)")

        # 2. RAG fair: top_k=num_tabs, semantic chunking (can exceed context limit on large docs)
        print(f"Running RAG (top_k={num_tabs}, semantic chunking)...")
        rag2 = RAGPipeline(top_k=num_tabs)
        cfg2 = run_rag_config("top_k=19", rag2, "rag_top19_semantic")
        results["rag_configs"][f"top_k={num_tabs}_semantic"] = cfg2
        if cfg2.get("coverage") is not None:
            print(f"  RAG top_k={num_tabs}: {cfg2['coverage']:.1%} ({cfg2['coverage'] * num_tabs:.0f}/{num_tabs} tabs)")

        # 3. RAG fairest: top_k=num_tabs, tab-level chunking (one chunk per tab - can exceed context limit)
        print(f"Running RAG (top_k={num_tabs}, tab-level chunking)...")
        rag3 = RAGPipeline(top_k=num_tabs, tab_contents=tabs)
        cfg3 = run_rag_config("top_k=19_tabs", rag3, "rag_top19_tabs", tab_ctx=tabs)
        results["rag_configs"][f"top_k={num_tabs}_tab_chunking"] = cfg3
        if cfg3.get("coverage") is not None:
            print(f"  RAG tab-level: {cfg3['coverage']:.1%} ({cfg3['coverage'] * num_tabs:.0f}/{num_tabs} tabs)")

        # Keep backward compat: primary "rag" = fairest config
        results["rag"] = results["rag_configs"][f"top_k={num_tabs}_tab_chunking"]

    # Run RLM
    if not args.rag_only:
        print("Running RLM...")
        rlm = RLMClient()
        rlm_response = rlm.completion(tabs, query)  # Pass dict for multi-tab
        rlm_coverage = compute_coverage(expected_tabs, rlm_response)
        results["rlm"] = {"response_preview": (rlm_response[:500] + "...") if len(rlm_response) > 500 else rlm_response, "coverage": rlm_coverage}
        print(f"RLM coverage: {rlm_coverage:.1%} ({rlm_coverage * len(expected_tabs):.0f}/{len(expected_tabs)} tabs)")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
