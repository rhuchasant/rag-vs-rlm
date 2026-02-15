"""
Recall Analysis: Empirical validation of theoretical P(full coverage) ≈ r^N.

Theoretical model:
- N = number of sections
- k = retrieval size (top-k)
- r = k/N (per-section retrieval probability under uniform retrieval)
- P(full coverage) ≈ r^N (exponential decay)

This script:
1. Runs RAG retrieval (no LLM) for "list all" query
2. Measures which sections are retrieved
3. Computes empirical recall and P(full coverage)
4. Varies N (subsample document) to show scaling
5. Compares empirical vs theoretical
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag import RAGPipeline


def extract_section_ids_from_chunks(chunks: list, section_ids: list) -> set:
    """Map retrieved chunks back to section IDs. Chunks are '## section_id\\ncontent'."""
    retrieved = set()
    for chunk in chunks:
        if not chunk:
            continue
        # Match "## 240.0-1" or "## Tab_1_Name" at start (primary chunk format)
        m = re.match(r"^##\s*([^\n]+)", chunk.strip())
        if m:
            sid = m.group(1).strip()
            if sid in section_ids:
                retrieved.add(sid)
    return retrieved


def run_recall_for_n(
    sections: dict,
    expected_ids: list,
    query: str,
    top_k: int,
    collection_prefix: str,
) -> dict:
    """Run RAG retrieval, return recall stats (no LLM generation)."""
    rag = RAGPipeline(top_k=top_k, tab_contents=sections)
    rag.index(text="", collection_name=collection_prefix)
    chunks = rag.retrieve(query)

    retrieved_ids = extract_section_ids_from_chunks(chunks, expected_ids)
    recall_count = len(retrieved_ids)
    n = len(expected_ids)
    recall_frac = recall_count / n if n else 0
    full_coverage = recall_count == n

    return {
        "n": n,
        "k": top_k,
        "retrieved_count": recall_count,
        "recall_frac": recall_frac,
        "full_coverage": full_coverage,
        "r_empirical": recall_count / n if n else 0,  # Actual retrieval rate
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="legal", choices=["legal", "excel"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-values", type=str, default="30,50,100,200,500", help="Comma-separated N values to test")
    parser.add_argument("--query-variants", type=int, default=5, help="Number of query phrasings for empirical P(full)")
    parser.add_argument("--download", action="store_true", help="Download legal benchmark first")
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    n_values = [int(x.strip()) for x in args.n_values.split(",")]

    # Load benchmark
    if args.benchmark == "legal":
        if args.download:
            from benchmarks.download_legal_benchmark import download_legal_benchmark
            data = download_legal_benchmark(output_dir=benchmark_dir)
        else:
            with open(benchmark_dir / "legal_benchmark.json") as f:
                data = {"sections": json.load(f)}
            data["expected_section_ids"] = list(data["sections"].keys())
        full_sections = data["sections"]
        full_ids = data["expected_section_ids"]
    else:
        with open(benchmark_dir / "data_layout_benchmark.json") as f:
            full_sections = json.load(f)
        full_ids = list(full_sections.keys())

    query_variants = [
        "List all section numbers in order. Include every section from the document.",
        "List every section ID in the document.",
        "What are all the section numbers? List them in order.",
        "Extract all section identifiers from the document.",
        "List all sections. Include every one.",
    ][: args.query_variants]

    results = {
        "benchmark": args.benchmark,
        "top_k": args.top_k,
        "n_values": n_values,
        "timestamp": datetime.now().isoformat(),
        "runs": [],
        "summary": {},
    }

    print("Recall Analysis: RAG retrieval (no LLM)")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark} | top_k: {args.top_k}")
    print(f"N values: {n_values}")
    print()

    for n in n_values:
        if n > len(full_ids):
            print(f"  N={n}: Skipped (max {len(full_ids)})")
            continue

        # Subsample first N sections
        section_ids = full_ids[:n]
        sections = {sid: full_sections[sid] for sid in section_ids}

        # Theoretical: r = k/N, P(full) = r^N
        r_theory = min(1.0, args.top_k / n)
        p_full_theory = r_theory ** n

        full_coverage_count = 0
        recall_fracs = []

        for i, query in enumerate(query_variants):
            run = run_recall_for_n(
                sections=sections,
                expected_ids=section_ids,
                query=query,
                top_k=args.top_k,
                collection_prefix=f"recall_{args.benchmark}_n{n}_q{i}",
            )
            run["query"] = query[:50] + "..."
            run["p_full_theory"] = p_full_theory
            results["runs"].append(run)

            if run["full_coverage"]:
                full_coverage_count += 1
            recall_fracs.append(run["recall_frac"])

        p_full_empirical = full_coverage_count / len(query_variants)
        avg_recall = sum(recall_fracs) / len(recall_fracs) if recall_fracs else 0

        results["summary"][str(n)] = {
            "p_full_theory": p_full_theory,
            "p_full_empirical": p_full_empirical,
            "avg_recall_frac": avg_recall,
            "full_coverage_count": full_coverage_count,
            "n_queries": len(query_variants),
        }

        print(f"  N={n:4d} | recall: {avg_recall:.1%} | P(full) theory: {p_full_theory:.2e} | P(full) empirical: {p_full_empirical:.1%}")

    # Save
    out_path = Path(__file__).parent.parent / "results" / f"recall_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
