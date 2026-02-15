"""
Compare Iterative RAG vs Single-Pass RAG vs RLM on "list all" tasks.

Iterative RAG: Multi-pass retrieval without code aggregation.
- Pass 1: Retrieve top-k, LLM extracts section IDs
- Pass 2: Retrieve top-2k, LLM merges new IDs with previous list
- ... repeat until all retrieved or max_passes
- No code execution — pure retrieval + LLM aggregation

Addresses reviewer: "Why not retrieve in multiple passes?"
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

_rlm_log = __import__("logging").getLogger("rlm")
_rlm_log.setLevel(__import__("logging").WARNING)

from src.rag import RAGPipeline
from src.rlm import RLMClient
from src.evaluation import compute_coverage_sections, compute_coverage


def _extract_section_ids_from_chunk(chunk: str) -> list:
    """Extract section IDs from chunk (format: ## section_id or ## Tab_X_Name)."""
    m = re.match(r"^##\s*([^\n]+)", chunk.strip())
    if m:
        return [m.group(1).strip()]
    return []


def iterative_rag_query(
    rag: RAGPipeline,
    query: str,
    expected_ids: list,
    batch_size: int = 20,
    max_passes: int = 30,
    max_context_chars: int = 80000,
) -> tuple[str, dict]:
    """
    Multi-pass RAG: retrieve in batches, LLM aggregates incrementally.
    Returns (final_response, stats).
    """
    n_total = len(expected_ids)
    accumulated_ids = []
    stats = {"passes": 0, "chunks_per_pass": [], "context_chars_per_pass": []}

    for pass_i in range(max_passes):
        n_retrieve = min(batch_size * (pass_i + 1), n_total)
        chunks = rag.retrieve(query, n_results=n_retrieve)

        # ChromaDB returns top 1..n; new chunks = chunks[n_prev:]
        n_prev = batch_size * pass_i
        new_chunks = chunks[n_prev:] if n_prev < len(chunks) else chunks

        if not new_chunks and pass_i > 0:
            break

        stats["passes"] += 1
        stats["chunks_per_pass"].append(len(chunks))

        # Build merge prompt
        prev_list_str = ", ".join(accumulated_ids) if accumulated_ids else "(none yet)"
        new_context = "\n\n---\n\n".join(new_chunks[:50])  # Cap new chunks
        if len(new_context) > 30000:
            new_context = new_context[:30000] + "\n\n[truncated]"

        merge_prompt = f"""You are merging section IDs from multiple retrieval passes.

PREVIOUS LIST (from earlier passes): {prev_list_str}

NEW CHUNKS (from this pass):
{new_context}

TASK: Extract ALL section IDs from the new chunks. Merge with the previous list, preserving order (as they appear in the document). Return ONLY a comma-separated list of section IDs, no explanation.

COMPLETE LIST:"""

        try:
            from src.llm_client import chat_completion
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You extract and merge section IDs. Return only a comma-separated list."},
                    {"role": "user", "content": merge_prompt}
                ],
                model=rag.llm_model,
                max_tokens=4096,
            )
        except Exception as e:
            stats["error"] = str(e)
            return ", ".join(accumulated_ids), stats

        # Parse response: extract section IDs
        ids_found = re.findall(r"[\w\.\-]+\.\d+[\w\-]*|Tab_\d+_[\w_]+", response)
        # Dedupe while preserving order
        for sid in ids_found:
            if sid in expected_ids and sid not in accumulated_ids:
                accumulated_ids.append(sid)

        total_chars = sum(len(c) for c in chunks) + len(prev_list_str)
        stats["context_chars_per_pass"].append(total_chars)

        if len(accumulated_ids) >= n_total or n_retrieve >= n_total:
            break
        if total_chars > max_context_chars:
            break

    return ", ".join(accumulated_ids), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="legal", choices=["legal", "excel"])
    parser.add_argument("--download", action="store_true", help="Download legal benchmark first")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-passes", type=int, default=30)
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"

    if args.benchmark == "legal":
        if args.download:
            from benchmarks.download_legal_benchmark import download_legal_benchmark
            data = download_legal_benchmark(output_dir=benchmark_dir)
        else:
            with open(benchmark_dir / "legal_benchmark.json") as f:
                data = {"sections": json.load(f)}
            data["expected_section_ids"] = list(data["sections"].keys())
            data["text"] = (benchmark_dir / "legal_benchmark.txt").read_text(encoding="utf-8")
        sections = data["sections"]
        text = data["text"]
        expected_ids = data["expected_section_ids"]
        compute_cov = compute_coverage_sections
    else:
        with open(benchmark_dir / "data_layout_benchmark.json") as f:
            sections = json.load(f)
        text = (benchmark_dir / "data_layout_benchmark.txt").read_text(encoding="utf-8")
        expected_ids = list(sections.keys())
        compute_cov = compute_coverage

    query = "List all section numbers in order. Include every section from the document."
    if args.benchmark == "excel":
        query = "List all tab names in order. Include every tab from the document."

    n = len(expected_ids)
    results = {
        "benchmark": args.benchmark,
        "num_sections": n,
        "doc_size_chars": len(text),
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Single-pass RAG (top-k=20)
    print("Running Single-Pass RAG (top_k=20)...")
    try:
        rag_sp = RAGPipeline(top_k=20, tab_contents=sections)
        rag_sp.index(text="", collection_name="iter_rag_sp")
        ctx_sp = rag_sp.retrieve(query)
        r_sp = rag_sp.generate(query, ctx_sp)
        cov_sp = compute_cov(expected_ids, r_sp)
        results["single_pass_rag"] = {"coverage": cov_sp, "response_preview": r_sp[:300] + "..."}
        print(f"  Coverage: {cov_sp:.1%}")
    except Exception as e:
        results["single_pass_rag"] = {"error": str(e), "coverage": None}
        print(f"  ERROR: {str(e)[:100]}")

    # 2. Iterative RAG
    print("Running Iterative RAG (multi-pass, no code)...")
    try:
        rag_iter = RAGPipeline(top_k=20, tab_contents=sections)
        rag_iter.index(text="", collection_name="iter_rag_iter")
        r_iter, stats = iterative_rag_query(
            rag_iter, query, expected_ids,
            batch_size=args.batch_size,
            max_passes=args.max_passes,
        )
        cov_iter = compute_cov(expected_ids, r_iter)
        results["iterative_rag"] = {
            "coverage": cov_iter,
            "response_preview": r_iter[:300] + "...",
            "stats": stats,
        }
        print(f"  Coverage: {cov_iter:.1%} ({stats['passes']} passes)")
    except Exception as e:
        results["iterative_rag"] = {"error": str(e), "coverage": None}
        print(f"  ERROR: {str(e)[:100]}")

    # 3. RLM
    print("Running RLM...")
    try:
        rlm = RLMClient()
        r_rlm = rlm.completion(sections, query, context_json=sections)
        cov_rlm = compute_cov(expected_ids, r_rlm)
        results["rlm"] = {"coverage": cov_rlm, "response_preview": r_rlm[:300] + "..."}
        print(f"  Coverage: {cov_rlm:.1%}")
    except Exception as e:
        results["rlm"] = {"error": str(e), "coverage": None}
        print(f"  ERROR: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  Single-Pass RAG: {results.get('single_pass_rag', {}).get('coverage', 0):.1%}")
    print(f"  Iterative RAG:   {results.get('iterative_rag', {}).get('coverage', 0):.1%}")
    print(f"  RLM:             {results.get('rlm', {}).get('coverage', 0):.1%}")

    out_path = Path(__file__).parent.parent / "results" / f"iterative_rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
