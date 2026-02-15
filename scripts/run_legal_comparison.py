"""
Run RLM vs RAG comparison on legal document benchmark (CFR-style).
Task: List all section numbers in order.
Tests hypothesis: RLM outperforms RAG when document exceeds context.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

_rlm_log = logging.getLogger("rlm")
_rlm_log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
_rlm_log.addHandler(_h)

from src.rag import RAGPipeline
from src.rlm import RLMClient
from src.evaluation import compute_coverage_sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download real CFR from GovInfo")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic benchmark (fallback)")
    parser.add_argument("--part", type=int, default=240)
    parser.add_argument("--sections", type=int, default=50)
    parser.add_argument("--size", type=int, default=750000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--rlm-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    if args.download:
        from benchmarks.download_legal_benchmark import download_legal_benchmark
        data = download_legal_benchmark(part_number=args.part, output_dir=benchmark_dir)
        sections = data["sections"]
        text = data["text"]
        expected_ids = data["expected_section_ids"]
    elif args.generate:
        from benchmarks.generate_legal_benchmark import generate_legal_benchmark
        data = generate_legal_benchmark(
            part_number=args.part,
            num_sections=args.sections,
            target_total_chars=args.size,
            seed=args.seed,
            output_dir=benchmark_dir,
        )
        sections = data["sections"]
        text = data["text"]
        expected_ids = data["expected_section_ids"]
    else:
        # Use existing benchmark (from prior --download or --generate)
        json_path = benchmark_dir / "legal_benchmark.json"
        txt_path = benchmark_dir / "legal_benchmark.txt"
        if not json_path.exists() or not txt_path.exists():
            print("Run with --generate first to create the benchmark")
            sys.exit(1)
        with open(json_path) as f:
            sections = json.load(f)
        text = txt_path.read_text(encoding="utf-8")
        expected_ids = sorted(sections.keys(), key=lambda s: (int(s.split(".")[0]), int(s.split(".")[1])))

    num_sections = len(expected_ids)
    query = "List all section numbers in order. Include every section from the document."

    results = {
        "query": query,
        "num_sections": num_sections,
        "expected_section_ids": expected_ids[:10] + ["..."] if len(expected_ids) > 10 else expected_ids,
        "doc_size_chars": len(text),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    def run_rag(name: str, top_k: int, use_section_chunking: bool):
        try:
            rag = RAGPipeline(top_k=top_k)
            if use_section_chunking:
                # One chunk per section (fair: same granularity as RLM)
                r = rag.query(text, query, collection_name=name, tab_contents=sections)
            else:
                r = rag.query(text, query, collection_name=name)
            cov = compute_coverage_sections(expected_ids, r)
            return {"coverage": cov, "response_preview": (r[:400] + "...") if len(r) > 400 else r}
        except Exception as e:
            err = str(e)
            is_ctx = "context_length" in err.lower() or "context length" in err.lower()
            print(f"  RAG {name}: ERROR - {'context length exceeded' if is_ctx else err[:150]}")
            return {"error": err, "coverage": None, "response_preview": None}

    if not args.rlm_only:
        results["rag_configs"] = {}
        print("Running RAG (top_k=20, section-level chunking)...")
        cfg = run_rag("legal_top20_sections", args.top_k, use_section_chunking=True)
        results["rag_configs"]["top_k_sections"] = cfg
        if cfg.get("coverage") is not None:
            print(f"  RAG: {cfg['coverage']:.1%} ({cfg['coverage'] * num_sections:.0f}/{num_sections} sections)")

        # Try full retrieval (may exceed context)
        print(f"Running RAG (top_k={num_sections}, section-level)...")
        cfg_full = run_rag("legal_full_sections", num_sections, use_section_chunking=True)
        results["rag_configs"]["top_k_full"] = cfg_full
        if cfg_full.get("coverage") is not None:
            print(f"  RAG full: {cfg_full['coverage']:.1%}")
        elif cfg_full.get("error"):
            print(f"  RAG full: FAILED (context limit)")

    if not args.rag_only:
        print("Running RLM...")
        rlm = RLMClient()
        try:
            r_rlm = rlm.completion(sections, query, context_json=sections)
            cov = compute_coverage_sections(expected_ids, r_rlm)
            results["rlm"] = {
                "coverage": cov,
                "response_preview": (r_rlm[:400] + "...") if len(r_rlm) > 400 else r_rlm,
            }
            print(f"  RLM: {cov:.1%} ({cov * num_sections:.0f}/{num_sections} sections)")
        except Exception as e:
            results["rlm"] = {"error": str(e), "coverage": None}
            print(f"  RLM: ERROR - {str(e)[:150]}")

    # Save
    out_path = Path(__file__).parent.parent / "results" / f"legal_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
