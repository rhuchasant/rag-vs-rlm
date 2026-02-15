"""
Run Hybrid RLM + RAG comparison on legal document benchmark.
Mixed tasks: 1 broad (list all sections) + 5 targeted (specific lookups).
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
_rlm_log.setLevel(logging.WARNING)

from src.rag import RAGPipeline
from src.rlm import RLMClient
from src.evaluation import compute_coverage_sections
from src.hybrid import HybridQueryEngine


def _check_targeted_match(response: str, expected: str | list) -> bool:
    """Check if expected value(s) appear in response."""
    r = response.lower()
    if isinstance(expected, str):
        return expected.lower() in r
    return any(e.lower() in r for e in expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download real CFR first")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    if args.download:
        from benchmarks.download_legal_benchmark import download_legal_benchmark
        data = download_legal_benchmark(output_dir=benchmark_dir)
        sections = data["sections"]
        text = data["text"]
        expected_ids = data["expected_section_ids"]
    else:
        json_path = benchmark_dir / "legal_benchmark.json"
        txt_path = benchmark_dir / "legal_benchmark.txt"
        if not json_path.exists() or not txt_path.exists():
            print("Run with --download first to create the benchmark")
            sys.exit(1)
        with open(json_path) as f:
            sections = json.load(f)
        text = txt_path.read_text(encoding="utf-8")
        expected_ids = list(sections.keys())  # Preserve order from download

    # Pick sections for targeted tasks (ensure they exist)
    keys = list(sections.keys())
    sec_1 = keys[0]  # 240.0-1
    sec_9 = "240.0-9" if "240.0-9" in sections else keys[min(8, len(keys) - 1)]
    sec_def = "240.0-1"  # Definitions
    sec_filing = "240.0-9" if "240.0-9" in sections else keys[min(8, len(keys) - 1)]  # Payment of filing fees

    # Build ground truth from section content (first sentence/topic)
    def _first_topic(content: str) -> str:
        """Extract main topic from content (e.g., 'Definitions.. Definitions.' -> 'Definitions')."""
        parts = content.split("..")[0].strip().split(".")
        return (parts[0] if parts else content[:50]).strip()

    tasks = [
        {
            "id": "broad_list_sections",
            "query": "List all section numbers in order. Include every section from the document.",
            "type": "broad",
            "expected_ids": expected_ids,
        },
        {
            "id": "targeted_subject_1",
            "query": f"What is the subject of section {sec_1}?",
            "type": "targeted",
            "expected": [_first_topic(sections[sec_1]), "Definitions"],
        },
        {
            "id": "targeted_subject_9",
            "query": f"What does section {sec_filing} cover?",
            "type": "targeted",
            "expected": ["filing fees", "payment", _first_topic(sections.get(sec_filing, ""))],
        },
        {
            "id": "targeted_which_section",
            "query": "Which section covers Definitions?",
            "type": "targeted",
            "expected": ["240.0-1", sec_def],
        },
        {
            "id": "targeted_which_filing",
            "query": "Which section covers payment of filing fees?",
            "type": "targeted",
            "expected": ["240.0-9", sec_filing, "filing fees"],
        },
        {
            "id": "targeted_dealer_exemption",
            "query": "Which section exempts banks from the dealer definition for riskless principal transactions?",
            "type": "targeted",
            "expected": ["240.3a5-1", "3a5-1", "riskless principal"],
        },
    ]

    results = {
        "num_sections": len(sections),
        "doc_size_chars": len(text),
        "timestamp": datetime.now().isoformat(),
        "tasks": [],
    }

    rag = RAGPipeline(top_k=args.top_k)
    rlm = RLMClient()
    hybrid_engine = HybridQueryEngine(rag, rlm)

    for task in tasks:
        q = task["query"]
        tid = task["id"]
        ttype = task["type"]
        print(f"\n--- Task: {tid} ({ttype}) ---")

        ex = {"task_id": tid, "type": ttype, "query": q[:70] + "..."}

        # RAG-only (section-level chunking)
        try:
            r_rag = rag.query(text, q, collection_name=f"legal_hybrid_{tid}", tab_contents=sections)
            if ttype == "broad":
                cov = compute_coverage_sections(task["expected_ids"], r_rag)
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
            r_rlm = rlm.completion(sections, q, context_json=sections)
            if ttype == "broad":
                cov = compute_coverage_sections(task["expected_ids"], r_rlm)
                ex["rlm_match"] = cov >= 1.0
                ex["rlm_coverage"] = cov
            else:
                ex["rlm_match"] = _check_targeted_match(r_rlm, task["expected"])
            ex["rlm_response"] = (r_rlm[:200] + "...") if len(r_rlm) > 200 else r_rlm
        except Exception as e:
            ex["rlm_match"] = False
            ex["rlm_error"] = str(e)[:150]
            ex["rlm_response"] = None

        # Hybrid
        try:
            h_out = hybrid_engine.query(q, sections, text, task_id=tid)
            r_hyb = h_out["response"] or ""
            if ttype == "broad":
                cov = compute_coverage_sections(task["expected_ids"], r_hyb)
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

    n = len(results["tasks"])
    rag_ok = sum(1 for t in results["tasks"] if t.get("rag_match"))
    rlm_ok = sum(1 for t in results["tasks"] if t.get("rlm_match"))
    hyb_ok = sum(1 for t in results["tasks"] if t.get("hybrid_match"))

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
    print(f"SUMMARY: RAG {rag_ok}/{n} | RLM {rlm_ok}/{n} | Hybrid {hyb_ok}/{n}")

    out_path = Path(__file__).parent.parent / "results" / f"legal_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
