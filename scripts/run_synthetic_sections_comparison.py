"""Generate synthetic long docs and compare RAG vs RLM on section-order listing."""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.rag import RAGPipeline
from src.rlm import RLMClient


def generate_long_document(
    doc_id: int,
    num_sections: int = 20,
    target_chars: Optional[int] = None,
) -> Dict:
    """Generate one synthetic long document with numbered sections.
    If target_chars is set, scale content length to reach ~target_chars total."""
    words_per_section = 50
    if target_chars and target_chars > 0:
        chars_per_section = max(200, target_chars // num_sections)
        words_per_section = max(50, (chars_per_section - 60) // 9)

    sections = []
    for sec_id in range(num_sections):
        content = f"Section {sec_id} content: " + " ".join(
            random.choice(["analysis", "results", "discussion", "methods"])
            for _ in range(words_per_section)
        )
        sections.append(
            {
                "id": f"section_{sec_id}",
                "title": f"Section {sec_id}: Topic {random.randint(1, 100)}",
                "content": content,
            }
        )

    return {
        "doc_id": doc_id,
        "num_sections": num_sections,
        "sections": sections,
        "full_text": "\n\n".join([f"{s['title']}\n{s['content']}" for s in sections]),
    }


def extract_section_ids(text: str) -> List[str]:
    """Extract unique section IDs in first-appearance order."""
    seen = set()
    ordered = []
    for match in re.finditer(r"\bsection_(\d+)\b", text.lower()):
        sec = f"section_{int(match.group(1))}"
        if sec not in seen:
            seen.add(sec)
            ordered.append(sec)
    return ordered


def evaluate_order(expected: List[str], response: str) -> Dict:
    """Score whether response lists all section IDs in the correct order."""
    found = extract_section_ids(response)

    prefix_len = 0
    for exp, got in zip(expected, found):
        if exp == got:
            prefix_len += 1
        else:
            break

    set_recall = len(set(found) & set(expected)) / len(expected) if expected else 1.0
    ordered_exact = found == expected

    return {
        "ordered_exact": ordered_exact,
        "set_recall": set_recall,
        "prefix_accuracy": (prefix_len / len(expected)) if expected else 1.0,
        "found_ids": found,
        "expected_ids": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=5, help="Number of synthetic documents")
    parser.add_argument("--sections", type=int, default=20, help="Sections per document")
    parser.add_argument(
        "--target-chars",
        type=int,
        default=50000,
        help="Target chars per doc for long-context test (default 50k)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=20, help="RAG retrieval top-k (increase for long docs)")
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--rlm-only", action="store_true")
    parser.add_argument(
        "--save-docs",
        default="benchmarks/synthetic_long_docs.json",
        help="Where to save generated docs (repo-relative path)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    documents = [
        generate_long_document(i, args.sections, target_chars=args.target_chars)
        for i in range(args.docs)
    ]

    save_docs_path = Path(args.save_docs)
    if not save_docs_path.is_absolute():
        save_docs_path = Path(__file__).parent.parent / save_docs_path
    with open(save_docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    task = "List all section IDs in order"
    prompt = (
        f"{task}. Return ONLY a comma-separated list like "
        f"section_0, section_1, ..., section_{args.sections - 1}."
    )

    print(f"Generated {len(documents)} documents")
    print(
        f"Example: {documents[0]['num_sections']} sections, "
        f"{len(documents[0]['full_text']):,} chars (target: {args.target_chars:,})"
    )
    print(f"Saved docs to: {save_docs_path}")
    print("-" * 60)

    results = {
        "task": task,
        "prompt": prompt,
        "num_docs": args.docs,
        "num_sections": args.sections,
        "target_chars": args.target_chars,
        "actual_chars_per_doc": len(documents[0]["full_text"]) if documents else 0,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "documents_file": str(save_docs_path),
        "examples": [],
    }

    rag_exact = 0
    rlm_exact = 0
    rag_recall_sum = 0.0
    rlm_recall_sum = 0.0
    rag_prefix_sum = 0.0
    rlm_prefix_sum = 0.0

    for doc in documents:
        expected = [f"section_{i}" for i in range(doc["num_sections"])]
        ex = {"doc_id": doc["doc_id"], "chars": len(doc["full_text"])}
        print(f"\n--- Doc {doc['doc_id']} ({len(doc['full_text']):,} chars) ---")

        if not args.rlm_only:
            rag = RAGPipeline(top_k=args.top_k)
            rag_resp = rag.query(doc["full_text"], prompt, collection_name=f"synthetic_doc_{doc['doc_id']}")
            rag_eval = evaluate_order(expected, rag_resp)
            ex["rag"] = {
                "response_preview": rag_resp[:300] + ("..." if len(rag_resp) > 300 else ""),
                **rag_eval,
            }
            rag_exact += 1 if rag_eval["ordered_exact"] else 0
            rag_recall_sum += rag_eval["set_recall"]
            rag_prefix_sum += rag_eval["prefix_accuracy"]
            print(f"  RAG exact: {rag_eval['ordered_exact']} | recall: {rag_eval['set_recall']:.2f}")

        if not args.rag_only:
            rlm = RLMClient()
            rlm_resp = rlm.completion(doc["full_text"], prompt)
            rlm_eval = evaluate_order(expected, rlm_resp)
            ex["rlm"] = {
                "response_preview": rlm_resp[:300] + ("..." if len(rlm_resp) > 300 else ""),
                **rlm_eval,
            }
            rlm_exact += 1 if rlm_eval["ordered_exact"] else 0
            rlm_recall_sum += rlm_eval["set_recall"]
            rlm_prefix_sum += rlm_eval["prefix_accuracy"]
            print(f"  RLM exact: {rlm_eval['ordered_exact']} | recall: {rlm_eval['set_recall']:.2f}")

        results["examples"].append(ex)

    denom = len(documents) if documents else 1
    results["summary"] = {}
    if not args.rlm_only:
        results["summary"]["rag_ordered_exact_rate"] = rag_exact / denom
        results["summary"]["rag_avg_set_recall"] = rag_recall_sum / denom
        results["summary"]["rag_avg_prefix_accuracy"] = rag_prefix_sum / denom
    if not args.rag_only:
        results["summary"]["rlm_ordered_exact_rate"] = rlm_exact / denom
        results["summary"]["rlm_avg_set_recall"] = rlm_recall_sum / denom
        results["summary"]["rlm_avg_prefix_accuracy"] = rlm_prefix_sum / denom

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"synthetic_sections_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    if not args.rlm_only:
        print(
            "RAG  | exact: "
            f"{results['summary']['rag_ordered_exact_rate']:.1%}, "
            f"avg recall: {results['summary']['rag_avg_set_recall']:.2f}, "
            f"avg prefix: {results['summary']['rag_avg_prefix_accuracy']:.2f}"
        )
    if not args.rag_only:
        print(
            "RLM  | exact: "
            f"{results['summary']['rlm_ordered_exact_rate']:.1%}, "
            f"avg recall: {results['summary']['rlm_avg_set_recall']:.2f}, "
            f"avg prefix: {results['summary']['rlm_avg_prefix_accuracy']:.2f}"
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
