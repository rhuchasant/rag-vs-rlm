"""Synthetic long-doc benchmark: structured section extraction for RAG vs RLM."""

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


TOPIC_CHOICES = ["analysis", "results", "discussion", "methods"]


def infer_main_topic(content: str) -> str:
    """Infer dominant topic token from synthetic section content."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", content.lower())
    counts = {t: 0 for t in TOPIC_CHOICES}
    for tok in tokens:
        if tok in counts:
            counts[tok] += 1
    # Stable tie-break by TOPIC_CHOICES order
    return max(TOPIC_CHOICES, key=lambda t: (counts[t], -TOPIC_CHOICES.index(t)))


def generate_long_document(
    doc_id: int,
    num_sections: int = 20,
    target_chars: Optional[int] = None,
    format_variant: str = "standard",
) -> Dict:
    """Generate synthetic document with section IDs, titles, and noisy content.
    If target_chars is set, scale content length to reach ~target_chars total."""
    # Base: ~50 chars overhead per section (title + prefix), ~9 chars per word
    words_per_section = 50
    if target_chars and target_chars > 0:
        chars_per_section = max(200, target_chars // num_sections)
        words_per_section = max(50, (chars_per_section - 60) // 9)

    sections = []
    for sec_id in range(num_sections):
        topic_num = random.randint(1, 100)
        topic_words = " ".join(random.choice(TOPIC_CHOICES) for _ in range(words_per_section))

        if format_variant == "alt_headers":
            title = f"Topic {topic_num} [Section {sec_id}]"
            content = (
                f"Section ID: section_{sec_id}\n"
                f"Main body for section_{sec_id}: {topic_words}\n"
                f"EndSection {sec_id}"
            )
        elif format_variant == "noisy":
            title = f"Section {sec_id}: Topic {topic_num}"
            noise = " ".join(random.choice(["meta", "note", "ref", "draft"]) for _ in range(20))
            content = (
                f"Section {sec_id} content: {topic_words}\n"
                f"[noise] {noise}\n"
                f"section_{sec_id} footer marker"
            )
        elif format_variant == "mixed_case":
            title = f"SECTION {sec_id}: TOPIC {topic_num}"
            content = f"SeCtIoN {sec_id} CoNtEnT: {topic_words}"
        else:
            title = f"Section {sec_id}: Topic {topic_num}"
            content = f"Section {sec_id} content: {topic_words}"

        sections.append(
            {
                "id": f"section_{sec_id}",
                "title": title,
                "content": content,
                "main_topic": infer_main_topic(content),
            }
        )

    return {
        "doc_id": doc_id,
        "num_sections": num_sections,
        "sections": sections,
        "full_text": "\n\n".join([f"{s['title']}\n{s['content']}" for s in sections]),
    }


def parse_structured_output(response: str) -> List[Dict[str, str]]:
    """Parse model response into a list of structured section records."""
    response = response.strip()
    if not response:
        return []

    # First try direct JSON parse
    try:
        obj = json.loads(response)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # Try extracting JSON list block from markdown/code fence
    match = re.search(r"\[[\s\S]*\]", response)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass

    return []


def normalize_record(rec: Dict) -> Optional[Dict[str, str]]:
    """Normalize keys to canonical schema: section_id, title, main_topic."""
    key_map = {
        "section_id": "section_id",
        "id": "section_id",
        "section": "section_id",
        "title": "title",
        "section_title": "title",
        "main_topic": "main_topic",
        "topic": "main_topic",
    }
    out: Dict[str, str] = {}
    for k, v in rec.items():
        if not isinstance(k, str):
            continue
        canon = key_map.get(k.strip().lower())
        if canon is None:
            continue
        out[canon] = str(v).strip()

    if "section_id" not in out:
        return None
    out.setdefault("title", "")
    out.setdefault("main_topic", "")
    return out


def canonical_section_id(value: str) -> str:
    """Normalize IDs like '0', 'Section 0', 'section_0' -> 'section_0'."""
    s = str(value).strip().lower()
    m = re.search(r"(\d+)", s)
    if m:
        return f"section_{int(m.group(1))}"
    return s


def normalize_title(value: str) -> str:
    """Normalize title variants to compare semantically."""
    s = str(value).strip().lower()
    s = re.sub(r"^section\s*\d+\s*:\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def evaluate_structured(expected_sections: List[Dict], response: str) -> Dict:
    """Evaluate structured extraction quality."""
    parsed = parse_structured_output(response)
    normalized = [x for x in (normalize_record(r) for r in parsed) if x is not None]

    expected_by_id = {s["id"]: s for s in expected_sections}
    expected_ids_order = [s["id"] for s in expected_sections]

    pred_ids = [canonical_section_id(r["section_id"]) for r in normalized]
    pred_by_id = {canonical_section_id(r["section_id"]): r for r in normalized}

    matched_ids = [sid for sid in expected_ids_order if sid in pred_by_id]
    completeness = len(matched_ids) / len(expected_sections) if expected_sections else 1.0

    structure_valid = 0
    for r in normalized:
        if r.get("section_id") and "title" in r and "main_topic" in r:
            structure_valid += 1
    structure_valid_rate = structure_valid / len(normalized) if normalized else 0.0

    title_correct = 0
    topic_correct = 0
    for sid in matched_ids:
        pred = pred_by_id[sid]
        exp = expected_by_id[sid]
        if normalize_title(pred.get("title", "")) == normalize_title(exp["title"]):
            title_correct += 1
        if pred.get("main_topic", "").strip().lower() == exp["main_topic"].strip().lower():
            topic_correct += 1

    denom = len(matched_ids) if matched_ids else 1
    title_accuracy = title_correct / denom if matched_ids else 0.0
    topic_accuracy = topic_correct / denom if matched_ids else 0.0

    exact_order = pred_ids == expected_ids_order
    overall_exact = (
        exact_order
        and completeness == 1.0
        and title_accuracy == 1.0
        and topic_accuracy == 1.0
        and len(normalized) == len(expected_sections)
    )

    return {
        "overall_exact": overall_exact,
        "completeness_recall": completeness,
        "structure_valid_rate": structure_valid_rate,
        "title_accuracy_on_matched": title_accuracy,
        "topic_accuracy_on_matched": topic_accuracy,
        "pred_count": len(normalized),
        "matched_count": len(matched_ids),
        "expected_count": len(expected_sections),
    }


def run_model_response(model_name: str, text: str, prompt: str, top_k: int, doc_id: int) -> str:
    """Run one model and return raw text response."""
    if model_name == "rag":
        rag = RAGPipeline(top_k=top_k)
        return rag.query(text, prompt, collection_name=f"synthetic_structured_{doc_id}")
    rlm = RLMClient()
    return rlm.completion(text, prompt)


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
    parser.add_argument(
        "--format-variant",
        default="standard",
        choices=["standard", "alt_headers", "noisy", "mixed_case"],
        help="Synthetic document format variation for robustness testing",
    )
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--rlm-only", action="store_true")
    parser.add_argument("--save-docs", default="benchmarks/synthetic_long_docs_structured.json")
    args = parser.parse_args()

    random.seed(args.seed)
    docs = [
        generate_long_document(
            i,
            args.sections,
            target_chars=args.target_chars,
            format_variant=args.format_variant,
        )
        for i in range(args.docs)
    ]

    docs_path = Path(args.save_docs)
    if not docs_path.is_absolute():
        docs_path = Path(__file__).parent.parent / docs_path
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    task = (
        "For each section, extract: section ID, title, and main topic. "
        "Return as a JSON list in section order."
    )
    prompt = (
        f"{task}\n\n"
        "Output requirements:\n"
        "- Return ONLY valid JSON.\n"
        '- JSON must be a list of objects with keys: "section_id", "title", "main_topic".\n'
        "- Preserve section order from section_0 to section_n.\n"
        "- main_topic must be one of: analysis, results, discussion, methods.\n"
    )

    print(f"Generated {len(docs)} docs, each with {args.sections} sections")
    print(f"Target size: {args.target_chars:,} chars/doc (actual: {len(docs[0]['full_text']):,} chars)")
    print(f"Saved docs to: {docs_path}")
    print("-" * 60)

    results = {
        "task": task,
        "prompt": prompt,
        "num_docs": args.docs,
        "num_sections": args.sections,
        "target_chars": args.target_chars,
        "format_variant": args.format_variant,
        "actual_chars_per_doc": len(docs[0]["full_text"]) if docs else 0,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "documents_file": str(docs_path),
        "examples": [],
    }

    agg = {
        "rag": {"overall_exact": 0, "completeness": 0.0, "structure": 0.0, "title_acc": 0.0, "topic_acc": 0.0},
        "rlm": {"overall_exact": 0, "completeness": 0.0, "structure": 0.0, "title_acc": 0.0, "topic_acc": 0.0},
    }

    for doc in docs:
        ex = {"doc_id": doc["doc_id"], "chars": len(doc["full_text"])}
        print(f"\n--- Doc {doc['doc_id']} ({len(doc['full_text']):,} chars) ---")

        if not args.rlm_only:
            rag_resp = run_model_response("rag", doc["full_text"], prompt, args.top_k, doc["doc_id"])
            rag_eval = evaluate_structured(doc["sections"], rag_resp)
            ex["rag"] = {
                "response_preview": rag_resp[:300] + ("..." if len(rag_resp) > 300 else ""),
                **rag_eval,
            }
            agg["rag"]["overall_exact"] += 1 if rag_eval["overall_exact"] else 0
            agg["rag"]["completeness"] += rag_eval["completeness_recall"]
            agg["rag"]["structure"] += rag_eval["structure_valid_rate"]
            agg["rag"]["title_acc"] += rag_eval["title_accuracy_on_matched"]
            agg["rag"]["topic_acc"] += rag_eval["topic_accuracy_on_matched"]
            print(
                "  RAG | exact: "
                f"{rag_eval['overall_exact']} | comp: {rag_eval['completeness_recall']:.2f} | "
                f"struct: {rag_eval['structure_valid_rate']:.2f}"
            )

        if not args.rag_only:
            rlm_resp = run_model_response("rlm", doc["full_text"], prompt, args.top_k, doc["doc_id"])
            rlm_eval = evaluate_structured(doc["sections"], rlm_resp)
            ex["rlm"] = {
                "response_preview": rlm_resp[:300] + ("..." if len(rlm_resp) > 300 else ""),
                **rlm_eval,
            }
            agg["rlm"]["overall_exact"] += 1 if rlm_eval["overall_exact"] else 0
            agg["rlm"]["completeness"] += rlm_eval["completeness_recall"]
            agg["rlm"]["structure"] += rlm_eval["structure_valid_rate"]
            agg["rlm"]["title_acc"] += rlm_eval["title_accuracy_on_matched"]
            agg["rlm"]["topic_acc"] += rlm_eval["topic_accuracy_on_matched"]
            print(
                "  RLM | exact: "
                f"{rlm_eval['overall_exact']} | comp: {rlm_eval['completeness_recall']:.2f} | "
                f"struct: {rlm_eval['structure_valid_rate']:.2f}"
            )

        results["examples"].append(ex)

    denom = len(docs) if docs else 1
    summary: Dict[str, float] = {}
    if not args.rlm_only:
        summary["rag_overall_exact_rate"] = agg["rag"]["overall_exact"] / denom
        summary["rag_avg_completeness_recall"] = agg["rag"]["completeness"] / denom
        summary["rag_avg_structure_valid_rate"] = agg["rag"]["structure"] / denom
        summary["rag_avg_title_accuracy_on_matched"] = agg["rag"]["title_acc"] / denom
        summary["rag_avg_topic_accuracy_on_matched"] = agg["rag"]["topic_acc"] / denom
    if not args.rag_only:
        summary["rlm_overall_exact_rate"] = agg["rlm"]["overall_exact"] / denom
        summary["rlm_avg_completeness_recall"] = agg["rlm"]["completeness"] / denom
        summary["rlm_avg_structure_valid_rate"] = agg["rlm"]["structure"] / denom
        summary["rlm_avg_title_accuracy_on_matched"] = agg["rlm"]["title_acc"] / denom
        summary["rlm_avg_topic_accuracy_on_matched"] = agg["rlm"]["topic_acc"] / denom
    results["summary"] = summary

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"synthetic_structured_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    if not args.rlm_only:
        print(
            "RAG  | exact: "
            f"{summary['rag_overall_exact_rate']:.1%}, "
            f"comp: {summary['rag_avg_completeness_recall']:.2f}, "
            f"struct: {summary['rag_avg_structure_valid_rate']:.2f}, "
            f"title: {summary['rag_avg_title_accuracy_on_matched']:.2f}, "
            f"topic: {summary['rag_avg_topic_accuracy_on_matched']:.2f}"
        )
    if not args.rag_only:
        print(
            "RLM  | exact: "
            f"{summary['rlm_overall_exact_rate']:.1%}, "
            f"comp: {summary['rlm_avg_completeness_recall']:.2f}, "
            f"struct: {summary['rlm_avg_structure_valid_rate']:.2f}, "
            f"title: {summary['rlm_avg_title_accuracy_on_matched']:.2f}, "
            f"topic: {summary['rlm_avg_topic_accuracy_on_matched']:.2f}"
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
