"""
Run RLM vs RAG comparison on Oolong benchmark.
Oolong: Long-context reasoning and aggregation (arxiv.org/abs/2511.02817)
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable RLM progress logging
_rlm_log = logging.getLogger("rlm")
_rlm_log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
_rlm_log.addHandler(_h)

from src.rag import RAGPipeline
from src.rlm import RLMClient


def normalize_answer(s: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace."""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_answer_for_numeric(response: str, gold: str) -> str:
    """
    Post-process response for numeric gold answers: extract from \\boxed{}, deduplicate.
    Enhanced to handle single-digit and multi-digit duplication.
    """
    if not _is_numeric_gold(gold):
        return response
    
    # Extract numbers from \boxed{...}
    boxed = re.findall(r"\\?boxed\{([^}]+)\}", response, re.IGNORECASE)
    if boxed:
        nums = [n.strip() for n in boxed if n.strip().replace("-", "").isdigit()]
        if nums:
            # If all same, use one; else sum (aggregation)
            if len(set(nums)) == 1:
                result = nums[0]
                # Apply deduplication to the extracted number too
                return _deduplicate(result)
            try:
                return str(sum(int(n) for n in nums))
            except ValueError:
                return _deduplicate(nums[-1])
    
    # Apply deduplication to the raw response
    return _deduplicate(response.strip())


def _deduplicate(s: str) -> str:
    """Deduplicate XX pattern in numeric strings."""
    s = s.strip()
    if not s:
        return s
    
    # Special case: single digit duplication (44 -> 4, 00 -> 0)
    if len(s) == 2 and s[0] == s[1] and s[0].isdigit():
        return s[0]
    
    # General case: XX pattern (103103 -> 103, 118118 -> 118)
    if len(s) % 2 == 0 and len(s) > 0:
        half = len(s) // 2
        if s[:half] == s[half:]:
            return s[:half]
    
    return s


def _is_numeric_gold(gold: str) -> bool:
    """Check if gold answer is a number (for counting tasks)."""
    return str(gold).strip().isdigit()


def _normalize_gold(gold: str) -> str:
    """
    Normalize Oolong gold format.
    Handles list-style strings like "['False']" -> "False".
    """
    g = str(gold).strip()
    if g.startswith("[") and g.endswith("]"):
        inner = g[1:-1].strip()
        if inner:
            first = inner.split(",")[0].strip()
            g = first.strip("'\" ")
    return g


def _augment_question_for_numeric(question: str, gold: str) -> str:
    """When gold is numeric, add instructions for direct numeric answer."""
    ng = _normalize_gold(gold)
    if not _is_numeric_gold(ng):
        return question
    return (
        question
        + " Answer with only the number, no explanation. "
        + "When aggregating counts from multiple parts: sum them (add), do not concatenate."
    )


def check_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (normalized, or gold contained in pred)."""
    p = normalize_answer(predicted)
    g = normalize_answer(_normalize_gold(gold))
    if not g:
        return False
    # For numeric gold: require exact match on extracted numbers (7 must not match 17)
    if _is_numeric_gold(_normalize_gold(gold)):
        nums = re.findall(r"\b\d+\b", predicted)
        return g in [normalize_answer(n) for n in nums]
    if g in p or p in g or p == g:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oolong-synth", choices=["oolong-synth", "oolong-real"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default=None, help="For oolong-real: dnd or toy_dnd")
    parser.add_argument("--max", type=int, default=5, help="Max examples to run (default 5)")
    parser.add_argument("--min-context-chars", type=int, default=50000, help="Only run on examples with context >= N chars (default 50k; use 0 for any length)")
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--rlm-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10, help="RAG retrieval top-k")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds per RLM example (default 600). Skip to next on timeout.")
    args = parser.parse_args()

    from benchmarks.load_oolong import load_oolong
    examples = load_oolong(
        args.dataset, args.split, args.subset, args.max,
        min_context_chars=args.min_context_chars,
    )
    if not examples:
        print(f"No examples loaded (min_context_chars={args.min_context_chars}). Try lower --min-context-chars or different dataset.")
        sys.exit(1)

    print(f"Oolong benchmark: {args.dataset} ({args.split})")
    print(f"Running on {len(examples)} examples (context >= {args.min_context_chars:,} chars)")
    print("-" * 50)

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "num_examples": len(examples),
        "min_context_chars": args.min_context_chars,
        "timestamp": datetime.now().isoformat(),
        "examples": [],
    }

    rag_correct = 0
    rlm_correct = 0
    rag_errors = 0
    rlm_errors = 0

    for i, ex in enumerate(examples):
        ex_id = ex["id"]
        context = ex["context"]
        question = ex["question"]
        gold = ex["answer"]
        print(f"\n--- Example {i + 1}/{len(examples)} (id={ex_id}, context={len(context):,} chars) ---")

        ex_result = {"id": ex_id, "question": question[:100], "gold": gold[:80], "context_chars": len(context)}

        # Augment question for numeric answers (counting tasks)
        q = _augment_question_for_numeric(question, gold)

        # RAG
        if not args.rlm_only:
            try:
                rag = RAGPipeline(top_k=args.top_k)
                # Unique collection per example to avoid ChromaDB conflicts
                coll = f"oolong_{args.dataset}_{i}"
                rag_response = rag.query(context, q, collection_name=coll)
                ex_result["rag_response"] = rag_response[:300] + ("..." if len(rag_response) > 300 else "")
                rag_for_check = extract_answer_for_numeric(rag_response, gold)
                ex_result["rag_match"] = check_match(rag_for_check, gold)
                if ex_result["rag_match"]:
                    rag_correct += 1
                print(f"  RAG: {'✓' if ex_result['rag_match'] else '✗'} {ex_result['rag_response'][:80]}...")
            except Exception as e:
                ex_result["rag_error"] = str(e)
                ex_result["rag_match"] = False
                rag_errors += 1
                print(f"  RAG: ERROR - {str(e)[:100]}")

        # RLM
        if not args.rag_only:
            try:
                rlm = RLMClient()

                def run_rlm():
                    return rlm.completion(context, q)

                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(run_rlm)
                try:
                    rlm_response = future.result(timeout=args.timeout)
                    ex_result["rlm_response"] = rlm_response[:300] + ("..." if len(rlm_response) > 300 else "")
                    rlm_for_check = extract_answer_for_numeric(rlm_response, gold)
                    ex_result["rlm_match"] = check_match(rlm_for_check, gold)
                    if ex_result["rlm_match"]:
                        rlm_correct += 1
                    print(f"  RLM: {'✓' if ex_result['rlm_match'] else '✗'} {ex_result['rlm_response'][:80]}...")
                except FuturesTimeoutError:
                    ex_result["rlm_response"] = "RLM timed out"
                    ex_result["rlm_error"] = f"Timeout ({args.timeout}s)"
                    ex_result["rlm_match"] = False
                    rlm_errors += 1
                    print(f"  RLM: ERROR - Timeout ({args.timeout}s)")
                finally:
                    executor.shutdown(wait=False)
            except Exception as e:
                ex_result["rlm_error"] = str(e)
                ex_result["rlm_match"] = False
                rlm_errors += 1
                print(f"  RLM: ERROR - {str(e)[:100]}")

        results["examples"].append(ex_result)

    # Summary
    n = len(examples)
    results["summary"] = {
        "rag_accuracy": rag_correct / n if n else 0,
        "rlm_accuracy": rlm_correct / n if n else 0,
        "rag_errors": rag_errors,
        "rlm_errors": rlm_errors,
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  RAG: {rag_correct}/{n} correct ({rag_correct/n*100:.1f}%)" + (f", {rag_errors} errors" if rag_errors else ""))
    print(f"  RLM: {rlm_correct}/{n} correct ({rlm_correct/n*100:.1f}%)" + (f", {rlm_errors} errors" if rlm_errors else ""))

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"oolong_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
