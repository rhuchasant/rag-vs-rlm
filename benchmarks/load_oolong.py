"""
Load Oolong benchmark from Hugging Face.
Oolong: Long-context reasoning and aggregation (arxiv.org/abs/2511.02817)

Uses streaming by default: fetches examples on-demand from HF servers, minimal local cache.
If disk full: set HF_HOME to a drive with space, e.g. $env:HF_HOME="D:\\cache\\hf"
"""

from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


def load_oolong(
    dataset: str = "oolong-synth",
    split: str = "test",
    subset: Optional[str] = None,
    max_examples: Optional[int] = None,
    min_context_chars: int = 0,
    streaming: bool = True,
) -> List[Dict]:
    """
    Load Oolong benchmark examples from Hugging Face.

    Args:
        dataset: "oolong-synth" or "oolong-real"
        split: "test" or "validation" (oolong-real has both)
        subset: For oolong-real, "dnd" or "toy_dnd"
        max_examples: Limit number of examples (for faster runs)
        min_context_chars: Skip examples with context shorter than this (for long-context runs)
        streaming: If True, fetch on-demand from HF (no full local cache, saves disk)

    Returns:
        List of {"id", "context", "question", "answer", "question_type", ...}
    """
    hf_name = f"oolongbench/{dataset}"
    kwargs = {"split": split}
    if subset and dataset == "oolong-real":
        kwargs["name"] = subset

    if streaming:
        kwargs["streaming"] = True

    try:
        ds = load_dataset(hf_name, **kwargs)
    except Exception as e:
        if streaming:
            # Fallback: some datasets may not support streaming
            kwargs.pop("streaming", None)
            ds = load_dataset(hf_name, **kwargs)
        else:
            raise

    examples = []
    for i, row in enumerate(ds):
        # Handle column name variations (oolong-synth vs oolong-real)
        context = (
            row.get("context_window_text")
            or row.get("context")
            or row.get("text")
            or row.get("input", "")
        )
        question = row.get("question") or row.get("query", "")
        answer = row.get("answer") or row.get("target", "")

        if not context or not question:
            continue

        if len(str(context)) < min_context_chars:
            continue

        examples.append({
            "id": row.get("id", str(i)),
            "context": str(context),
            "question": str(question),
            "answer": str(answer).strip() if answer else "",
            "question_type": row.get("question_type", "unknown"),
        })
        if max_examples and len(examples) >= max_examples:
            break

    return examples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oolong-synth", choices=["oolong-synth", "oolong-real"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default=None, help="For oolong-real: dnd or toy_dnd")
    parser.add_argument("--max", type=int, default=3, help="Max examples to load")
    parser.add_argument("--min-context-chars", type=int, default=0, help="Only load examples with context >= N chars (e.g. 50000 for long-context)")
    parser.add_argument("--no-streaming", action="store_true", help="Download full dataset (uses more disk)")
    args = parser.parse_args()

    examples = load_oolong(
        args.dataset, args.split, args.subset, args.max,
        min_context_chars=args.min_context_chars,
        streaming=not args.no_streaming,
    )
    print(f"Loaded {len(examples)} examples from {args.dataset}")
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        print(f"Context: {len(ex['context']):,} chars")
        print(f"Question: {ex['question'][:100]}...")
        print(f"Answer: {ex['answer'][:80]}...")
