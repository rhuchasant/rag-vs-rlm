"""Run RLM on a document."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable RLM progress logging
_rlm_log = logging.getLogger("rlm")
_rlm_log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(message)s"))
_rlm_log.addHandler(_h)

from src.rlm import RLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input (txt or json for multi-tab)")
    parser.add_argument("--query", "-q", default="Extract data file layouts for the entire document/sheet", help="Query")
    parser.add_argument("--model", default="gpt-4o", help="Root model")
    parser.add_argument("--recursive-model", default="gpt-4o-mini", help="Sub-call model")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Load context - JSON for multi-tab (Excel-like), TXT for single doc
    if input_path.suffix == ".json":
        context = json.loads(input_path.read_text(encoding="utf-8"))
        total_chars = sum(len(v) for v in context.values() if isinstance(v, str))
        print(f"Loaded {len(context)} tabs, {total_chars:,} chars total")
    else:
        context = input_path.read_text(encoding="utf-8")
        print(f"Document length: {len(context):,} chars")

    print(f"Query: {args.query}")
    print("-" * 50)

    rlm = RLMClient(model=args.model, recursive_model=args.recursive_model)
    response = rlm.completion(context, args.query)

    print("RLM Response:")
    print(response)


if __name__ == "__main__":
    main()
