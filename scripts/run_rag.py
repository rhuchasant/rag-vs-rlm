"""Run RAG pipeline on a document."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input document (txt)")
    parser.add_argument("--query", "-q", default="Extract all data file layouts from this document", help="Query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    print(f"Document length: {len(text):,} chars (~{len(text)//4:,} tokens)")
    print(f"Query: {args.query}")
    print("-" * 50)

    rag = RAGPipeline(top_k=args.top_k, chunk_size=args.chunk_size)
    response = rag.query(text, args.query)

    print("RAG Response:")
    print(response)


if __name__ == "__main__":
    main()
