# RLM vs RAG: Long-Document Exhaustive Extraction

Code and benchmarks for comparing **Retrieval-Augmented Generation (RAG)** with **RLM-style iterative traversal** on long-document tasks that require complete coverage (e.g., "list all section IDs," "enumerate all tabs") rather than localized QA.

## Research Question

How do bounded-retrieval systems behave on tasks that require full document coverage? This repo implements benchmarks and evaluation to study RAG (single-pass and multi-pass) versus iterative traversal with external control flow.

## Preliminary Results

Initial experiments across several benchmarks suggest traversal-based approaches achieve higher coverage on exhaustive extraction tasks. See `RESULTS_TABLE.md` for details.

| Benchmark | Single-Pass RAG | Iterative RAG | Traversal |
|-----------|-----------------|---------------|-----------|
| Excel 30 tabs (~750k chars) | Context overflow | — | 100% |
| Legal 587 sections (17 CFR Part 240) | ~2–3% | ~32% | 100% |
| Synthetic structured extraction | 0% exact | — | 100% |

A hybrid router (broad extraction → traversal, targeted QA → RAG) outperforms either approach alone on mixed task sets.

## Project Structure

```
rlm-vs-rag-research/
├── src/
│   ├── rag/           # RAG pipeline (chunking, embedding, retrieval)
│   ├── rlm/           # Custom RLM (REPL + code execution)
│   ├── hybrid/        # Hybrid router (traversal vs RAG)
│   └── evaluation/    # Metrics (coverage, etc.)
├── benchmarks/        # Datasets and generators
├── scripts/           # Run scripts
├── experiments/      # Wrapper experiments (alexzhang13/rlm)
├── results/          # Output JSON
└── config/            # Config files
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # Add OPENAI_API_KEY
```

**Generate benchmark data** (not in repo):

```powershell
python benchmarks/download_legal_benchmark.py
python benchmarks/generate_data_layout_benchmark.py --tabs 30
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_comparison.py` | RLM vs RAG on Excel data layout (multi-tab extraction) |
| `run_legal_comparison.py` | RLM vs RAG on 17 CFR Part 240 (587 sections) |
| `run_iterative_rag_comparison.py` | Single-pass vs multi-pass RAG vs RLM |
| `run_hybrid_comparison.py` | Hybrid router on mixed tasks |
| `run_synthetic_structured_extraction.py` | Extract section ID/title/topic (synthetic) |
| `run_recall_analysis.py` | Empirical recall vs theoretical r^N |
| `run_oolong_comparison.py` | RLM vs RAG on Oolong long-context benchmark |

## Benchmarks

1. **Excel (30 tabs, ~750k chars)** – List all tab names; RAG hits context overflow.
2. **Legal (17 CFR Part 240, 587 sections)** – List all section numbers; real GovInfo data.
3. **Synthetic docs** – Long docs with sections; "list all" and structured extraction tasks.
4. **Oolong** – Long-context reasoning (targeted QA; RAG competitive here).

## References

- [Recursive Language Models (MIT CSAIL)](https://arxiv.org/abs/2512.24601)
- [RLM Official Code](https://github.com/alexzhang13/rlm)
- [OOLONG Benchmark](https://huggingface.co/oolongbench/datasets)
