# RLM vs RAG: Long-Document Extraction

Research comparing **Recursive Language Models (RLM)** with **Retrieval-Augmented Generation (RAG)** for long-document tasks, especially "extract everything" scenarios where RAG hits context limits.

## Hypothesis

For long documents and broad extraction ("extract everything"), RLM outperforms RAG because RAG is constrained by context limits. RLM uses code-based iteration to process chunks and aggregate results.

## Project Structure

```
rlm-vs-rag-research/
├── src/
│   ├── rag/           # RAG pipeline (chunking, embedding, retrieval)
│   ├── rlm/           # Custom RLM (REPL + code execution)
│   └── evaluation/    # Metrics (coverage, etc.)
├── benchmarks/        # Datasets and generators
├── scripts/           # Run scripts
├── experiments/       # Wrapper experiments (alexzhang13/rlm)
├── results/           # Output JSON (archive/ for old runs)
└── config/            # Config files
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env
```

## Scripts

| Script | Purpose |
|-------|---------|
| `run_comparison.py` | RLM vs RAG on **Excel data layout** (multi-tab extraction) |
| `run_oolong_comparison.py` | RLM vs RAG on **Oolong** (long-context reasoning) |
| `run_oolong_wrapper_comparison.py` | RAG vs **official RLM wrapper** on Oolong |
| `run_synthetic_sections_comparison.py` | List section IDs (synthetic long docs) |
| `run_synthetic_structured_extraction.py` | Extract section ID/title/topic (synthetic) |
| `run_robustness_suite.py` | Multi-seed, format variants, confidence intervals |

## Benchmarks

1. **Data layout (Excel)** – Multi-tab extraction; RAG hit context errors at 19 tabs.
2. **Oolong** – Long-context reasoning (oolong-synth, oolong-real).
3. **Synthetic docs** – Long docs with sections; "list all" and "extract structured" tasks.

## References

- [Recursive Language Models (MIT CSAIL)](https://arxiv.org/abs/2512.24601)
- [RLM Official Code](https://github.com/alexzhang13/rlm)
- [OOLONG Benchmark](https://huggingface.co/oolongbench/datasets)
