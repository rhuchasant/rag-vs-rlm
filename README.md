# RLM vs RAG: Long-Document Exhaustive Extraction

Research comparing **Recursive Language Models (RLM)** with **Retrieval-Augmented Generation (RAG)** on long-document "extract everything" tasks, where the goal is complete coverage rather than localized QA.

## Key Finding

**Bounded retrieval is structurally mismatched to exhaustive extraction.** RAG (single-pass and multi-pass) shows limited coverage under context limits; iterative traversal with external control flow achieves full enumeration.

| Benchmark | Single-Pass RAG | Iterative RAG | RLM (Traversal) |
|-----------|-----------------|---------------|-----------------|
| Excel 30 tabs (~750k chars) | Context overflow | — | 100% |
| Legal 587 sections (17 CFR Part 240) | ~2–3% | ~32% | 100% |
| Synthetic structured extraction | 0% exact | — | 100% |

A **hybrid router** (broad extraction → traversal, targeted QA → RAG) outperforms either approach alone on mixed task sets.

## Hypothesis

For tasks requiring *complete document coverage* (e.g., "list all section IDs," "enumerate all tabs"), RLM-style iterative traversal outperforms RAG because:

- RAG retrieval does not guarantee global coverage under fixed context limits
- Iterative traversal processes chunks under external control flow and aggregates outside the context window
- Multi-pass RAG improves recall but still falls far short of deterministic enumeration

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
