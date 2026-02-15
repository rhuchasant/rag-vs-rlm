# Project Plan & To-Do

## Research Focus

**Claim:** For long documents and "extract everything" tasks, RLM outperforms RAG because RAG hits context limits.

**Evidence so far:** 19-tab Excel benchmark → RAG context error; RLM succeeded.

---

## Plan

### Phase 1: Benchmarks
- [ ] **1.1** Scale Excel benchmark to 30+ tabs; task: "List all tab names in order"
- [ ] **1.2** Synthetic "extract everything" docs (long, broad extraction)
- [ ] **1.3** Define metrics: coverage, recall, order accuracy; document RAG failures

### Phase 2: Experiments
- [ ] **2.1** Run RAG on scaled Excel → record context errors, coverage
- [ ] **2.2** Run RLM on same benchmark → coverage, success rate
- [ ] **2.3** Both on synthetic "extract everything" (50k+ chars)
- [ ] **2.4** Robustness: seeds, format variants, confidence intervals

### Phase 3: Analysis & Write-up
- [x] **3.1** Results table: RAG vs RLM by benchmark → `RESULTS_TABLE.md`
- [x] **3.2** Discussion: when RAG fails vs when RLM succeeds → in `RESULTS_TABLE.md`
- [ ] **3.3** Optional: Oolong as secondary (RAG competitive on targeted reasoning)

### Phase 4 (Medium) — Done
- [x] **4.1** Hybrid: RLM for full extraction + RAG for targeted retrieval
- [x] **4.2** Compare hybrid vs RLM-only vs RAG-only → Hybrid 3/3, RAG 2/3, RLM 2/3

---

## Quick Commands

```powershell
# Excel benchmark (30 tabs by default, generate + run)
venv\Scripts\python scripts\run_comparison.py --generate --tabs 30 --rag-only
venv\Scripts\python scripts\run_comparison.py --generate --tabs 30 --rlm-only

# Oolong (RAG + custom RLM)
venv\Scripts\python scripts\run_oolong_comparison.py --dataset oolong-synth --max 5

# Synthetic structured extraction
venv\Scripts\python scripts\run_synthetic_structured_extraction.py --docs 5 --target-chars 50000

# Robustness suite
venv\Scripts\python scripts\run_robustness_suite.py --excel-seeds 1,2,3 --synthetic-seeds 1,2

# Phase 4: Hybrid comparison (mixed tasks)
venv\Scripts\python scripts\run_hybrid_comparison.py --generate --tabs 30

# Legal benchmark (real CFR)
venv\Scripts\python scripts\run_legal_comparison.py --download
venv\Scripts\python scripts\run_legal_hybrid_comparison.py

# Recall analysis (empirical P(full coverage) vs theoretical r^N)
venv\Scripts\python scripts\run_recall_analysis.py --benchmark legal --n-values 30,100,200,500

# Iterative RAG vs Single-Pass RAG vs RLM
venv\Scripts\python scripts\run_iterative_rag_comparison.py --benchmark legal
```

---

## Results Layout

- `results/*.json` – Latest runs (comparison, oolong, synthetic, robustness)
- `results/archive/` – Old runs (archived during cleanup)
- `benchmarks/` – Generated data (data_layout_benchmark.json, synthetic_long_docs*.json)
