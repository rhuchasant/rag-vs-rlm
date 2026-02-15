# Results Table: RLM vs RAG on "Extract Everything" Tasks

## Summary

| Benchmark | Doc Size | Task | RAG | RLM |
|-----------|----------|------|-----|-----|
| **Excel 30 tabs** | ~700k chars | List all tab names in order | Context error | **100%** |
| **Synthetic sections** | ~50k chars × 5 docs | List all section IDs in order | 100% | 100% |
| **Synthetic structured** | ~50k chars × 5 docs | Extract section ID, title, main topic | 0% exact, 30% topic | **100% exact, 100% topic** |

---

## Excel 30-Tab Benchmark (Robustness: 3 seeds)

| Metric | RAG | RLM |
|--------|-----|-----|
| Top-5 coverage | 13.3% ± 0 (4/30 tabs) | — |
| Full retrieval (semantic) | Context error (3/3 runs) | — |
| Full retrieval (tab-level) | Context error (3/3 runs) | — |
| Coverage | — | **100% ± 0** (30/30 tabs) |

**Config:** 30 tabs, ~750k chars, seeds 42–44.

---

## Synthetic Sections (List IDs)

| Metric | RAG | RLM |
|--------|-----|-----|
| Ordered exact rate | 100% | 100% |
| Set recall | 1.00 | 1.00 |
| Prefix accuracy | 1.00 | 1.00 |

**Config:** 5 docs, 20 sections each, ~50k chars/doc.

---

## Synthetic Structured Extraction

| Metric | RAG | RLM |
|--------|-----|-----|
| Overall exact rate | 0% | **100%** |
| Completeness recall | 1.00 | 1.00 |
| Structure valid rate | 1.00 | 1.00 |
| Title accuracy (matched) | 1.00 | 1.00 |
| Topic accuracy (matched) | 0.30 | **1.00** |

**Config:** 5 docs, 20 sections each, ~50k chars/doc. Task: extract section_id, title, main_topic as JSON.

---

## Discussion

**When RAG fails:** On the Excel benchmark (~700k chars), RAG hits the model context limit (128k tokens) when retrieving all 30 tab chunks. Full retrieval fails in 100% of runs.

**When RLM succeeds:** RLM iterates over the tab dict in code, so it never needs to fit the full document in context. It achieves 100% coverage across all seeds.

**Structured extraction:** Even when RAG fits in context (~50k chars), RLM outperforms on structured extraction (exact match, topic accuracy). RLM’s code-based approach yields more consistent, accurate structured output.

---

## Phase 4: Hybrid (Mixed Tasks) — Extended

| Approach | Mean Rate | ± CI95 | Correct (avg/seed) |
|----------|-----------|--------|--------------------|
| RAG-only | 66.67% | 0% | 4/6 |
| RLM-only | 33.33% | 0% | 2/6 |
| **Hybrid** | **83.33%** | **0%** | **5/6** |

**Tasks:** 1 broad (list all tab names in order) + 5 targeted:
- Filename for Tab_5_Shipping_Info
- First column for Tab_1_Customer_Data
- Primary key for Tab_10_Supplier_Info
- Which tab uses shipping.csv
- Export file for Tab_1_Customer_Data

**Config:** 30 tabs, ~700k chars, seeds 42–44.

**Hybrid routing:** Broad → RLM, Targeted → RAG. Hybrid outperforms both single approaches by routing each task to the appropriate tool.

---

## Legal Benchmark (17 CFR Part 240)

| Metric | RAG (top_k=20) | RAG (top_k=587) | RLM |
|--------|----------------|-----------------|-----|
| Coverage (list all 587 sections) | 3.4% | 2.0% | **100%** |

**Config:** Real CFR from GovInfo, 587 sections, ~85k chars.

**Iterative RAG (multi-pass, no code):** 32.2% (10 passes) — improves over single-pass (3.4%) but RLM still wins (100%).

---

## Legal Hybrid (Mixed Tasks)

| Approach | Correct | Total | Rate |
|----------|---------|-------|------|
| RAG-only | 4 | 6 | 66.7% |
| RLM-only | 3 | 6 | 50% |
| **Hybrid** | **5** | **6** | **83.3%** |

**Tasks:** 1 broad (list all section numbers) + 5 targeted (subject of §240.0-1, §240.0-9; which section covers Definitions, filing fees, dealer exemption).

**Hybrid routing:** Broad → RLM (100%), Targeted → RAG. Hybrid combines RLM’s full extraction with RAG’s targeted retrieval.
