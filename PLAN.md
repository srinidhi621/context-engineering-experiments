# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 10-12 Weeks (Realistic)  
**Start Date:** October 30, 2025  
**Budget:** $0 (Free Tier - gemini-2.0-flash-exp)  
**Team Size:** 1  

**Last Updated:** October 31, 2025  
**Status:** ✅ Infrastructure Complete - Ready to Build Experiments

---

## 📊 Current Status

**Infrastructure:** ✅ **Production-Ready**
- Python 3.13.3 environment configured
- Model: `gemini-2.0-flash-exp` (15 RPM, 1M TPM, 1500 RPD, $0.00 on free tier)
- Unified monitoring: `src/utils/monitor.py` with persistent state
- Budget: $174 enforced automatically
- All integration tests passing

**Experiments:** ❌ Not Started
- No corpus collected
- No context assemblers implemented
- No questions generated
- No experiments run

**Time Spent:** ~25 hours on infrastructure  
**Remaining:** ~10-12 weeks for experiments and analysis (realistic estimate)

**Repository:** https://github.com/srinidhi621/context-engineering-experiments

---

## 🎯 Execution Strategy: Iterative & Practical

**Philosophy:** Validate small, then scale. Build incrementally with continuous validation.

**Approach:**
```
Pilot (60 calls) → Validate → Build Core (180 calls) → Validate → Full Scale (4,200 calls)
```

**Revised Scope (Focused & Achievable):**
1. **Pilot Phase** - 10 questions, 2 strategies, validate entire pipeline
2. **Experiment 1** - Needle in Haystacks (establishes baseline, tests all 4 strategies)
3. **Experiment 2** - Context Pollution (tests robustness)
4. **Experiment 5** - Cost-Latency Frontier (analysis of Exp 1-2)

**Experiments DROPPED (Too Ambitious):**
- ❌ Experiment 3 - Multi-Turn Memory (complex, limited generalizability)
- ❌ Experiment 4 - Precision Retrieval (requires extensive PDF parsing)

**Total Estimated API Calls:** 4,380 (vs original 8,400)  
**Total Estimated Cost:** $0 (free tier)  
**Total Estimated Time:** 10-12 weeks (realistic)

---

## 🧪 PILOT PHASE: Validate Pipeline (Week 1)

**Duration:** 3-5 days  
**Goal:** End-to-end validation before scaling up  
**API Calls:** 180 (10 questions × 2 strategies × 3 fill levels × 3 reps)  
**Est. Cost:** $0 (free tier)

### Scope
- **Corpus:** 50k tokens AWS Lambda docs only
- **Strategies:** Naïve 1M + Basic RAG 128k (test both extremes)
- **Fill Levels:** [0.3, 0.5, 0.7] (representative sample)
- **Questions:** 10 manually crafted with ground truth

### Deliverables
- [ ] Minimal corpus collected and validated
- [ ] Naive + RAG assemblers working
- [ ] Padding system working (RAG padded to match fill %)
- [ ] 10 questions + ground truth answers
- [ ] All 180 API calls completed
- [ ] Metrics computed (correctness, cost, latency)
- [ ] Checkpoint/resume system tested
- [ ] **Go/No-Go Decision:** If pilot fails, fix before scaling

### Success Criteria
- Pipeline runs end-to-end without crashes
- Results are reasonable (not random garbage)
- Metrics are computable and interpretable
- Token counting is accurate (±5%)
- Checkpointing works (can resume from failure)

---

## 🧪 EXPERIMENT 1: Needle in Multiple Haystacks (Weeks 2-4)

**Duration:** 2-3 weeks  
**Goal:** Establish baseline and test all 4 context strategies across fill levels  
**Hypothesis:** Engineered 1M > Naïve 1M by ≥15% at high fill %  
**Domain:** API Documentation (AWS, GCP, Azure)  
**API Calls:** 3,000 (50 questions × 4 strategies × 5 fill levels × 3 reps)  
**Est. Cost:** $0 (free tier)

**CRITICAL FIX:** RAG strategies will be **padded to match fill %** of naive strategies. This ensures we're comparing context engineering quality, not fill % effects.

Example at 70% fill (700k tokens):
- Naïve 1M: 700k tokens (real docs + padding)
- RAG 128k: Retrieve ~90k relevant chunks + pad to 700k with irrelevant content
- This isolates RAG quality from attention dilution effects

### Phase 1: Data Collection (~2 days)

**Corpus Needed:**
- [ ] API documentation (500k-700k tokens)
  - AWS: Lambda, API Gateway, DynamoDB, S3 (~250k)
  - GCP: Cloud Functions, Storage, Firestore (~200k)
  - Azure: Functions, Blob Storage, Cosmos DB (~200k)
  - Sources: Official docs, web scraping
  - Save to: `data/raw/api_docs/{provider}/{service}/`

- [ ] Padding corpus (2M+ tokens, reusable across experiments)
  - Wikipedia articles: History, Geography, Literature, Arts, Sports
  - Filter out tech topics
  - Chunk into ~2k segments
  - Save to: `data/raw/padding_corpus/`

**Utilities:**
- [ ] Implement `src/corpus/loaders.py` (load, count tokens, metadata)
- [ ] Create `data/corpus_manifest.json` (catalog all documents)

### Phase 2: Implementation (~3 days)

**Context Assemblers to Build:**

- [ ] **Naïve** (`src/context_engineering/naive.py`)
  - Sequential concatenation + padding
  - Token-aware truncation
  - No structure

- [ ] **Structured** (`src/context_engineering/structured.py`)
  - XML/JSON structure with metadata
  - Table of contents
  - Navigation instructions
  - Hierarchical organization

- [ ] **Basic RAG** (`src/context_engineering/rag.py`)
  - Chunking (512 tokens, 50 overlap)
  - Embeddings (text-embedding-004)
  - Vector store (FAISS/ChromaDB)
  - Top-k retrieval

- [ ] **Advanced RAG** (`src/context_engineering/advanced_rag.py`)
  - Hybrid search (dense + BM25)
  - Reciprocal Rank Fusion
  - Optional reranking

**Dependencies:**
```bash
pip install faiss-cpu rank-bm25 tqdm
# Already in requirements.txt
```

**Tests:**
- [ ] Unit tests for each assembler
- [ ] Integration test with sample corpus
- [ ] Verify token budgets respected

### Phase 3: Question Generation (~2 days)

**Create 50 Questions:**
- [ ] 20 simple lookups (e.g., "What is default Lambda timeout?")
- [ ] 20 synthesis (e.g., "Compare AWS/GCP/Azure authentication")
- [ ] 10 complex (e.g., resolve contradictions in docs)

**For Each Question:**
- Ground truth answer (100-200 words)
- Required source documents
- Evaluation criteria
- Difficulty level
- Answer type (factual/comparative/analytical)

**Save to:** `data/questions/exp1_questions.json`

**Format:**
```json
{
  "experiment": "exp1_needle",
  "questions": [
    {
      "id": "exp1_q001",
      "question": "...",
      "ground_truth": "...",
      "difficulty": "simple_lookup | synthesis | complex",
      "required_docs": ["doc1.txt"],
      "evaluation_criteria": "..."
    }
  ]
}
```

### Phase 4: Execution (~2 days)

**Configuration:**
- Fill levels: [0.1, 0.3, 0.5, 0.7, 0.9]
- Target tokens: 1M for full-context, 128k for RAG
- Repetitions: 3 per configuration

**Runner Script:** `scripts/run_experiment_1.py`

- [ ] Implement experiment runner
  - Load questions and corpus
  - For each question × strategy × fill level × rep:
    - Assemble context
    - Query model
    - Log response
  - Progress tracking
  - Error handling and retry
  - Checkpoint every 100 calls

- [ ] Run all 3,000 API calls (~8 hours with rate limits)

**Save Results:** `results/raw/exp1_results.jsonl` (one JSON per line)

**Result Format:**
```json
{
  "experiment": "exp1",
  "question_id": "exp1_q001",
  "strategy": "naive_1m",
  "fill_pct": 0.7,
  "repetition": 1,
  "response": "...",
  "tokens_input": 700000,
  "tokens_output": 150,
  "latency": 2.5,
  "cost": 0.015,
  "timestamp": "2025-11-02T10:30:00"
}
```

### Phase 5: Analysis (~1 day)

**Metrics to Compute:**
- [ ] Correctness (LLM-as-judge using GPT-4 or Claude)
- [ ] Citation accuracy (claims grounded in context?)
- [ ] Cost per query
- [ ] Latency statistics

**Analysis:**
- [ ] Compare strategies at each fill level
- [ ] Plot degradation curves (fill % vs correctness)
- [ ] Statistical tests (paired t-test for H1)
- [ ] Effect sizes (Cohen's d)

**Visualizations:**
- [ ] Degradation curves (one per strategy)
- [ ] Bar chart comparison at 90% fill
- [ ] Cost vs quality scatter

**Save:**
- `results/metrics/exp1_metrics.csv`
- `results/analysis/exp1_analysis.json`
- `results/visualizations/exp1_*.png`

**Key Question:** Does Engineered 1M beat Naïve 1M by ≥15%?

---

## 🧪 EXPERIMENT 2: Context Pollution (Weeks 5-6)

**Duration:** 1-2 weeks  
**Goal:** Test robustness to irrelevant information  
**Domain:** Financial Reports (SEC filings)  
**API Calls:** 1,200 (20 questions × 4 strategies × 5 pollution levels × 3 reps)  
**Est. Cost:** $0 (free tier)

**Note:** RAG strategies also padded to match pollution levels for fair comparison.

### Phase 1: Data Collection (~2 days)

- [ ] Base corpus (50k tokens): 5 Q4 reports (Apple, Google, Microsoft, Amazon, Meta)
- [ ] Pollution corpus (1M tokens): 20-30 unrelated financial reports
- [ ] Source: SEC EDGAR (https://www.sec.gov/edgar)
- [ ] Parse sections, extract financials, label metadata
- [ ] Save to: `data/raw/financial_reports/{base|pollution}/`

### Phase 2: Implementation (~1 day)

- [ ] Pollution injection utility
- [ ] Modify assemblers to handle base + pollution mixing
- [ ] RAG should retrieve only relevant chunks

### Phase 3: Question Generation (~1 day)

- [ ] 20 questions answerable ONLY from base corpus
- [ ] Ensure answers NOT in pollution docs
- [ ] Save to: `data/questions/exp2_questions.json`

### Phase 4: Execution (~1 day)

- [ ] Pollution levels: [50k, 200k, 500k, 700k, 950k]
- [ ] Run 1,200 API calls
- [ ] Script: `scripts/run_experiment_2.py`

### Phase 5: Analysis (~1 day)

- [ ] Accuracy vs pollution level
- [ ] Hallucination rate (false positives from pollution)
- [ ] Degradation curves per strategy
- [ ] Which strategy most robust?

---

## ❌ EXPERIMENTS 3 & 4: DROPPED

**Experiment 3 (Multi-Turn Memory)** and **Experiment 4 (Precision Retrieval)** have been removed from scope to keep the project achievable:

**Why Dropped:**
- **Exp 3:** Requires complex stateful conversation management, limited generalizability
- **Exp 4:** Requires extensive PDF parsing (100+ papers), time-intensive data prep

**Impact:** Reduces API calls from 8,400 → 4,200 and timeline from 6 weeks → 10-12 weeks (more realistic)

**Future Work:** These experiments can be added later if initial results warrant expansion.

---

## 🧪 EXPERIMENT 5: Cost-Latency Frontier (Weeks 7-8)

**Duration:** 3-5 days  
**Goal:** Find optimal strategy for different constraints  
**Data:** Analysis of Experiments 1-2 (no new API calls)  
**Est. Cost:** $0

### Tasks

- [ ] Aggregate all metrics from Exp 1-2
- [ ] Normalize quality, cost, latency to [0, 1]
- [ ] Compute efficiency score: quality / (cost × latency)
- [ ] Find Pareto frontier (non-dominated points)
- [ ] Generate 3D visualization (quality × cost × latency)
- [ ] Generate 2D projections
- [ ] Rank strategies by efficiency
- [ ] Create decision framework (when to use which strategy)

---

## 📊 Final Analysis & Reporting (Weeks 9-12)

**Duration:** 3-4 weeks  
**Goal:** Complete statistical analysis and write final report

### Statistical Analysis (~3 days)

- [ ] Test H1: Engineered 1M > Naïve 1M by ≥15% (paired t-test)
- [ ] Test H2: 128k RAG ≈ Naïve 1M quality (equivalent test)
- [ ] Compute effect sizes (Cohen's d)
- [ ] ANOVA across all strategies
- [ ] Regression analysis (quality ~ strategy + fill_pct + interaction)
- [ ] Bootstrap confidence intervals

### Visualizations (~2 days)

- [ ] Master degradation curves (all experiments)
- [ ] Strategy comparison bar charts
- [ ] Cost-quality scatter plots
- [ ] Pollution robustness curves
- [ ] 3D Pareto frontier (interactive Plotly)
- [ ] Position bias heatmaps

### Final Report (~5 days)

**Structure:**
1. Executive Summary (1 page)
2. Introduction & Hypotheses (2 pages)
3. Methodology (3 pages)
4. Results (10 pages, one section per experiment)
5. Discussion & Limitations (3 pages)
6. Recommendations & Decision Framework (2 pages)
7. Conclusion (1 page)

**Save to:** `FINAL_REPORT.md`

### Code Documentation (~2 days)

- [ ] Add docstrings to all functions
- [ ] Update README with results
- [ ] Add troubleshooting guide
- [ ] Clean up code (linter, type hints)
- [ ] Achieve >80% test coverage
- [ ] Tag release: `v1.0.0`

---

## 📋 Success Criteria

**Minimum:**
- ✅ Both hypotheses tested with statistical significance
- ✅ Clear conclusion (accept/reject)
- ✅ Final report written
- ✅ Code runs without errors

**Good:**
- ✅ All above + p < 0.05
- ✅ Effect sizes > 0.3
- ✅ Publication-quality visualizations
- ✅ Code documented and tested
- ✅ Shared publicly (GitHub + blog)

**Exceptional:**
- ✅ Novel insights beyond hypotheses
- ✅ Practical adoption in production
- ✅ Community recognition (100+ stars)

---

## 🚀 Quick Start Commands

```bash
# Activate environment
source scripts/activate.sh

# Test API integration
python scripts/test_api_integration.py

# Monitor costs
python scripts/monitor_costs.py
python scripts/generate_cost_report.py

# Run experiments (when ready)
python scripts/run_pilot.py              # Start with this!
python scripts/run_experiment_1.py       # After pilot succeeds
python scripts/run_experiment_2.py       # After Exp 1 succeeds

# Generate analysis
python scripts/analyze_results.py
python scripts/generate_visualizations.py
```

---

## 📝 Notes

- **Budget:** $174 enforced automatically by monitor, hard stop if exceeded
- **Rate Limits:** 15 RPM, 1M TPM, 1500 RPD (free tier)
- **Persistence:** All API calls tracked in `results/.monitor_state.json`
- **Checkpointing:** Save progress every 100 calls
- **Reproducibility:** Set random seeds, pin library versions

---

**Last Updated:** October 31, 2025  
**Version:** 2.0 (Experiment-by-Experiment)  
**Status:** Infrastructure Complete - Ready for Exp 1
