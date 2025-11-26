# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 10-12 Weeks (Realistic)  
**Start Date:** October 30, 2025  
**Budget:** $0 (Free Tier - gemini-2.0-flash-exp)  
**Team Size:** 1  

**Last Updated:** November 26, 2025  
**Status:** ✅ Readiness Sprint Complete – 🚀 Experiment 1 Ready to Launch

---

## 📊 Current Status

**Infrastructure:** ✅ **Complete & Production-Ready**
- Python 3.13.3 environment configured (required for google-generativeai >= 0.3.0)
- Model: `gemini-2.0-flash-exp` (15 RPM, 1M TPM, 1500 RPD, $0.00 on free tier)
- Unified monitoring: `src/utils/monitor.py` with persistent state
- Budget: $174 enforced automatically
- All integration tests passing
- ✅ **Corpus loaders implemented:** Hugging Face Hub + Gutenberg (460+ lines)
- ✅ **Tokenizer utilities complete:** counting, chunking, truncation

**Pilot Phase:** ✅ **COMPLETE**
- ✅ Phase 1A: Infrastructure setup
- ✅ Phase 1B: Data collection
- ✅ Phase 1C: Context assemblers
- ✅ Phase 1D: Minimal runner execution
- ✅ Phase 1E: Go/No-Go Decision (Decision: GO)

**Experiments:**
- Experiment 1: Needle in Multiple Haystacks – 🚀 **Ready to Launch**
- Experiment 2: Context Pollution – **Queued** (begin only after Exp 1 results)
- Experiment 5: Cost-Latency Frontier – **Queued** (analysis after Exp 1–2)

**Time Spent:** ~32 hours on infrastructure, pilot & readiness  
**Remaining:** ~9-11 weeks for experiments + analysis

**Repository:** https://github.com/srinidhi621/context-engineering-experiments

## 📝 Execution Log

| Date | Time | Activity | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 2025-11-26 | 16:06 | **Setup Run** | ✅ Success | Cached chunks & embeddings (0 API calls) |
| 2025-11-26 | 16:56 | **Full Dry Run** | ✅ Success | Verified 3,000 run combinations & status logic |
| 2025-11-26 | 17:09 | **Mini Live Run** | ✅ Success | 2 real API calls + analysis pipeline verification |

### ✅ Readiness Sprint (Completed Nov 26, 2025)

All blocking issues for Experiment 1 have been resolved:

1.  ✅ **Rate Limits Fixed:** Separated `text-embedding-004` and `gemini-2.0-flash-exp` monitors.
2.  ✅ **Persistence Added:** RAG chunks, embeddings, and FAISS/BM25 indexes are now cached to `results/cache/` to prevent re-computation.
3.  ✅ **Data Collected:** 700k+ tokens of model cards collected to `data/raw/exp1/`.
4.  ✅ **Analysis Stack Built:** Implemented `metrics.py`, `judges.py` (LLM-as-a-Judge), and `analyze_results.py`.
5.  ✅ **End-to-End Verification:**
    *   Synthetic run proved the analysis pipeline works.
    *   Live smoke test (limit=2) proved the API integration works.

---

## 🛠️ Immediate Readiness Sprint (Week 0–1) - ✅ COMPLETE

Goal: unblock Experiment 1 by fixing infrastructure gaps. All items below are prerequisites for any large-scale run.

1. **Separate monitoring for embeddings vs. generations**
   - ✅ Update `GeminiClient` so `text-embedding-004` calls use their own monitor (or bypass RPD accounting) and create persistent embedding caches to avoid reusing quota.
   - ✅ Add regression tests confirming mixed workloads respect limits.
   - *Exit criteria:* indexing the Exp 1 corpus completes without tripping the generation RPD hard stop.

2. **Persist & resume embeddings + padding**
   - ✅ Cache FAISS/BM25 indexes and chunk vectors under `results/cache/` and load them when present.
   - ✅ Switch `PaddingGenerator` to reuse `data/raw/padding/gutenberg_corpus.json` rather than re-downloading books.
   - *Exit criteria:* restarting Exp 1 skips embedding recomputation and performs no network calls for padding.

3. **Align configuration and experiment runners**
   - ✅ Set `config.model_name = "gemini-2.0-flash-exp"`, update docs, and extend `scripts/run_experiment.py` with explicit handlers for pilot/exp1/exp2.
   - ✅ Harden `scripts/run_experiment_1.py` with `PerMinuteTokenThrottle`, persistent status files, and automatic sleep/resume when RPM/RPD/TPM limits trigger.
   - *Exit criteria:* `python scripts/run_experiment.py --experiment exp1 --dry-run` validates orchestration; live runs checkpoint and resume after quota resets.

4. **Implement scoring & analysis stack**
   - ✅ Finish `src/experiments/base_experiment.py` + `exp1_needle.py`, implement evaluation utilities (metrics, judges, human eval hooks), and ship `scripts/analyze_results.py` + `scripts/generate_report.py`.
   - ✅ Add pytest coverage for evaluation logic.
   - *Exit criteria:* given sample results, `scripts/analyze_results.py` outputs metrics/visualization data; tests pass.

5. **Packaging & validation**
   - ✅ Expand `setup.py` to include all experiment dependencies and flesh out placeholder pytest modules plus lint/format tooling (e.g., `ruff`).
   - ✅ Document a pre-flight checklist (pytest + lint) to run before API traffic.
   - *Exit criteria:* clean `pip install -e .` on a fresh venv; `python -m pytest` succeeds locally/CI.

Completion of this sprint is the gate to start Experiment 1.

---

## 🎯 Execution Strategy: Iterative & Practical

**Philosophy:** Validate small, fix infrastructure, then run each experiment end-to-end (collect → run → analyze → document) before advancing.

**Approach:**
```
Pilot ✅ → Readiness Sprint → Exp1 (run+analyze) → Retrospective → Exp2 (run+analyze) → Frontier Analysis
```

**Revised Scope (Focused & Iterative):**
1. **Pilot Phase** – already complete; serves as regression harness.
2. **Experiment 1** – Needle in Haystacks (full loop: execute, score, analyze, publish findings).
3. **Experiment 2** – Context Pollution (repeat loop, incorporate Exp 1 learnings).
4. **Experiment 5** – Cost-Latency Frontier (analysis-only using Exp 1–2 outputs).

**Experiments DROPPED (Too Ambitious):**
- ❌ Experiment 3 - Multi-Turn Memory (complex, limited generalizability)
- ❌ Experiment 4 - Precision Retrieval (requires extensive PDF parsing)

**Total Estimated API Calls:** 4,380 (vs original 8,400)  
**Total Estimated Cost:** $0 (free tier)  
**Total Estimated Time:** 10-12 weeks (realistic)

---

## ✅🧪 PILOT PHASE: COMPLETE

The pilot phase was completed successfully, validating the entire experimental pipeline. Key learnings and a full summary are documented in [results/pilot_summary.md](./results/pilot_summary.md).

---
## 🧪 EXPERIMENT 1: Needle in Multiple Haystacks (Weeks 1–3 after readiness)

**Status:** ⏸️ Blocked – waiting on readiness sprint

**Duration:** 2–3 weeks once unblocked  
**Goal:** Establish baseline and test all 4 context strategies across fill levels, then immediately analyze/document findings before advancing.  
**Hypothesis:** Engineered 1M > Naïve 1M by ≥15% at high fill %  
**Domain:** GitHub Repository Documentation  
**API Calls:** 3,000 (50 questions × 4 strategies × 5 fill levels × 3 reps)  
**Est. Cost:** $0 (free tier)

**CRITICAL:** RAG strategies will be **padded to match fill %** of naive strategies (implemented in pilot phase).

### Phase Breakdown

1. **Data readiness (complete):** Corpora and question set already collected (see acceptance checklist).
2. **Readiness sprint (Week 0–1):** Finish the blocking tasks described above (rate-limit separation, caching, runner resiliency, scoring stack, packaging/tests).
3. **Experiment run (Week 1–2):** Build/reuse embedding cache, run 3,000 calls with checkpointing and quota-aware throttling, append to `results/raw/exp1_results.jsonl`.
4. **Analysis & publication (Week 2–3):** Score outputs, generate metrics/visualizations, write `results/analysis/exp1_summary.md`, and update README/PLAN before starting Exp 2.

### Acceptance Checklist

- [x] **GitHub corpus available:** `data/raw/exp1/github_corpus.json` (>600k tokens) exists.
- [x] **Padding corpus available:** `data/raw/padding/gutenberg_corpus.json` (>1.5M tokens) exists.
- [x] **Questions validated:** `python scripts/validate_question_set.py data/questions/exp1_questions.json --require-experiment exp1` exits 0 with ≥50 entries.
- [ ] **Embedding cache built once:** first run emits `results/cache/exp1_index.*`; reruns reuse cache without consuming embedding RPD quota.
- [ ] **Runner completion:** `python scripts/run_experiment.py --experiment exp1` yields 3,000 lines in `results/raw/exp1_results.jsonl` and records checkpoints (`results/exp1_status.json` or similar).
- [ ] **Automated scoring:** `python scripts/analyze_results.py --input results/raw/exp1_results.jsonl --output-dir results/analysis/exp1` produces metrics CSV + markdown summary; visualizations generated via `scripts/generate_visualizations.py`.
- [ ] **Documentation:** README + PLAN updated with Exp 1 findings, rate-limit notes, and learnings before Experiment 2 begins.

---

## 🧪 EXPERIMENT 2: Context Pollution (Weeks 3–6)

**Status:** 🚫 Do not start until Exp 1 analysis + documentation complete

**Duration:** 1-2 weeks  
**Goal:** Test robustness to irrelevant information  
**Domain:** GitHub Documentation + Gutenberg Books  
**API Calls:** 1,200 (20 questions × 4 strategies × 5 pollution levels × 3 reps)  
**Est. Cost:** $0 (free tier)

**Note:** RAG strategies also padded to match pollution levels for fair comparison.

### Week 5: Data Collection & Question Generation (5 days)

**Day 1: Collect Base Corpus (50k tokens of relevant GitHub docs)**

Use 3-5 repositories not used in Experiment 1:
- Example: FastAPI, Pydantic, SQLAlchemy, Celery, Requests
- Fetch recent documentation (after 2024-08-01)
- Target: 50k tokens total
- Save to: `data/raw/exp2/base_corpus.json`

**Day 2: Prepare Pollution Corpus (reuse Gutenberg)**

Use the Gutenberg corpus collected in Experiment 1:
- Already have 2M+ tokens from `data/raw/padding/gutenberg_corpus.json`
- Classic literature is clearly irrelevant to technical questions
- No additional collection needed

**Day 3-5: Generate 20 Questions**

Create questions answerable ONLY from base corpus:
- 15 simple lookups (single fact)
- 5 synthesis (2-3 docs from base)
- Verify answers are NOT in pollution corpus
- Example: "What is the default timeout for FastAPI requests?" (should NOT appear in Dickens)

Save to: `data/questions/exp2_questions.json`

**Acceptance Criteria:**
- [ ] **Gate check:** `results/analysis/exp1_summary.md` exists and README/PLAN include Exp 1 outcomes.
- [ ] **Base Corpus:** `python scripts/collect_exp2_corpus.py` runs successfully and `ls data/raw/exp2/base_corpus.json` lists the file.
- [ ] **Padding Corpus:** `ls data/raw/padding/gutenberg_corpus.json` confirms the padding corpus from Exp1 exists.
- [ ] **Question Set:** `python scripts/validate_question_set.py data/questions/exp2_questions.json --require-experiment exp2` exits with code 0.
- [ ] **Question Count:** `python -c "import json; f = open('data/questions/exp2_questions.json'); data = json.load(f); assert len(data['questions']) >= 20, f'Expected 20+ questions, found {len(data[\'questions\'])}'"` exits with code 0.
- [ ] **Manual Check:** Reviewer confirms that answers to the questions are not present in the Gutenberg padding corpus.

### Week 6: Execution & Analysis (5 days)

**Day 1-2: Implement Pollution Injection**

File: `src/corpus/pollution.py` (NEW)
```python
class PollutionInjector:
    """Inject irrelevant content at varying levels"""
    
    def inject_pollution(self, 
                        base_content: str,
                        pollution_content: str,
                        pollution_tokens: int) -> str:
        """
        Mix base content with pollution.
        
        Args:
            base_content: Relevant documents
            pollution_content: Irrelevant text (Gutenberg)
            pollution_tokens: Amount of pollution to add
        
        Returns:
            Mixed content (base + pollution)
        """
        # Truncate pollution to target size
        pollution = truncate_to_tokens(pollution_content, pollution_tokens)
        
        # Strategy: Append pollution after base
        # (Could also interleave, but simpler for now)
        return base_content + "\n\n" + pollution
```
**Acceptance Criteria:**
- [ ] **Unit Test:** `pytest tests/test_corpus.py::test_pollution_injector` exits with code 0.

**Day 3-4: Run Experiment 2 (1,200 API calls)**

Pollution levels:
- 50k tokens pollution (50% of base)
- 200k tokens pollution (4x base)
- 500k tokens pollution (10x base)
- 700k tokens pollution (14x base)
- 950k tokens pollution (19x base)

Script: `scripts/run_experiment_2.py`

**Acceptance Criteria:**
- [ ] **Dry Run:** `python scripts/run_experiment_2.py --dry-run` exits with code 0 and logs that it would make 1,200 API calls.
- [ ] **File Creation:** After a full run, `ls results/raw/exp2_results.jsonl` successfully lists the file.
- [ ] **Result Count:** `wc -l results/raw/exp2_results.jsonl` reports 1200.

**Day 5: Analyze Results**

Metrics:
- Accuracy vs pollution level (does it degrade?)
- Hallucination rate (false positives from pollution)
- Degradation curves per strategy
- Which strategy most robust to noise?

**Acceptance Criteria:**
- [ ] **Analysis Script:** `python scripts/analyze_results.py --input results/raw/exp2_results.jsonl --output-dir results/analysis/exp2` exits with code 0.
- [ ] **Summary File:** `ls results/analysis/exp2/exp2_summary.md` successfully lists the file.
- [ ] **Visualization Script:** `python scripts/generate_visualizations.py --input results/analysis/exp2/exp2_metrics.csv --output-dir results/visualizations/exp2` exits with code 0.
- [ ] **Plot Verification:** `ls results/visualizations/exp2/*.png | wc -l` reports a count of 2 or more.

---

## ❌ EXPERIMENTS 3 & 4: DROPPED

**Experiment 3 (Multi-Turn Memory)** and **Experiment 4 (Precision Retrieval)** have been removed from scope to keep the project achievable:

**Why Dropped:**
- **Exp 3:** Requires complex stateful conversation management, limited generalizability
- **Exp 4:** Requires extensive PDF parsing (100+ papers), time-intensive data prep

**Impact:** Reduces API calls from 8,400 → 4,200 and timeline from 6 weeks → 10-12 weeks (more realistic)

**Future Work:** These experiments can be added later if initial results warrant expansion.

---

## 🧪 EXPERIMENT 5: Cost-Latency Frontier (Weeks 6-8)

**Status:** ⏳ Planned

**Duration:** 3-5 days  
**Goal:** Find optimal strategy for different constraints  
**Data:** Analysis of Experiments 1-2 (no new API calls)  
**Est. Cost:** $0

### Acceptance Criteria
- [ ] **Analysis Script:** `python scripts/analyze_frontier.py --exp1-metrics results/analysis/exp1_metrics.csv --exp2-metrics results/analysis/exp2_metrics.csv --output-dir results/analysis/exp5` exits with code 0.
- [ ] **Summary File:** `ls results/analysis/exp5/frontier_analysis.md` successfully lists the file.
- [ ] **Visualization Script:** `python scripts/generate_visualizations.py --input results/analysis/exp5/frontier_data.csv --output-dir results/visualizations/exp5` exits with code 0.
- [ ] **Plot Verification:** `ls results/visualizations/exp5/*.png` reports a count of 1 or more.

---

## 📊 Final Analysis & Reporting (Weeks 8-12)

**Status:** ⏳ Planned

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

**Acceptance Criteria:**
- [ ] **Report Generation:** `python scripts/generate_report.py --all-results results/analysis/ --output FINAL_REPORT.md` exits with code 0.
- [ ] **File Verification:** `ls FINAL_REPORT.md` successfully lists the final report file.

### Code Documentation (~2 days)

- [ ] Add docstrings to all functions
- [ ] Update README with results
- [ ] Add troubleshooting guide
- [ ] Clean up code (linter, type hints)
- [ ] Achieve >80% test coverage
- [ ] Tag release: `v1.0.0`

**Acceptance Criteria:**
- [ ] **Code Quality:** `ruff check .` and `ruff format . --check` both exit with code 0.
- [ ] **Test Coverage:** `pytest --cov=src tests/` runs successfully and the coverage report shows a total coverage of 80% or higher.

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
- **Checkpointing:** Main experiment runners must support robust checkpointing and resumption (see Exp1, Phase 4).
- **Reproducibility:** Pin library versions. All experiment runner scripts must accept a `--seed` command-line argument to control stochastic processes (e.g., padding generation) for fully reproducible results.
- **Idempotent Execution:** All runner scripts must be idempotent. They should check for existing results in the output file before execution and automatically skip any runs that are already complete. This prevents redundant API calls, saves costs, and avoids rate-limiting issues on re-runs.

---

**Last Updated:** November 3, 2025  
**Version:** 2.1 (Phase 1A Complete)  
**Status:** Phase 1A Complete - Ready for Phase 1B (Data Collection)
