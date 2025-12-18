# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 10-12 Weeks (Realistic)  
**Start Date:** October 30, 2025  
**Budget:** $0 (Free Tier - gemini-2.0-flash-exp)  
**Team Size:** 1  

**Last Updated:** December 3, 2025  
**Status:** ✅ Experiment 1 run + analysis completed; preparing Experiment 2

---

## 📊 Current Status

**Infrastructure:** ✅ **Operational**
- Python 3.13.3 environment configured (required for google-generativeai >= 0.3.0).
- Model: `models/gemini-2.0-flash` (stable 1M-token context confirmed).
- Unified monitoring + budget guardrail from `src/utils/monitor.py` persists usage under `results/.monitor_state.json`.
- Corpus loaders/tokenizers are implemented and cached (`results/cache/*`).

**Pilot Phase:** ✅ **COMPLETE**
- ✅ Phase 1A–1E covering infra, data, assemblers, runner, and go/no-go decision.

**Experiment 1 (Needle in Multiple Haystacks):** ✅ **Complete**
- 3,000 run keys executed (pending list now empty); analysis saved under `results/analysis/exp1_rerun/`.
- Key finding: Structured ≈ RAG > Advanced RAG >> Naive; best accuracy at 10–30% fill, degradation at higher fills.

**Experiment 2 (Context Pollution):** 🛠️ **Ready to implement** (requires code + data prep).

**Experiment 5 (Cost-Latency Frontier):** ⏸ Dependent on Exp 2 metrics.

**Time Spent:** ~32 hours on infrastructure, pilot & readiness  
**Remaining:** ~9-11 weeks for experiments + analysis

**Repository:** https://github.com/srinidhi621/context-engineering-experiments

## 📝 Execution Log

| Date | Time | Activity | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 2025-11-26 | 16:06 | **Setup Run** | ✅ Success | Cached chunks & embeddings (0 API calls) |
| 2025-11-26 | 16:56 | **Full Dry Run** | ✅ Success | Verified 3,000 run combinations & status logic |
| 2025-11-26 | 17:09 | **Mini Live Run** | ✅ Success | 2 real API calls + analysis pipeline verification |
| 2025-11-26 | ~18:25 | **TPM Limit Troubleshooting & Model Switch** | ✅ Resolved | Identified `gemini-2.0-flash-exp` had a 250k TPM limit. Switched to `models/gemini-2.0-flash` with confirmed 1M TPM. |
| 2025-11-27–29 | Multi | **Incremental partial runs** | ⚠️ Mixed | Created historical entries in `results/raw/exp1_results.jsonl` but status file resets caused duplicates. |
| 2025-11-30 | 12:05–20:52 | **Full run attempt** | ⚠️ Partial | Hit `ResourceExhausted` churn + 0.9-fill throttle skips; stopped with 264 configs pending. |
| 2025-12-01 | 00:10 | **Post-mortem audit** | ⚠️ Pending | `results/raw/exp1_pending_runs.json` + `exp1_failure_breakdown.json` generated for rerun planning. |

### ✅ Experiment 1 Remediation Sprint (Complete)

Goal: land the remaining 264 run keys, stabilize rerun tooling, and unlock analysis.

Highlights:
- ✅ Pending list cleared; all 3,000 run keys present in `results/raw/exp1_results.jsonl`.
-, ✅ Analysis generated under `results/analysis/exp1_rerun/`; conclusions summarized in `ARTICLE_CONCLUSIONS.md`.
-, ✅ Rate-limit checker now reflects real RPD by reading monitor state; monitor RPD set to free-tier limits for embeddings.
-, ✅ Per-minute token limit bumped to 1M for 0.9-fill runs.

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

1. Data readiness ✅ (corpus + padding + questions in place).
2. Readiness sprint ✅ (monitor separation, caching, CLI, scoring tooling, packaging/tests).
3. Experiment run ⚠️ **Partial Nov 30** (3,000 configs attempted; 2,736 recorded successes, 264 pending reruns per `results/raw/exp1_pending_runs.json`). RAG caches rebuilt with 990 k caps; final logs archived (`exp1_run_2025-11-30*.tar.gz`).
4. Analysis & publication ⏸ (blocked on rerun + dedup cleanup).

### Acceptance Checklist

- [x] GitHub corpus available.
- [x] Padding corpus available.
- [x] Question set validated (≥50).
- [x] Embedding caches rebuilt with 990 k max (Nov 27) and reused per restart.
- [ ] Runner completion (3,000 run keys recorded in `results/raw/exp1_status.json`; rerun backlog = 264).
- [ ] Automated scoring/visualizations for the full dataset (progress snapshot exists; rerun post-archival for official results).
- [ ] Documentation updates with Exp 1 findings (once scoring/viz done).

---

## 🧪 EXPERIMENT 2: Context Pollution (Weeks 3–6)

**Status:** 🛠️ Implementation phase (Exp 1 complete)

**Duration:** ~1–2 weeks  
**Goal:** Measure robustness to irrelevant information (pollution) across strategies.  
**Domain:** Fresh GitHub docs (base corpus) + Gutenberg padding (pollution).  
**Scope:** 1,200 generation calls (20 questions × 4 strategies × 5 pollution levels × 3 reps).  
**Model/limits:** `gemini-2.0-flash-lite-preview-02-05`, 1M context, 4M TPM, free-tier RPD 1,500 (Gen) / 1,000 (Emb).

### Step-by-Step Plan (detailed)

1) **Prereqs & gating (0.5 day)**
   - Confirm Exp 1 artifacts are archived and conclusions logged (done).
   - Verify monitor state is clean for today (`python scripts/check_rate_limits.py`).
   - Ensure venv active (`source venv/bin/activate`) and `pip install -e .` succeeds.

2) **Data prep (1–2 days)**
   - **Base corpus (relevant):** Collect 3–5 repos not used in Exp 1 (e.g., FastAPI, Pydantic, SQLAlchemy, Celery, Requests). Target ~50k tokens total; save to `data/raw/exp2/base_corpus.json`.
     - Command: author `scripts/collect_exp2_corpus.py` (or reuse `collect_*` pattern) to fetch and tokenize; log token counts.
     - Verify: `python - <<'PY' ...` to assert ≥50k tokens; file exists.
   - **Pollution corpus (irrelevant):** Reuse `data/raw/padding/gutenberg_corpus.json`; no new collection. If missing, copy from Exp 1 assets.
   - **Questions:** Expand `data/questions/exp2_questions.template.json` into `data/questions/exp2_questions.json` with 20 Qs (15 lookup, 5 synthesis) answerable only from base corpus; ensure answers aren’t in Gutenberg.
     - Validate: `python scripts/validate_question_set.py data/questions/exp2_questions.json --require-experiment exp2`.
     - Manual spot check: confirm required docs appear in base corpus, not in pollution corpus.

3) **Pollution injector utilities (0.5 day)**
   - Implement `src/corpus/pollution.py` with `PollutionInjector`:
     - `inject_pollution(base_docs, pollution_docs, target_pollution_tokens, strategy='append'|'interleave')`
     - Uses `count_tokens`/`truncate_to_tokens` to enforce budgets.
   - Tests: add `tests/test_pollution.py` to cover append/interleave, token budgeting, and idempotent behavior.
   - Verify: `pytest tests/test_pollution.py`.

4) **Experiment runner implementation (1 day)**
   - Implement `src/experiments/exp2_pollution.py`:
     - Strategies: reuse existing assemblers (naive, structured, rag, advanced_rag).
     - Pollution levels: [50k, 200k, 500k, 700k, 950k] tokens of noise added to a fixed base context (~50k tokens).
     - Generate mixed context via `PollutionInjector`, then assemble/pad to max context limit (respect 1M ceiling).
     - Three repetitions per question/strategy/pollution level; temperature=0.0; log tokens and latency.
     - Add `per_minute_token_limit` support (default 1,000,000) and `--limit` for smoke tests.
   - Runner script: add `scripts/run_experiment_2.py` and wire `scripts/run_experiment.py` handler for `exp2` (accepts `--runs-file`, `--dry-run`, `--per-minute-token-limit`, `--limit`).
   - Status tracking: mirror Exp 1 pattern (`results/raw/exp2_status.json`, `exp2_results.jsonl`, `exp2_pending_runs.json`).
   - Verify: `python scripts/run_experiment.py --experiment exp2 --dry-run --per-minute-token-limit 1000000 --limit 2`.

5) **Dry-run + smoke (0.5 day)**
   - Dry run to list planned 1,200 configs without API calls.
   - Smoke run with `--limit 3` to confirm RAG/structured paths work and monitor captures calls; ensure no token-limit warnings at chosen budgets.

6) **Full execution (1–2 days)**
   - Command: `python scripts/run_experiment.py --experiment exp2 --per-minute-token-limit 1000000`.
   - Monitor: `python scripts/check_rate_limits.py` between batches; stop if RPD approaches 1,400.
   - Success criteria: `wc -l results/raw/exp2_results.jsonl` == 1200; pending list empty.

7) **Analysis (0.5–1 day)**
   - Dedup if needed; score: `python scripts/analyze_results.py --input results/raw/exp2_results.jsonl --questions data/questions/exp2_questions.json --output-dir results/analysis/exp2 --mock-judge`.
   - Visuals: `python scripts/generate_visualizations.py --input results/analysis/exp2/summary_metrics.csv --output-dir results/visualizations/exp2`.
   - Key outputs: accuracy vs pollution level per strategy, hallucination/false-positive rate, robustness curves.
   - Summarize findings in `results/analysis/exp2/analysis_report.md` and update ARTICLE_CONCLUSIONS.md/README.

8) **Post-mortem + handoff (0.5 day)**
   - Archive logs/results; update PLAN/README with completion status and next steps (Exp 5).

### Acceptance Checklist (Exp 2)
- [ ] Base corpus present: `ls data/raw/exp2/base_corpus.json` and token count ≥50k.
- [ ] Questions validated: `python scripts/validate_question_set.py data/questions/exp2_questions.json --require-experiment exp2` exits 0.
- [ ] Pollution injector tests: `pytest tests/test_pollution.py` exits 0.
- [ ] Dry-run passes: `python scripts/run_experiment.py --experiment exp2 --dry-run --per-minute-token-limit 1000000 --limit 2` exits 0.
- [ ] Full run artifacts: `wc -l results/raw/exp2_results.jsonl` == 1200; `results/raw/exp2_status.json` shows completed=1200.
- [ ] Analysis artifacts: `results/analysis/exp2/summary_metrics.csv`, `results/analysis/exp2/analysis_report.md`, and plots under `results/visualizations/exp2/`.
- [ ] Conclusions propagated to ARTICLE_CONCLUSIONS.md/README.

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
# IMPORTANT: Use --per-minute-token-limit 1000000 for models/gemini-2.0-flash
python scripts/run_pilot.py              # Start with this!
python scripts/run_experiment_1.py --per-minute-token-limit 1000000       # After pilot succeeds
python scripts/run_experiment_2.py --per-minute-token-limit 1000000       # After Exp 1 succeeds

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
