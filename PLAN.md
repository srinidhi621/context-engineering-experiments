# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 10-12 Weeks (Realistic)  
**Start Date:** October 30, 2025  
**Budget:** $0 (Free Tier - gemini-2.0-flash-exp)  
**Team Size:** 1  

**Last Updated:** November 24, 2025  
**Status:** ✅ Pilot Complete - Ready for Experiment 1

---

## 📊 Current Status

**Infrastructure:** ✅ **Complete & Production-Ready**
- Python 3.13.3 environment configured
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

**Experiments:** ⏳ **Ready to Start**
- ⏳ Experiment 1: Needle in Multiple Haystacks (Next)
- ⏳ Experiment 2: Context Pollution
- ⏳ Experiment 5: Cost-Latency Frontier

**Time Spent:** ~28 hours on infrastructure & pilot  
**Remaining:** ~9-11 weeks for experiments + analysis

**Repository:** https://github.com/srinidhi621/context-engineering-experiments

---

## 🎯 Execution Strategy: Iterative & Practical

**Philosophy:** Validate small, then scale. Build incrementally with continuous validation.

**Approach:**
```
Pilot (18 calls) → Validate → Build Core → Validate → Full Scale (4,380 calls)
```

**Revised Scope (Focused & Achievable):**
1. **Pilot Phase** - 1 question, 2 strategies, validate entire pipeline (18 API calls)
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

## ✅🧪 PILOT PHASE: COMPLETE

The pilot phase was completed successfully, validating the entire experimental pipeline. Key learnings and a full summary are documented in [results/pilot_summary.md](./results/pilot_summary.md).

---
## 🧪 EXPERIMENT 1: Needle in Multiple Haystacks (Weeks 2-4)

**Status:** ⏳ Planned

**Duration:** 2-3 weeks  
**Goal:** Establish baseline and test all 4 context strategies across fill levels  
**Hypothesis:** Engineered 1M > Naïve 1M by ≥15% at high fill %  
**Domain:** GitHub Repository Documentation  
**API Calls:** 3,000 (50 questions × 4 strategies × 5 fill levels × 3 reps)  
**Est. Cost:** $0 (free tier)

**CRITICAL:** RAG strategies will be **padded to match fill %** of naive strategies (implemented in pilot phase).

### Week 2: Data Collection & Question Generation (7 days)

**Day 1-2: Collect GitHub Corpus (~700k tokens from 30 repos)**

Use `scripts/collect_exp1_corpus.py` to fetch documentation from 30 popular repositories across different languages/frameworks:
- Python ML/AI (PyTorch, TensorFlow, scikit-learn, HuggingFace)
- JavaScript/TypeScript (Next.js, React, Node.js, TypeScript)
- Go (Golang, Kubernetes, Docker)
- Rust (rust-lang, tokio)
- Other (Django, Flask, Rails, Terraform, etc.)

All docs filtered to modifications after 2024-08-01.
Save to: `data/raw/exp1/github_corpus.json`

**Day 3: Collect Gutenberg Padding Corpus (2M+ tokens)**

Use `scripts/collect_padding_corpus.py` to fetch 15+ classic books from Project Gutenberg.
Books: Pride & Prejudice, Frankenstein, Tale of Two Cities, Sherlock Holmes, etc.
Save to: `data/raw/padding/gutenberg_corpus.json`

**Day 4-7: Generate 50 Questions with Ground Truth**

Manually/semi-automatically create questions by reading the collected docs:
- 20 simple lookups (single fact from single doc)
- 20 synthesis (combine info from 2-3 docs)
- 10 contradiction detection (find conflicts or compare statements)

Each question must have:
- Ground truth answer (verifiable from docs)
- Required documents list
- Evaluation criteria
- Keywords for retrieval testing

Save to: `data/questions/exp1_questions.json`

**Acceptance Criteria:**
- [ ] **GitHub Corpus:** `python scripts/collect_exp1_corpus.py --dry-run` reports >600k tokens, then `python scripts/collect_exp1_corpus.py` runs successfully and `ls data/raw/exp1/github_corpus.json` lists the file.
- [ ] **Padding Corpus:** `python scripts/collect_padding_corpus.py --dry-run` reports >1.5M tokens, then `python scripts/collect_padding_corpus.py` runs successfully and `ls data/raw/padding/gutenberg_corpus.json` lists the file.
- [ ] **Question Set:** `python scripts/validate_question_set.py data/questions/exp1_questions.json --require-experiment exp1` exits with code 0.
- [ ] **Question Count:** `python -c "import json; f = open('data/questions/exp1_questions.json'); data = json.load(f); assert len(data['questions']) >= 50, f'Expected 50+ questions, found {len(data[\'questions\'])}'"` exits with code 0.

### Phase 2: Implementation (~3 days)

**Context Assemblers to Build:**

- [ ] **Naïve** (`src/context_engineering/naive.py`)
  - Sequential concatenation + padding
  - Token-aware truncation
  - No structure

- [ ] **Structured** (`src/context_engineering/structured.py`)
  - **Task:** Before implementation, add a section to this `PLAN.md` file detailing the proposed data structure (e.g., XML schema vs JSON, format of the Table of Contents, metadata fields). This ensures the implementation aligns with the experimental goals.
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

**Acceptance Criteria:**
- [ ] **Unit Tests:** `pytest tests/test_context_engineering.py` and `pytest tests/test_corpus.py` exit with code 0, indicating all assemblers and helpers are working as expected.
- [ ] **Manual Check:** The `PLAN.md` file has been updated with the design for the `Structured` context assembler as per the task.

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
  - **Seed Management:** Accept a `--seed` command-line argument to ensure full reproducibility of any stochastic processes (e.g., padding generation).
  - Load questions and corpus
  - For each question × strategy × fill level × rep:
    - Assemble context
    - Query model
    - Log response
  - Progress tracking
  - Error handling and retry
  - **Robust Checkpointing:** The script MUST save its state after each API call to a temporary file. On startup, it should check for this file to resume from the last successful call, making it resilient to interruptions. This is a core requirement, not an enhancement.

**Acceptance Criteria:**
- [ ] **Dry Run:** `python scripts/run_experiment_1.py --dry-run` exits with code 0 and logs that it would make 3,000 API calls.
- [ ] **File Creation:** After a full run, `ls results/raw/exp1_results.jsonl` successfully lists the file.
- [ ] **Result Count:** `wc -l results/raw/exp1_results.jsonl` reports 3000.

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

**Task 5.1: Implement and Calibrate LLM-as-Judge**
- **Status:** ⏳ Planned
- **Action:** Before full-scale analysis, implement the `src/evaluation/judges.py` module.
- **Calibration:** Create a "golden set" of ~50-100 manually scored results from the pilot or a small batch from Experiment 1. Run the LLM-as-judge on this set and measure its accuracy, bias, and correlation with human scores. This step is critical to ensure the reliability of automated evaluation at scale.
- **Acceptance:**
  - [ ] `judges.py` is implemented.
  - [ ] Calibration report shows >90% agreement with manual scores.

**Metrics to Compute:**
- [ ] Correctness (using the calibrated LLM-as-judge)
- [ ] Citation accuracy (claims grounded in context?)
- [ ] Cost per query
- [ ] Latency statistics

**Acceptance Criteria:**
- [ ] **Analysis Script:** `python scripts/analyze_results.py --input results/raw/exp1_results.jsonl --output-dir results/analysis` exits with code 0.
- [ ] **Metrics File:** `ls results/analysis/exp1_metrics.csv` successfully lists the file.
- [ ] **Analysis File:** `ls results/analysis/exp1_analysis.json` successfully lists the file.
- [ ] **Visualization Script:** `python scripts/generate_visualizations.py --input results/analysis/exp1_metrics.csv --output-dir results/visualizations/exp1` exits with code 0.
- [ ] **Plot Verification:** `ls results/visualizations/exp1/*.png | wc -l` reports a count of 3 or more.

**Key Question:** Does Engineered 1M beat Naïve 1M by ≥15%?

---

## 🧪 EXPERIMENT 2: Context Pollution (Weeks 5-6)

**Status:** ⏳ Planned

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

## 🧪 EXPERIMENT 5: Cost-Latency Frontier (Weeks 7-8)

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

## 📊 Final Analysis & Reporting (Weeks 9-12)

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
