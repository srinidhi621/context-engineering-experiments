# Context Engineering for LLMs: Experimental Suite

A rigorous experimental framework to test the importance of context engineering for Large Language Models, comparing long-context (1M tokens) and shorter-context (128k tokens) approaches.

For contributor practices and workflow expectations, see [Repository Guidelines](AGENTS.md).

### ⚖️ Token Limits & Throttling

- The pilot runner enforces a rolling per-minute input token cap to mirror Gemini quotas. The default fallback is 240 k tokens/min (free tier). Set `PER_MINUTE_TOKEN_LIMIT` in `.env` (or pass `--per-minute-token-limit`) to match your actual quota when using the paid tier (e.g., `3600000` for a 3.6 M guardrail).
- If a prompt exceeds the configured ceiling, the script logs a skip instead of letting the API return 429. This protects both free and paid tiers from accidental bursts, while still allowing 1 M-token contexts when the quota supports it.

## ⚠️ CURRENT STATUS: PHASE 1A COMPLETE

**✅ What's Complete (Infrastructure + Data Loaders):**
- ✅ Project scaffolding and directory structure
- ✅ Configuration system (config.py, .env)
- ✅ **Unified API Monitor** (rate limiting + cost tracking + budget enforcement in ONE system)
- ✅ API integration (GeminiClient wrapper with unified monitoring)
- ✅ **Corpus loaders implemented** (Hugging Face Hub + Gutenberg, 460+ lines)
  - `load_hf_model_card()` - Single model card loader
  - `load_hf_curated_models()` - Fast collection from 60+ recent models
  - `load_gutenberg_books()` - Classic literature loader
  - Tested: 50k tokens collected in 2-3 seconds
- ✅ **Enhanced tokenizer utilities** (counting, chunking with overlap, truncation)
- ✅ Logging infrastructure
- ✅ API key configured and verified
- ✅ **Model selected: gemini-2.0-flash-exp (6x faster than 2.5 Flash on free tier)**
- ✅ **End-to-end integration test passing**
- ✅ **google-generativeai upgraded to v0.8.5**

**🔄 In Progress (Pilot Phase):**
- ⏳ Phase 1B: Collect 10k tokens pilot corpus + create 1 test question
- ⏳ Phase 1C: Implement context assemblers (naive, padding, RAG enhancement)
- ⏳ Phase 1D: Create minimal runner (18 API calls)
- ⏳ Phase 1E: Go/No-Go decision

**❌ What's NOT Complete:**
- ❌ Context engineering implementations (naive, structured - only basic RAG done)
- ❌ Pilot corpus collected (need 10k tokens)
- ❌ Pilot question created (need 1 test question)
- ❌ Evaluation questions for full experiments (none generated yet)
- ❌ Metrics implementation (empty TODO)
- ❌ Experiment runner scripts (empty TODOs)
- ❌ Checkpoint/resume system (not implemented)

**Bottom Line:** Phase 1A complete (infrastructure + data loaders). Ready for Phase 1B (data collection).

**Revised Scope:** Dropped Exp 3 & 4 (too ambitious). Focus on Pilot → Exp 1 → Exp 2 → Analysis. Estimated 10-12 weeks total.

## 🎯 Data Strategy: Hugging Face + Gutenberg

**Why This Combination?**

1. **Post-Training Cutoff:** Model cards from Sept-Dec 2024 ensure fresh, unmemorized data (Gemini trained through Aug 2024)
2. **No Authentication Required:** Hugging Face Hub allows public read access - no API keys needed
3. **Verifiable:** Easy to check dates, trace sources, validate ground truth
4. **Clearly Separable:** Technical ML docs vs. classic literature = unambiguous relevance distinction
5. **Free & Fast:** No rate limits, no costs, no copyright concerns

**Data Sources:**
- **Pilot:** Llama 3.2/3.3 model cards (10k tokens)
- **Experiment 1 Corpus:** 60+ recent model cards from Hugging Face Hub:
  - Llama 3.2 & 3.3 (Meta, Sept-Dec 2024)
  - Qwen 2.5 family (Alibaba, Sept 2024)
  - Mistral Small/Pixtral/Ministral (Sept-Oct 2024)
  - Phi 3.5, Gemma 2, Whisper v3 Turbo, Stable Diffusion 3.5
  - **Target:** 700k tokens
- **Padding Corpus:** 15+ Gutenberg classics (Pride & Prejudice, Frankenstein, Sherlock Holmes, etc.) - 2M+ tokens
- **Experiment 2 Base:** Additional recent model cards (SmolLM2, etc.) - 50k tokens
- **Experiment 2 Pollution:** Reuse Gutenberg corpus (irrelevant to ML questions)

**Key Advantages:**
- Model cards have lastModified timestamps → easy to filter post-cutoff
- Comprehensive technical documentation (usage, parameters, capabilities)
- Fast collection: 50k tokens in 2-3 seconds
- Gutenberg provides high-quality, clean text for padding/pollution
- No ambiguity about what's relevant (ML docs) vs. irrelevant (Victorian novels)

## 🎯 Project Goal

Design a replicable experiment suite that isolates the impact of context engineering on LLM quality, cost, and latency—separately for:
- **(a)** Very-long-context models (~1M tokens)
- **(b)** Shorter-context models (~128k tokens)

## 🔬 Hypotheses

### H1: Long Context ≠ Magic Bullet
> "Even with 1M-token windows, naïvely stuffing context underperforms engineered retrieval + packaging."

**Predicted Results:**
- Engineered 1M context outperforms naïve 1M by ≥15% on quality
- At high pollution levels (≥50% irrelevant content), engineered maintains >90% accuracy vs <70% for naïve
- Cost per query is ≥20% lower for engineered due to better instruction-following

### H2: Smart Beats Big
> "128k-token models, with disciplined context engineering, can match or beat naïve long-context use on practical tasks."

**Predicted Results:**
- Advanced 128k RAG matches within 5% of naïve 1M on quality
- Advanced 128k RAG costs <40% of naïve 1M per query
- Advanced 128k RAG has <2x latency of naïve 1M

**Methodology Fix:** RAG strategies will be **padded to match fill %** of naive strategies. After retrieving relevant chunks, we pad with irrelevant content to reach the target context size. This ensures we're comparing context engineering quality, not confounding fill % with retrieval quality.

## ✅ FREE TIER CONFIGURATION (Optimized)

**Using Gemini 2.0 Flash Experimental:**
- **RPM (Requests Per Minute):** 15
- **TPM (Tokens Per Minute):** 1,000,000  
- **RPD (Requests Per Day):** 1,500
- **Cost:** $0.00 (free tier)

**Impact on Experiment Suite:**
- **Revised scope: 4,380 requests** (down from 9,000 - dropped Exp 3 & 4)
- At 1,500 requests/day: **~3 days minimum**
- TPM allows large contexts (up to 1M tokens)
- **Realistic estimate: 4-5 days** (with retry buffer)

**Automatic Enforcement:**
The unified monitor automatically enforces all limits and budget:
- Blocks requests exceeding RPM/TPM limits (waits for reset)
- Hard stops at RPD limit (must resume next day)
- Hard stops at budget limit ($174)
- No manual tracking needed

### Unified Monitoring System

The project uses a **single unified monitor** that handles everything:

**Features:**
1. **Rate Limiting:** Enforces RPM, TPM, RPD limits automatically
2. **Cost Tracking:** Tracks tokens and costs comprehensively
3. **Budget Enforcement:** Hard stops at $174 limit
4. **Experiment Tracking:** Tag calls with experiment_id and session_id
5. **Persistent State:** Survives restarts (saved to `results/.unified_monitor_state.json`)
6. **No Conflicts:** Single source of truth for all usage

**Benefits Over Separate Systems:**
- ✅ No duplicate tracking
- ✅ No state file conflicts
- ✅ Budget + rate limits enforced together
- ✅ Simpler API (one system to integrate)
- ✅ Comprehensive reporting

### Check Feasibility FIRST

```bash
# Run this before starting experiments
python scripts/estimate_feasibility.py

# This will show you:
# - Time needed for each experiment
# - Total days required
# - Recommendations for staying within limits
```

### Monitor Usage

```bash
# Check current usage anytime
python scripts/check_rate_limits.py

# Shows:
# - Requests used this minute/day
# - Tokens consumed
# - % utilization of each limit
```

### Options for Large-Scale Experiments

**Option 1: Reduced Scope (Free Tier)**
- Reduce repetitions: 3 → 2 (saves 33%)
- Reduce fill levels: 5 → 3 (saves 40%)  
- Focus on 2-3 key experiments
- **Estimated time: 4-5 days**

**Option 2: Multiple API Keys**
- Create 2-3 Google accounts
- Get separate API keys (free per project)
- Run experiments in parallel
- **Estimated time: 2-3 days**

**Option 3: Upgrade to Paid Tier (Recommended)**
- Gemini 2.0 Flash is **STILL FREE** on paid tier
- But rate limits increase dramatically:
  - RPM: 15 → 1,000 (66x faster)
  - TPM: 1M → 4M (4x more)
  - RPD: 1,500 → unlimited
- Just need credit card on file
- **Estimated time: <1 day**

**Option 4: Hybrid (Most Practical)**
- Test methodology on 10% sample (free tier)
- Validate approach and code
- Then upgrade to paid tier for full run
- **Best of both worlds**

### Rate Limit Details

**How the Rate Limiter Works:**
1. **Pre-Request Token Estimation:** Counts tokens BEFORE making API call
2. **Automatic Waiting:** Blocks execution when limits are approached
3. **Persistent State:** Survives restarts (saved to `results/.rate_limiter_state.json`)
4. **Daily Reset:** Automatically handles midnight PT reset for RPD
5. **Smart Backoff:** Handles 429 errors with exponential backoff

**Important Notes:**
- Limits are per-project, not per-key (multiple keys in same project don't help)
- Per-minute limits use rolling 60-second window
- Per-day limits reset at midnight Pacific Time
- State persists across computer restarts/power failures

## 📊 Experimental Design

### Model Configuration ✅ OPTIMIZED FOR FREE TIER

**✅ CONFIGURED: gemini-2.0-flash-exp**

This project uses **gemini-2.0-flash-exp** for optimal free tier performance:

| Metric | Value | Comparison |
|--------|-------|------------|
| **RPM** | 15 | Standard |
| **TPM** | 1,000,000 | Excellent for long contexts |
| **RPD** | 1,500 | **6x better than 2.5 Flash** |
| **Cost** | $0.00 | Free tier |

**Timeline Impact:**
- ✅ **9,000 requests: ~6 days minimum** (gemini-2.0-flash-exp)
- ❌ Alternative (gemini-2.5-flash): ~36 days (250 RPD limit)

**Unified Monitoring:**
- ✅ Rate limiting (RPM, TPM, RPD enforcement)
- ✅ Cost tracking (by experiment, session, model, day)
- ✅ Budget enforcement ($174 limit from project plan)
- ✅ All in one system - no conflicts or inconsistencies

**Configuration:**
- **Primary Model:** gemini-2.0-flash-exp (production-ready, free tier)
- **Embedding Model:** text-embedding-004 (latest, free tier)
- **Temperature:** 0.0 (deterministic)
- **Repetitions:** 3 runs per condition per question
- **Budget Limit:** $174 (enforced automatically)

### Key Variables

**Independent Variables:**
- Context engineering approach (naïve, structured, RAG variants)
- Context fill percentage (10%, 30%, 50%, 70%, 90%)

**Dependent Variables:**
- **Quality:** Correctness (0-1), citation accuracy, completeness
- **Cost:** Token usage, API costs per query
- **Latency:** Total response time
- **Robustness:** Performance degradation vs. fill % and pollution level

### Experimental Conditions

| Condition | Description | Context Window | Engineering |
|-----------|-------------|----------------|-------------|
| **Naïve 1M** | Sequential document dump | 1M tokens | None |
| **Engineered 1M** | Hierarchical TOC + metadata | 1M tokens | Full |
| **RAG 128k** | Vector search + reranking | 128k tokens | Retrieval |
| **Advanced RAG 128k** | Hybrid search + query decomposition | 128k tokens | Advanced |

### Critical Control: Fill Percentage

To isolate context engineering effects from context window utilization effects:

```
Fill %  | Tokens Used | What It Tests
--------|-------------|------------------------------------------
10%     | 100k        | Baseline (minimal attention dilution)
30%     | 300k        | Moderate usage
50%     | 500k        | Half-full (significant degradation starts)
70%     | 700k        | High utilization
90%     | 900k        | Near-limit (maximum attention strain)
```

**Implementation:** Pad contexts with domain-irrelevant content to reach target fill percentage at each level.

**Why This Matters:** Models exhibit "Lost in the Middle" phenomenon—recall accuracy degrades as context fills up, independent of content quality. Controlling fill % ensures we measure engineering quality, not just fill % artifacts.

## 🧪 Experiment Suite (Revised Scope)

### Pilot Phase: Validate Pipeline
**Purpose:** End-to-end validation before scaling up

- **Corpus:** 10k tokens from recent model cards (Llama 3.2/3.3, Qwen 2.5)
- **Strategies:** Naïve 1M + Basic RAG 128k (test both extremes)
- **Tasks:** 1 test question with ground truth (minimal validation)
- **API Calls:** 18 (1 question × 2 strategies × 3 fill levels × 3 reps)
- **What It Tests:** Pipeline functionality, context assembly, API integration
- **Data Source:** Hugging Face Hub (no authentication needed, guaranteed post-cutoff)

### Experiment 1: Needle in Multiple Haystacks
**Purpose:** Test retrieval quality vs. context stuffing under information overload

- **Corpus:** 500k-1M tokens of recent model cards from Hugging Face Hub
  - 60+ models (Llama 3.2/3.3, Qwen 2.5 family, Mistral, Phi 3.5, Gemma 2, etc.)
  - Comprehensive model documentation (architecture, parameters, usage, capabilities)
  - All content from Sept-Dec 2024 (post-training cutoff)
- **Tasks:** 50 questions (20 lookups, 20 synthesis, 10 contradiction detection)
- **API Calls:** 3,000 (50 questions × 4 strategies × 5 fill levels × 3 reps)
- **What It Tests:** Multi-document reasoning, cross-referencing
- **Metrics:** Correctness, citation accuracy, cost per query
- **Data Source:** Hugging Face Hub (no auth needed, fast, reliable)

### Experiment 2: Context Pollution
**Purpose:** Measure robustness to irrelevant information

- **Base Corpus:** 50k tokens relevant ML documentation (recent model cards)
- **Pollutant:** Add 50k → 950k tokens of classic literature (Project Gutenberg)
  - Clearly irrelevant to ML/technical questions
  - Pre-cleaned plain text (no parsing needed)
  - Mix of fiction and non-fiction from different eras
- **Tasks:** 20 questions strictly answerable from base corpus
- **API Calls:** 1,200 (20 questions × 4 strategies × 5 pollution levels × 3 reps)
- **What It Tests:** Resistance to distraction, precision
- **Metrics:** Accuracy vs pollution level, false positive rate
- **Data Sources:** Hugging Face Hub + Project Gutenberg (both free, no auth needed)

### Experiment 5: Cost-Latency Frontier
**Purpose:** Map Pareto frontier of quality vs. cost vs. latency

- **Analysis:** 3D optimization across Experiments 1-2
- **Output:** Dominant strategies (no approach beats them on all 3 metrics)
- **API Calls:** 0 (analysis only)
- **What It Tests:** Real-world deployment trade-offs
- **Metrics:** Efficiency score = Quality / (Cost × Latency)

### ❌ Experiments 3 & 4: DROPPED
**Experiment 3 (Multi-Turn Memory)** and **Experiment 4 (Precision Retrieval)** have been removed to keep the project achievable within 10-12 weeks. They required complex implementation (stateful conversations, PDF parsing) with limited generalizability. Can be added as future work.

## 🏗️ Project Structure

```
context-engineering-experiments/
├── README.md                      # This file
├── PLAN.md                        # Detailed implementation roadmap
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Experiment configuration
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── gemini_client.py     # Gemini API wrapper
│   │
│   ├── context_engineering/
│   │   ├── __init__.py
│   │   ├── naive.py             # Naïve context assembly
│   │   ├── structured.py        # Engineered context (TOC, metadata)
│   │   ├── rag.py               # Basic RAG pipeline
│   │   └── advanced_rag.py      # Advanced RAG (hybrid search, etc.)
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── base_experiment.py   # Base experiment class
│   │   ├── exp1_needle.py       # Experiment 1
│   │   ├── exp2_pollution.py    # Experiment 2
│   │   ├── exp3_memory.py       # Experiment 3
│   │   ├── exp4_precision.py    # Experiment 4
│   │   └── exp5_frontier.py     # Experiment 5
│   │
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Corpus loading utilities
│   │   ├── generators.py        # Synthetic corpus generation
│   │   └── padding.py           # Fill % padding generation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Correctness, cost, latency metrics
│   │   ├── judges.py            # LLM-as-judge evaluation
│   │   └── human_eval.py        # Human evaluation interface
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tokenizer.py         # Token counting
│       ├── logging.py           # Structured logging
│       └── stats.py             # Statistical analysis
│
├── data/
│   ├── raw/                      # Original corpus files
│   │   ├── api_docs/
│   │   ├── financial_reports/
│   │   ├── academic_papers/
│   │   └── padding_corpus/
│   ├── processed/                # Preprocessed data
│   └── questions/                # Evaluation questions + ground truth
│       ├── exp1_questions.json
│       ├── exp2_questions.json
│       ├── exp3_questions.json
│       └── exp4_questions.json
│
├── results/
│   ├── raw/                      # Raw experiment outputs
│   ├── metrics/                  # Computed metrics
│   ├── analysis/                 # Statistical analysis
│   └── visualizations/           # Plots and charts
│
├── notebooks/
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_baseline_calibration.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_visualization.ipynb
│
├── scripts/
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_calibration.py        # Baseline fill % calibration
│   ├── analyze_results.py        # Post-experiment analysis
│   └── generate_report.py        # Final report generation
│
└── tests/
    ├── __init__.py
    ├── test_context_engineering.py
    ├── test_corpus.py
    ├── test_evaluation.py
    └── test_models.py
```

## 🚀 Quick Start (Command Line Ready)

### Prerequisites
- Python 3.10+ (tested on 3.13.3)
- Virtual environment capability
- Google API key from https://aistudio.google.com/app/apikey

### One-Time Setup

```bash
# Clone repository
git clone <your-repo-url>
cd context-engineering-experiments

# Run automated setup
bash scripts/setup_environment.sh

# This will:
# ✅ Create virtual environment (venv/)
# ✅ Install all dependencies (15 packages)
# ✅ Create data directories
# ✅ Configure environment (.env file)
# ✅ Run verification tests
```

### Daily Activation & Command Line Usage

```bash
# Activate environment (do this every time you start work)
cd /path/to/context_engineering_experiments
source scripts/activate.sh

# Verify installation
python scripts/test_api_integration.py
# Expected: ✅ ALL TESTS PASSED

# Check monitoring status
python scripts/generate_cost_report.py

# Monitor API usage
python scripts/monitor_costs.py

# When done
deactivate
```

All scripts are executable from command line with `python scripts/<script_name>.py`

### Verify Everything Works

```bash
# Run comprehensive integration test (makes 1 real API call)
python scripts/test_api_integration.py

# Expected output:
# ✅ Client initialized
# ✅ Unified monitor configured
# ✅ API call successful
# ✅ Experiment tracking working
# ✅ ALL TESTS PASSED

# Check current costs and usage
python scripts/generate_cost_report.py
```

### Configure API Key

```bash
# The setup script creates .env from .env.example
# Edit .env and add your actual GOOGLE_API_KEY:

nano .env
# Set: GOOGLE_API_KEY=your_actual_api_key_here

# Get your key at: https://aistudio.google.com/app/apikey
```

### Download Corpus Data

```bash
# (Optional - download when ready for experiments)

# Experiment 1 model-card corpus (~700k tokens)
python scripts/collect_exp1_corpus.py --dry-run   # inspect stats
python scripts/collect_exp1_corpus.py             # write to data/raw/exp1/

# Gutenberg padding corpus (~2M tokens)
python scripts/collect_padding_corpus.py --dry-run
python scripts/collect_padding_corpus.py

# Download all corpora (~4GB, 20-60 min):
bash scripts/download_all.sh

# OR download selectively:
python -c "from src.corpus.loaders import download_api_docs; download_api_docs()"
python -c "from src.corpus.loaders import download_financial_reports; download_financial_reports()"
python -c "from src.corpus.loaders import download_academic_papers; download_academic_papers()"
```

### Run Experiments

```bash
# After environment is activated:

# 1. Check feasibility first
python scripts/estimate_feasibility.py

# 2. Run PILOT first (validate entire pipeline)
python scripts/run_pilot.py --output results/pilot_results.jsonl

# 3. If pilot succeeds, run Experiment 1
python scripts/run_experiment_1.py --output results/exp1_results.jsonl

# 3b. Score pilot results before proceeding
python scripts/evaluate_pilot_manually.py

# Unified runner scaffolding (preferred for future experiments)
python scripts/run_experiment.py --experiment pilot --dry-run
python scripts/run_calibration.py --dry-run
python scripts/validate_question_set.py data/questions/exp1_questions.template.json

# 4. Run Experiment 2
python scripts/run_experiment_2.py --output results/exp2_results.jsonl

# 5. Analyze results
python scripts/analyze_results.py --input results/raw/ --output results/analysis/

# 6. Generate report
python scripts/generate_report.py --output FINAL_REPORT.md
```

---

## 📋 Installation

**Set up Python virtual environment**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

**Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

**Set up environment variables**
  ```bash
  cp .env.example .env
  # Edit .env with your API keys:
  # GOOGLE_API_KEY=your_actual_api_key_here
  # PER_MINUTE_TOKEN_LIMIT=3600000   # Example paid-tier guardrail
  ```

### Question Set Authoring

- Start from `data/questions/exp1_questions.json` (in-progress set) or copy the `.template.json` files when authoring new batches.
- After editing any question file run `python scripts/validate_question_set.py <file>` to enforce field coverage, unique IDs, and required document lists before committing.
- Corpus collection helpers:
  - `scripts/collect_exp1_corpus.py` → Hugging Face model cards for Experiment 1.
  - `scripts/collect_padding_corpus.py` → Project Gutenberg padding corpus shared across experiments.

---

## 🚀 Run Guide

### Current Status
- **Phase 1A** (infrastructure + loaders): ✅ complete.
- **Phase 1B** (pilot corpus + question + runner): in progress (data scripts + pilot runner + scoring script exist; waiting on quota increase to finish API runs).
- **Full experiments** (Exp1/Exp2 runners + analysis): scaffolds exist but need corpora, question sets, and validation.

### Environment Checklist
1. Activate the virtualenv (every session):
   ```bash
   source venv/bin/activate
   # or: source scripts/activate.sh
   ```
2. Confirm `.env` contains `GOOGLE_API_KEY=...` and (optionally) `PER_MINUTE_TOKEN_LIMIT=...`.
3. Quick sanity check:
   ```bash
   python -c "from src.config import api_config; print(bool(api_config.google_api_key))"
   ```

### Smoke Tests & Monitoring
- **API integration:** `python scripts/test_api_integration.py`
- **API key verification:** `python scripts/verify_api_key.py`
- **Monitor usage / limits:**  
  `python scripts/monitor_costs.py`, `python scripts/generate_cost_report.py`, `python scripts/check_rate_limits.py`, `python scripts/estimate_feasibility.py`
- **Corpus loaders:** run small snippets (examples in README “Download Corpus Data”) to ensure HF + Gutenberg fetches succeed.

### What’s Ready vs Pending
- ✅ `scripts/run_minimal_pilot.py` (with throttling) + `scripts/evaluate_pilot_manually.py`
- ✅ Corpus collectors (`collect_pilot_corpus.py`, `collect_exp1_corpus.py`, `collect_padding_corpus.py`)
- ⚠️ `scripts/run_experiment_1.py`, `scripts/run_experiment_2.py`, and advanced analysis scripts are scaffolds/placeholders—do not expect full functionality yet.

### Pilot Workflow Snapshot (Phase 1B)
1. `python scripts/collect_pilot_corpus.py`
2. Populate `data/questions/pilot_question_01.json`
3. `python scripts/build_pilot_contexts.py`
4. `python scripts/run_minimal_pilot.py --dry-run` (until quotas ready) then run for real
5. `python scripts/evaluate_pilot_manually.py`

### Troubleshooting Cheatsheet
- `ModuleNotFoundError: src`: ensure you’re at repo root, venv is active, and `pip install -e .` was run.
- `GOOGLE_API_KEY environment variable not set`: confirm `.env` exists and re-source it.
- Missing packages (e.g., `gutenbergpy`): re-run `pip install -r requirements.txt`.
- Token quota 429s: adjust `PER_MINUTE_TOKEN_LIMIT` or wait/retry—pilot runner now surfaces these as skips.

### Next Steps & References
- Follow detailed sequencing in `PLAN.md` for upcoming tasks (Exp1 corpora, question authoring, runner integration).
- Use `python scripts/validate_question_set.py ...` whenever editing question files.
- Keep an eye on `results/.monitor_state.json` (auto-managed by the unified monitor) for quota/budget guardrails.

**Verify setup:**
  ```bash
  # Check rate limits
  python scripts/check_rate_limits.py

  # Estimate experiment feasibility
  python scripts/estimate_feasibility.py
  ```

### Available Command Line Tools

**Testing & Verification:**
```bash
python scripts/test_api_integration.py    # End-to-end integration test (use this first!)
python scripts/verify_api_key.py          # Verify your Google API key
```

**Monitoring & Reporting:**
```bash
python scripts/generate_cost_report.py              # Comprehensive cost report
python scripts/monitor_costs.py                     # Quick status check
python scripts/monitor_costs.py --by-day            # Daily breakdown
python scripts/monitor_costs.py --by-experiment     # Experiment breakdown
python scripts/check_rate_limits.py                 # Check rate limits
python scripts/estimate_feasibility.py              # Estimate experiment timeline
```

**Experiments (Not Yet Implemented):**
```bash
python scripts/run_experiment.py          # Run experiments
python scripts/run_calibration.py         # Run calibration
python scripts/analyze_results.py         # Analyze results
python scripts/generate_report.py         # Generate final report
```

### Monitoring Costs

```bash
# View current costs and usage
python scripts/monitor_costs.py

# View comprehensive report with experiment/session breakdowns
python scripts/generate_cost_report.py

# Export report to file
python scripts/generate_cost_report.py --save cost_report.txt

# Get JSON output for programmatic access
python scripts/generate_cost_report.py --format json

# View daily breakdown
python scripts/monitor_costs.py --by-day

# View hourly breakdown
python scripts/monitor_costs.py --by-hour
```

The unified monitoring system tracks:
- ✅ **Rate limits:** RPM, TPM, RPD with auto-enforcement
- ✅ **API calls:** Total, by experiment, session, model, day
- ✅ **Tokens:** Input, output, total (all breakdowns)
- ✅ **Costs:** Calculated with current pricing (all breakdowns)
- ✅ **Budget:** $174 limit enforced automatically
- ✅ **Persistent:** Survives restarts (`results/.unified_monitor_state.json`)

**Key Improvement:**
The unified system prevents exceeding budget or rate limits before making the API call, not after. This protects your free tier quota.

### Running Experiments (NOT YET IMPLEMENTED)

```bash
# These scripts are scaffolds only - not yet implemented:

# 1. Calibrate baseline
python scripts/run_calibration.py --output results/baseline_calibration.json

# 2. Run individual experiment
python scripts/run_experiment.py --experiment exp1_needle --conditions all

# 3. Analyze results
python scripts/analyze_results.py --input results/raw/ --output results/analysis/

# 4. Generate final report
python scripts/generate_report.py --output FINAL_REPORT.md
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_context_engineering.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📈 Success Criteria

### H1 is SUPPORTED if:
- ✅ Engineered 1M context > Naïve 1M by ≥15% quality
- ✅ At ≥2 pollution levels, engineered maintains >90% accuracy vs <70% naïve
- ✅ Cost per query ≥20% lower for engineered

### H2 is SUPPORTED if:
- ✅ Advanced 128k RAG within 5% of naïve 1M quality
- ✅ Advanced 128k RAG costs <40% of naïve 1M
- ✅ Advanced 128k RAG latency <2x naïve 1M

### Both REJECTED if:
- ❌ Naïve long-context wins on quality by >10% with no cost disadvantage
- (Would indicate context engineering is unnecessary—surprising but valid result!)

## 💰 Budget Estimation

**API Costs (Gemini 2.0 Flash Experimental):**
- Input: $0.00 per 1k tokens (free tier)
- Output: $0.00 per 1k tokens (free tier)

**Estimated Usage (Revised Scope):**
- 4,380 queries × avg 500k input tokens = 2.2B input tokens
- 4,380 queries × avg 2k output tokens = 9M output tokens

**Total Cost:** **$0** (completely free on free tier)

**Rate Limit Constraint:** 1,500 requests/day → ~3 days minimum, ~4-5 days realistic

## 🔍 Key Insights Expected

1. **Fill % matters more than raw context size** - The "Lost in the Middle" effect dominates performance
2. **Engineering amplifies at high utilization** - Benefits increase as context fills up
3. **RAG avoids the fill % trap** - Never overfills context, maintains consistent quality
4. **Cost-quality Pareto frontier exists** - No single approach dominates all metrics
5. **Position matters** - Information placement within context significantly affects recall

## 📊 Expected Output

At the end of this project, you will have:

1. **Quantitative Evidence:**
   - Statistical validation of both hypotheses
   - Effect sizes for each engineering technique
   - Cost-benefit analysis for practitioners

2. **Visualizations:**
   - Fill % degradation curves
   - Pareto frontiers (quality vs cost vs latency)
   - Pollution robustness charts
   - Position bias heatmaps

3. **Actionable Recommendations:**
   - When to use RAG vs long context
   - Optimal chunk sizes and retrieval strategies
   - Cost optimization strategies
   - Architecture decision flowchart

4. **Reproducible Methodology:**
   - Complete codebase with tests
   - Documented evaluation metrics
   - Statistical analysis templates
   - Example corpora and questions

## 🔬 Methodology Highlights

### Why This Design is Rigorous

1. **Controlled Variables:** Fill % and position are explicitly controlled
2. **Multiple Experiments:** 5 different tasks test different aspects
3. **Statistical Power:** 3 repetitions × multiple fill levels = robust results
4. **Realistic Tasks:** Based on real-world use cases (documentation, reports, papers)
5. **Comprehensive Metrics:** Quality, cost, latency, and robustness all measured
6. **Blind Evaluation:** Human eval uses blinded samples to avoid bias

### Known Limitations

1. **Model-Specific:** Results specific to Gemini Flash 2.5
2. **Fill % Confound:** 128k conditions run at lower fill % (acknowledged in H2)
3. **English Only:** All experiments use English-language corpora
4. **Synthetic Scenarios:** Some experiments (e.g., customer support) are simulated
5. **LLM Judge Bias:** Automated evaluation uses another LLM, which has its own biases

## 📚 References

**Key Papers:**
- Liu et al. (2023) - "Lost in the Middle: How Language Models Use Long Contexts" ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172))
- Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401))
- Antropic (2024) - "Contextual Retrieval" ([Blog Post](https://www.anthropic.com/news/contextual-retrieval))

**Documentation:**
- Google Gemini API: [ai.google.dev](https://ai.google.dev/gemini-api/docs)
- Gemini Context Caching: [Docs](https://ai.google.dev/gemini-api/docs/caching)

**Tools & Libraries:**
- tiktoken: Token counting
- FAISS: Vector store for dense retrieval
- rank-bm25: Sparse retrieval for hybrid search
- huggingface_hub: Hugging Face API access for model/dataset cards
- gutenbergpy: Project Gutenberg access for padding corpus
- scipy: Statistical analysis
- plotly: Interactive visualizations

## 🤝 Contributing

This is a research project. If you'd like to:
- **Replicate:** Follow the setup instructions and PLAN.md
- **Extend:** Add new experiments or conditions
- **Improve:** Submit PRs for code improvements

Please open an issue first to discuss major changes.

## 📝 License

MIT License - See LICENSE file for details

## 👥 Author

**Your Name**
- Research Question Design: Claude (Anthropic) conversation
- Implementation: [Your Name]
- Institution/Organization: [Your Org]

## 🙏 Acknowledgments

- Google AI for Gemini API access
- Anthropic for experimental design guidance
- Research community for prior work on context windows and RAG

## 📞 Contact

For questions about this research:
- Email: [your.email@example.com]
- GitHub Issues: [repository-url]/issues
- Twitter/X: [@yourhandle]

---

**Last Updated:** November 3, 2025  
**Version:** 1.2 (Phase 1A Complete)  
**Status:** Phase 1A Complete - Ready for Phase 1B  
**Estimated Completion:** 10-12 weeks from start  
**Changes:** Phase 1A infrastructure complete (corpus loaders, tokenizer utils)
