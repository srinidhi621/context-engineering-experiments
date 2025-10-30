# Context Engineering for LLMs: Experimental Suite

A rigorous experimental framework to test the importance of context engineering for Large Language Models, comparing long-context (1M tokens) and shorter-context (128k tokens) approaches.

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

**Note:** H2 uses Option A (first 128k tokens only, no padding), which means RAG conditions run at lower fill % (~13%) than naïve 1M (up to 90%). This is an acknowledged limitation—results may conflate RAG quality with fill % effects.

## ⚠️ CRITICAL: Free Tier Rate Limits

**Using Gemini 2.0 Flash Experimental (Free Tier):**
- **RPM (Requests Per Minute):** 15
- **TPM (Tokens Per Minute):** 1,000,000
- **RPD (Requests Per Day):** 1,500

**Impact on 9,000+ Request Experiment Suite:**
- At 1,500 requests/day max, full suite would take **~6 days minimum**
- At average 500k tokens/request, TPM becomes the bottleneck
- **Realistic estimate: 2-3 weeks** with careful batching

### Rate Limiting Strategy

The project includes comprehensive rate limiting:

1. **Pre-Request Token Estimation:** Counts tokens BEFORE API call
2. **Automatic Wait Logic:** Blocks when limits approached
3. **Persistent State:** Survives restarts, tracks across sessions
4. **Daily Reset Handling:** Automatically resets at midnight PT
5. **Smart Backoff:** Handles 429 errors gracefully

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

### Model Configuration
- **Primary Model:** Google Gemini Flash 2.5 (1M token context window)
- **Simulated 128k:** First 128k tokens only (Option A)
- **Temperature:** 0.0 (deterministic)
- **Repetitions:** 3 runs per condition per question

### Key Variables

**Independent Variables:**
- Context engineering approach (naïve, structured, RAG variants)
- Context fill percentage (10%, 30%, 50%, 70%, 90%)
- Information position (start, middle, end)

**Dependent Variables:**
- **Quality:** Correctness (0-1), citation accuracy, completeness
- **Cost:** Token usage, API costs per query
- **Latency:** Time-to-first-token (TTFT), total response time
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

## 🧪 Experiment Suite

### Experiment 1: Needle in Multiple Haystacks
**Purpose:** Test retrieval quality vs. context stuffing under information overload

- **Corpus:** 500k-1M tokens of technical documentation (AWS, GCP, Azure API docs)
- **Tasks:** 50 questions (20 lookups, 20 synthesis, 10 contradiction detection)
- **What It Tests:** Multi-document reasoning, cross-referencing
- **Metrics:** Correctness, citation accuracy, cost per query

### Experiment 2: Context Pollution
**Purpose:** Measure robustness to irrelevant information

- **Base Corpus:** 50k tokens relevant content (company Q4 financial report)
- **Pollutant:** Add 50k → 950k tokens of plausible but irrelevant content
- **Tasks:** 20 questions strictly answerable from base corpus
- **What It Tests:** Resistance to distraction, precision
- **Metrics:** Accuracy vs pollution level, false positive rate

### Experiment 3: Multi-Turn Memory
**Purpose:** Test conversational context management

- **Scenario:** 10-turn customer support conversation
- **Memory Requirements:** Customer history (50k) + docs (200k) + conversation
- **Tasks:** Each turn requires integrating new + historical context
- **What It Tests:** Stateful memory, conversation coherence
- **Metrics:** Coherence score, fact retention, cumulative cost

### Experiment 4: Precision Retrieval
**Purpose:** Measure information extraction from dense, structured data

- **Corpus:** 100 academic papers (500k tokens total)
- **Tasks:** 30 fact lookups, 20 comparisons, 10 meta-analyses
- **What It Tests:** Structured navigation, citation accuracy
- **Metrics:** Precision@K, completeness, token efficiency

### Experiment 5: Cost-Latency Frontier
**Purpose:** Map Pareto frontier of quality vs. cost vs. latency

- **Analysis:** 3D optimization across all prior experiments
- **Output:** Dominant strategies (no approach beats them on all 3 metrics)
- **What It Tests:** Real-world deployment trade-offs
- **Metrics:** Efficiency score = Quality / (Cost × Latency)

## 🏗️ Project Structure

```
context-engineering-experiments/
├── README.md                      # This file
├── PROJECT_PLAN.md               # Detailed implementation roadmap
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

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# Google Cloud SDK (for Gemini API)
# Install from: https://cloud.google.com/sdk/docs/install
```

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd context-engineering-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_actual_api_key_here
```

**Get your Google AI API key:**
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and paste into `.env`

**Verify setup:**
```bash
# Check rate limits
python scripts/check_rate_limits.py

# Estimate experiment feasibility
python scripts/estimate_feasibility.py
```

### Running Experiments

```bash
# 1. Calibrate baseline (measure Gemini's intrinsic fill % degradation)
python scripts/run_calibration.py --output results/baseline_calibration.json

# 2. Run individual experiment
python scripts/run_experiment.py --experiment exp1_needle --conditions all

# 3. Run full suite (WARNING: This will make ~9,000 API calls)
python scripts/run_experiment.py --all --parallel 4

# 4. Analyze results
python scripts/analyze_results.py --input results/raw/ --output results/analysis/

# 5. Generate final report
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

**API Costs (Gemini Flash 2.5):**
- Input: $0.00001875 per 1k tokens
- Output: $0.000075 per 1k tokens

**Estimated Usage:**
- 9,000 queries × avg 500k input tokens = 4.5B input tokens
- 9,000 queries × avg 2k output tokens = 18M output tokens

**Total Cost:** ~$86 (input) + $1.35 (output) = **~$87**

**Recommended Budget:** $174 (2x buffer for reruns and debugging)

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
- FAISS/ChromaDB: Vector stores
- scipy: Statistical analysis
- plotly: Interactive visualizations

## 🤝 Contributing

This is a research project. If you'd like to:
- **Replicate:** Follow the setup instructions and PROJECT_PLAN.md
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

**Last Updated:** October 30, 2025  
**Version:** 1.0  
**Status:** Ready for Implementation  
**Estimated Completion:** 6 weeks from start