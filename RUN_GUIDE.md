# 🚀 How to Run the Experimental Harness

## 📊 Current Status

**Phase 1A: Complete** ✅ (Infrastructure + Data Loaders)  
**Phase 1B-1E: Not Started** ⏳ (Pilot experiments need to be built)

---

## 🔧 Environment Setup

### 1. **Activate Virtual Environment** (Required Every Time)

```bash
cd /Users/Srinidhi/my_projects/context_engineering_experiments
source venv/bin/activate

# You'll see (venv) in your prompt when activated
```

**OR use the convenience script:**
```bash
cd /Users/Srinidhi/my_projects/context_engineering_experiments
source scripts/activate.sh
```

### 2. **Verify Environment Variables**

Your `.env` file should have:
```bash
GOOGLE_API_KEY=your_actual_key_here
```

**Check it's loaded:**
```bash
python -c "from src.config import api_config; print('✓ API key configured' if api_config.google_api_key else '✗ Missing')"
```

---

## ✅ What You CAN Run Right Now

### **Test 1: API Integration Test** (Recommended First)

Tests the complete stack end-to-end.

```bash
# From project root with venv activated:
python scripts/test_api_integration.py
```

**What it does:**
- ✅ Verifies GOOGLE_API_KEY is set
- ✅ Initializes Gemini client
- ✅ Tests rate limiter
- ✅ Tests cost monitor
- ✅ Makes 1 real API call (costs ~$0.00)
- ✅ Verifies experiment tracking

**Expected output:**
```
🧪 API INTEGRATION TEST
========================================
✅ Client initialized
✅ Model: models/gemini-2.0-flash-exp
✅ Rate Limiter: Active
✅ Monitor configured
✅ API call successful
✅ Experiment tracking working
✅ ALL TESTS PASSED
```

---

### **Test 2: Verify API Key**

Quick check that your API key works.

```bash
python scripts/verify_api_key.py
```

---

### **Test 3: Check Monitoring Status**

View current API usage and costs.

```bash
# Quick view
python scripts/monitor_costs.py

# Detailed report
python scripts/generate_cost_report.py

# Check rate limits
python scripts/check_rate_limits.py

# Estimate experiment timeline
python scripts/estimate_feasibility.py
```

---

### **Test 4: Test Data Loaders**

Verify corpus loading works.

```bash
# Test Hugging Face loader
python -c "
from src.corpus.loaders import load_hf_model_card
doc = load_hf_model_card('meta-llama/Llama-3.2-3B')
print(f'✓ Loaded {doc[\"tokens\"]:,} tokens from {doc[\"model_id\"]}')
"

# Test curated models collection
python -c "
from src.corpus.loaders import load_hf_curated_models
docs = load_hf_curated_models(max_tokens=10000)
total = sum(d['tokens'] for d in docs)
print(f'✓ Loaded {len(docs)} models, {total:,} tokens total')
"

# Test Gutenberg loader
python -c "
from src.corpus.loaders import load_gutenberg_books
books = load_gutenberg_books([1342], max_tokens=10000)
print(f'✓ Loaded {books[0][\"title\"]} by {books[0][\"author\"]}')
print(f'  {books[0][\"tokens\"]:,} tokens')
"
```

---

## ❌ What You CANNOT Run Yet

These need to be implemented (Phase 1B-1E):

```bash
# NOT YET CREATED:
python scripts/run_pilot.py              # Phase 1D
python scripts/run_experiment_1.py       # Phase 2
python scripts/run_experiment_2.py       # Phase 3
python scripts/collect_pilot_corpus.py   # Phase 1B
python scripts/evaluate_pilot_manually.py # Phase 1D

# PLACEHOLDER ONLY (empty TODOs):
python scripts/run_experiment.py         # Empty
python scripts/run_calibration.py        # Empty
```

---

## 📋 Complete Workflow (When Ready)

**Phase 1B: Data Collection**
```bash
# Step 1: Collect pilot corpus (to be created)
python scripts/collect_pilot_corpus.py
# → Saves to: data/raw/pilot/hf_model_cards.json

# Step 2: Create pilot question (manual)
# → Create: data/questions/pilot_question_01.json
```

**Phase 1C: Build Context Assemblers**
```bash
# Implement these files:
# - src/context_engineering/naive.py
# - src/corpus/padding.py
# - src/context_engineering/rag.py (enhance with padding)
```

**Phase 1D: Run Pilot**
```bash
# Step 1: Run minimal pilot (to be created)
python scripts/run_minimal_pilot.py
# → Makes 18 API calls
# → Saves to: results/pilot_minimal_results.jsonl

# Step 2: Evaluate results (to be created)
python scripts/evaluate_pilot_manually.py
# → Manual scoring
# → Saves to: results/pilot_minimal_results_scored.jsonl
```

**Phase 2+: Full Experiments**
```bash
# After pilot succeeds:
python scripts/run_experiment_1.py       # 3,000 calls (~2 days)
python scripts/run_experiment_2.py       # 1,200 calls (~1 day)
python scripts/analyze_results.py
python scripts/generate_report.py
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
# Make sure you're in the project root:
cd /Users/Srinidhi/my_projects/context_engineering_experiments

# Make sure venv is activated:
source venv/bin/activate

# Verify installation:
pip install -e .
```

### "GOOGLE_API_KEY environment variable not set"
```bash
# Check .env file exists:
ls -la .env

# Edit it:
nano .env

# Add:
GOOGLE_API_KEY=your_actual_key_here

# Get key from: https://aistudio.google.com/app/apikey
```

### "ModuleNotFoundError: No module named 'gutenbergpy'"
```bash
# Reinstall dependencies:
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎯 Next Steps

**To start running experiments:**

1. ✅ **Done**: Phase 1A (infrastructure + data loaders)
2. ⏳ **Next**: Implement Phase 1B scripts:
   - `scripts/collect_pilot_corpus.py`
   - Create `data/questions/pilot_question_01.json`
3. ⏳ **Then**: Implement Phase 1C assemblers:
   - `src/context_engineering/naive.py`
   - `src/corpus/padding.py`
4. ⏳ **Then**: Implement Phase 1D runner:
   - `scripts/run_minimal_pilot.py`
   - `scripts/evaluate_pilot_manually.py`

**See `PLAN.md` lines 170-620 for detailed implementation tasks.**

---

## 📚 Reference

- **Full Plan**: `PLAN.md`
- **Session Notes**: `SESSION_NOTES.md`
- **Infrastructure**: All working (API client, monitoring, loaders, tokenizer)
- **Experiments**: Ready to build (infrastructure complete)

---

**Last Updated**: November 3, 2025  
**Status**: Infrastructure ready, experiments need implementation
