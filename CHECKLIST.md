# Pre-flight Checklist

Before running any large-scale experiment (like Experiment 1), run this checklist to ensure the environment, code, and configuration are ready.

## 1. Environment Verification
- [ ] **Python Version:** Ensure you are running Python 3.10+ (ideally 3.13).
  ```bash
  python --version
  ```
- [ ] **Dependencies:** Dependencies must be installed.
  ```bash
  pip check
  ```
- [ ] **API Keys:** Ensure `.env` has a valid `GOOGLE_API_KEY`.
  ```bash
  grep GOOGLE_API_KEY .env
  ```

## 2. Code Quality & Tests
Run the full test suite to catch regressions.
```bash
# Run all tests
pytest tests/

# Run linting (if configured, e.g. ruff)
# ruff check .
```

## 3. Data Readiness
Verify that input data exists and is valid.
- [ ] **Experiment 1 Corpus:** `data/raw/exp1/hf_model_cards.json` (>500KB)
- [ ] **Padding Corpus:** `data/raw/padding/gutenberg_corpus.json` (>5MB)
- [ ] **Questions:** `data/questions/exp1_questions.json` (50 items)

## 4. Dry Run
Perform a dry run to validate the orchestration logic without spending money.
```bash
python scripts/run_experiment.py --experiment exp1 --dry-run
```
**Success Criteria:**
- Output shows `[DRY RUN] Would execute...` for 3000 runs.
- No crashes or tracebacks.
- `results/raw/exp1_results.jsonl` is populated with dry-run entries (optional).

## 5. Feasibility Check
Run the feasibility estimator to confirm budget and time.
```bash
python scripts/estimate_feasibility.py
```

## 6. Clean Start (Optional)
If you want a fresh run, archive or move existing results.
```bash
# mv results/raw/exp1_results.jsonl results/archive/exp1_results_old.jsonl
# rm results/cache/* (Only if you want to rebuild indexes)
```
