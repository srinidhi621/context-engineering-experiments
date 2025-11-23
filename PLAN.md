# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 10-12 Weeks (Realistic)  
**Start Date:** October 30, 2025  
**Budget:** $0 (Free Tier - gemini-2.0-flash-exp)  
**Team Size:** 1  

**Last Updated:** November 23, 2025  
**Status:** ✅ Phases 1A, 1B, 1C Complete - Ready for Pilot Live Run

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

**Pilot Phase:** 🔄 **In Progress (Phases 1A, 1B, 1C Complete)**
- ✅ Phase 1A: Infrastructure setup (Hugging Face verified, loaders built)
- ✅ Phase 1B: Data collection (Pilot corpus and question are complete)
- ✅ Phase 1C: Context assemblers (All assemblers implemented and tested)
- 🔄 Phase 1D: Minimal runner (Script implemented, dry run complete, ready for live run)
- ⏳ Phase 1E: Go/No-Go decision

**Experiments:** ❌ Not Started
- No questions generated yet
- Context assemblers not implemented (naive, structured)
- Experiments 1 & 2 not run

**Time Spent:** ~27 hours on infrastructure  
**Remaining:** ~10-12 weeks for pilot validation + experiments + analysis

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

## 🧪 PILOT PHASE: Validate Pipeline (Week 1)

**Duration:** 5-7 days  
**Goal:** End-to-end validation with 1 question before scaling up  
**API Calls:** 18 (1 question × 2 strategies × 3 fill levels × 3 reps)  
**Est. Cost:** $0 (free tier)

### Phase 1A: Infrastructure (Days 1-2) ✅ COMPLETE

**Task 1.1: Verify Hugging Face Hub access (10 min)** ✅ COMPLETE
```bash
# No authentication required for public model cards
# Optional: Get token from https://huggingface.co/settings/tokens
# Can add to .env: HUGGING_FACE_TOKEN=your_token_here (optional)
# Benefits of token: Higher rate limits, access to gated models
```

**Acceptance Criteria:** ✅ ALL MET
- [x] Can fetch model cards from Hugging Face Hub
- [x] Verified connectivity and API access
- [x] Tested with Llama 3.2-3B model card

**Task 1.2: Install additional dependencies (30 min)** ✅ COMPLETE
```bash
source venv/bin/activate
pip install huggingface_hub gutenbergpy
pip freeze > requirements.txt  # Update requirements
```

**Acceptance Criteria:** ✅ ALL MET
- [x] huggingface_hub>=0.36.0 installed
- [x] gutenbergpy>=0.3.4 installed
- [x] requirements.txt updated
- [x] All dependencies verified

**Task 1.3: Implement corpus loaders (Day 1 afternoon)** ✅ COMPLETE

File: `src/corpus/loaders.py`
```python
def load_hf_model_card(model_id, after_date="2024-08-01"):
    """Load model card from Hugging Face Hub
    
    Args:
        model_id: e.g., "meta-llama/Llama-3.2-3B"
        after_date: ISO date string, only fetch if modified after
    
    Returns:
        dict with 'content', 'url', 'last_modified', 'tokens'
    """
    # IMPLEMENTED

def load_hf_curated_models(after_date="2024-08-01", max_tokens=50000):
    """Load model cards from curated list of 60+ recent models
    
    Returns:
        list of dicts with model card content and metadata
    """
    # IMPLEMENTED - includes Llama 3.2/3.3, Qwen 2.5, Mistral, Phi 3.5, etc.

def load_gutenberg_books(book_ids, max_tokens=100000):
    """Load books from Project Gutenberg
    
    Args:
        book_ids: list of Gutenberg IDs (e.g., [1342, 84, 98])
        max_tokens: maximum tokens to load
    
    Returns:
        list of dicts with 'content', 'title', 'author', 'tokens'
    """
    # IMPLEMENTED
```

**Acceptance Criteria:** ✅ ALL MET
- [x] Can fetch Llama 3.2-3B model card via HF Hub
- [x] Can verify lastModified date is after 2024-08-01
- [x] Token count is accurate (within ±5% of tiktoken)
- [x] Returns structured dict with metadata
- [x] Tested: 50k tokens collected in 2-3 seconds

**Task 1.4: Implement tokenizer utilities (2 hours)** ✅ COMPLETE

File: `src/utils/tokenizer.py`
```python
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    # IMPLEMENTED

def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into chunks by token count (not word count)"""
    # IMPLEMENTED

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to exact token count"""
    # IMPLEMENTED
```

**Acceptance Criteria:** ✅ ALL MET
- [x] All functions implemented and tested
- [x] Chunking works correctly with overlap
- [x] Token counts match tiktoken exactly

### Phase 1B: Minimal Data Collection (Day 2)

**Status:** ✅ COMPLETE

**Task 2.1: Collect pilot corpus (2 hours)**

File: `scripts/collect_pilot_corpus.py` (NEW)
```python
#!/usr/bin/env python3
"""Collect minimal corpus for pilot testing"""

from src.corpus.loaders import load_hf_curated_models
import json

# Target: 10k tokens from recent model cards
corpus = load_hf_curated_models(
    after_date="2024-08-01",
    max_tokens=10000
)

# Save to data/raw/pilot/hf_model_cards.json
output_path = "data/raw/pilot/hf_model_cards.json"
with open(output_path, 'w') as f:
    json.dump(corpus, f, indent=2)

print(f"Collected {len(corpus)} model cards, {sum(d['tokens'] for d in corpus)} tokens")
```

**Acceptance Criteria:**
- [x] **Execution:** `python scripts/collect_pilot_corpus.py` exits with code 0.
- [x] **File Creation:** `ls data/raw/pilot/hf_model_cards.json` successfully lists the file.
- [x] **Token Count:** The command `python -c "import json; f = open('data/raw/pilot/hf_model_cards.json'); data = json.load(f); tokens = sum(d['tokens'] for d in data); assert 8000 <= tokens <= 12000, f'Token count {tokens} out of range'; print(f'OK: {tokens} tokens')"` exits with code 0.
- [x] **Date Verification:** The command `python -c "import json; from datetime import datetime; f = open('data/raw/pilot/hf_model_cards.json'); data = json.load(f); assert all(datetime.fromisoformat(d['lastModified'].replace('Z', '+00:00')).date() > datetime.fromisoformat('2024-08-01').date() for d in data), 'Found model card before cutoff date'"` exits with code 0.

**Task 2.2: Create 1 test question (1 hour)**

File: `data/questions/pilot_question_01.json` (NEW)
```json
{
  "experiment": "pilot",
  "question_id": "pilot_q001",
  "question": "What is the context window size of Llama 3.2-3B model?",
  "ground_truth": "The context window size is 128k tokens (131,072 tokens).",
  "difficulty": "simple_lookup",
  "required_docs": ["meta-llama/Llama-3.2-3B"],
  "evaluation_criteria": "Answer must state 128k or 131,072 tokens",
  "source_url": "https://huggingface.co/meta-llama/Llama-3.2-3B",
  "source_model": "meta-llama/Llama-3.2-3B",
  "keywords": ["Llama", "context window", "128k", "tokens"]
}
```

**Acceptance Criteria:**
- [x] **Validation Script:** `python scripts/validate_question_set.py data/questions/pilot_question_01.json` exits with code 0.
- [x] **Manual Check:** A human reviewer confirms the following:
  - [x] The question is unambiguously worded.
  - [x] The question can be answered using the corpus from Task 2.1.
  - [x] The `ground_truth` is factually correct and verifiable from the source URL.

## 🔜 Immediate Next Steps (Phase 1B/1C Bridge)

While the Gemini quota increase is pending, we can continue prepping the remainder of Phase 1B and unblock Phase 1C so the experiment run can start the moment quotas are lifted.

### Step 1: Pilot Evaluation + Reporting
- **Status:** ⏳ Planned
- **Task 4.2 Implementation:** Build `scripts/evaluate_pilot_manually.py` as originally scoped (load JSONL responses, compare to ground truth, emit scored JSONL + markdown summary).
- **Acceptance:** Script reports accuracy for each fill level, flags missing calls, and produces a `results/pilot_minimal_results_scored.jsonl`.
- **Dependency:** Requires at least one successful response per fill level; the script should gracefully handle skipped entries so we can run it immediately after the quota rerun.

### Step 2: Additional Context Assemblers
- **Status:** ⏳ Planned
- **Structured Context (`src/context_engineering/structured.py`):** Implement metadata headers + TOC scaffolding described in Phase 1C Task 3.1; include unit tests mirroring `tests/test_context_engineering.py`.
- **RAG Variants (`rag.py`, `advanced_rag.py`):** Wire tokenizer, retriever, and padding hooks now so only vector data needs to plug in later. Provide smoke tests with mocked retrievers.

### Step 3: Runner & CLI Integration
- **Status:** ⏳ Planned
- **Runner Skeletons:** Flesh out `scripts/run_experiment.py` and `scripts/run_calibration.py` with argument parsing, config loading, and placeholder experiment dispatch so pilot/Exp1 share infrastructure.
- **Monitoring Hooks:** Ensure runners tag `experiment_id`/`session_id` consistently and respect the `PER_MINUTE_TOKEN_LIMIT` environment variable just like the pilot runner.

### Step 4: Question Set Authoring Framework
- **Status:** ⏳ Planned
- **Schema Definition:** Create templates for `data/questions/exp1_questions.json` and `data/questions/exp2_questions.json` (fields, validation rules). **Status:** Templates staged as `.template.json` files ready to duplicate.
- **Validation Helper:** Add `scripts/validate_question_set.py` to lint question files (unique IDs, required-doc coverage) so we can iterate quickly once we start writing prompts. **Status:** Implemented; run `python scripts/validate_question_set.py data/questions/exp1_questions.template.json`.

### Interim Tasks While API Quota Upgrade is Pending

1. ⏳ Planned — **Collect corpora now**  
   - `python scripts/collect_exp1_corpus.py` → saves `data/raw/exp1/hf_model_cards.json` (~700k tokens).  
   - `python scripts/collect_padding_corpus.py` → saves `data/raw/padding/gutenberg_corpus.json` (~2M tokens).

2. ⏳ Planned — **Expand Experiment 1 question set**  
   - Continue adding entries to `data/questions/exp1_questions.json` until the full 50-question target (20 lookup / 20 synthesis / 10 contradiction) is reached.  
   - Validate after each batch: `python scripts/validate_question_set.py data/questions/exp1_questions.json --require-experiment exp1`.

3. ⏳ Planned — **Implement Experiment 1 runner (dry-run ready)**  
   - Build `scripts/run_experiment_1.py` with CLI arguments, data loading, context assembly hooks, per-minute throttle, and JSONL logging.  
   - Provide a `--dry-run` flag so the loop can be exercised without issuing API calls until quotas are raised.

4. ⏳ Planned — **Metrics/analyzer scaffolding**  
   - Define result schemas under `results/metrics/` and stub analyzer/report scripts (e.g., `scripts/analyze_exp1_results.py`) so scoring can begin immediately once real responses exist.

Completing these offline tasks keeps Experiment 1 on track; once the higher quota is live we can rerun the pilot and execute Exp1 without additional engineering delay.

Executing these four threads in parallel keeps Phase 1B moving despite quota delays and positions us to jump directly into Experiment 1 as soon as the pilot rerun succeeds.

### Phase 1C: Implement Context Assemblers (Days 3-4)

**Status:** ✅ COMPLETE

**Task 3.1: Implement naive context assembler (Day 3 morning)**

File: `src/context_engineering/naive.py`
```python
from typing import List, Dict
from src.utils.tokenizer import count_tokens_accurate, truncate_to_tokens

class NaiveContextAssembler:
    """Sequential document concatenation with no structure"""
    
    def assemble(self, documents: List[Dict], max_tokens: int) -> str:
        """
        Concatenate documents sequentially.
        
        Args:
            documents: List of dicts with 'content', 'title', 'url'
            max_tokens: Maximum tokens in output
        
        Returns:
            Assembled context string
        """
        # Simply concatenate with double newline separator
        context_parts = []
        total_tokens = 0
        
        for doc in documents:
            content = doc['content']
            tokens = count_tokens_accurate(content)
            
            if total_tokens + tokens <= max_tokens:
                context_parts.append(content)
                total_tokens += tokens
            else:
                # Truncate last document to fit
                remaining = max_tokens - total_tokens
                if remaining > 100:  # Only add if meaningful
                    truncated = truncate_to_tokens(content, remaining)
                    context_parts.append(truncated)
                break
        
        return "\n\n".join(context_parts)
```

**Unit Test:** `tests/test_naive_assembler.py`
```python
def test_naive_assembly_token_limit():
    docs = [
        {"content": "Doc 1 content...", "title": "Doc 1"},
        {"content": "Doc 2 content...", "title": "Doc 2"},
    ]
    assembler = NaiveContextAssembler()
    result = assembler.assemble(docs, max_tokens=1000)
    
    assert count_tokens_accurate(result) <= 1000
    assert "Doc 1 content" in result
```

**Acceptance Criteria:**
- [x] **Unit Test:** `pytest tests/test_context_engineering.py::test_naive_assembly` exits with code 0. (This test should cover token limits, document order, and edge cases).

**Task 3.2: Implement padding system (Day 3 afternoon)**

File: `src/corpus/padding.py`
```python
from typing import List, Dict
import random
from src.corpus.loaders import load_gutenberg_books
from src.utils.tokenizer import count_tokens_accurate, truncate_to_tokens

class PaddingGenerator:
    """Generate irrelevant padding content to reach target fill %"""
    
    def __init__(self):
        # Pre-load some Gutenberg books for padding
        # Book IDs: 1342 (Pride & Prejudice), 84 (Frankenstein), 
        #           98 (A Tale of Two Cities), 1661 (Sherlock Holmes)
        self.padding_books = load_gutenberg_books([1342, 84, 98, 1661])
        self.padding_text = "\n\n".join([b['content'] for b in self.padding_books])
    
    def generate_padding(self, target_tokens: int) -> str:
        """
        Generate padding text of target token count.
        
        Randomly samples from pre-loaded books to create padding.
        """
        if target_tokens <= 0:
            return ""
        
        # Sample random chunks from padding_text
        total_tokens = count_tokens_accurate(self.padding_text)
        if target_tokens >= total_tokens:
            # Need multiple copies
            copies = (target_tokens // total_tokens) + 1
            result = (self.padding_text + "\n\n") * copies
        else:
            # Sample from middle of text (more variety)
            start = random.randint(0, total_tokens - target_tokens)
            result = self.padding_text  # Simplified for now
        
        # Truncate to exact token count
        return truncate_to_tokens(result, target_tokens)
    
    def pad_to_fill_percentage(self, 
                               content: str, 
                               fill_pct: float,
                               max_context_tokens: int = 1_000_000) -> str:
        """
        Pad content to reach target fill percentage.
        
        Args:
            content: The actual relevant content
            fill_pct: Target fill percentage (0.1 to 0.9)
            max_context_tokens: Maximum context window size
        
        Returns:
            content + padding to reach fill_pct * max_context_tokens
        """
        target_total = int(max_context_tokens * fill_pct)
        content_tokens = count_tokens_accurate(content)
        
        if content_tokens >= target_total:
            # Content already exceeds target, truncate
            return truncate_to_tokens(content, target_total)
        
        padding_needed = target_total - content_tokens
        padding = self.generate_padding(padding_needed)
        
        # Interleave or append? For now, append
        result = content + "\n\n" + padding
        
        return result
```

**Unit Test:** `tests/test_padding.py`
```python
def test_padding_matches_fill_percentage():
    gen = PaddingGenerator()
    content = "Short content." * 100
    
    result = gen.pad_to_fill_percentage(
        content, 
        fill_pct=0.5, 
        max_context_tokens=10000
    )
    
    tokens = count_tokens_accurate(result)
    assert 4950 <= tokens <= 5050  # 5000 ± 1%
```

**Acceptance Criteria:**
- [x] **Unit Test:** `pytest tests/test_corpus.py::test_padding_generator` exits with code 0. (This test should cover token count accuracy, fill percentage correctness, and handling of edge cases).

**Task 3.3: Enhance RAG with padding (Day 4)**

File: `src/context_engineering/rag.py` (enhance existing)

Add method to RAGPipeline class:
```python
def assemble_context_with_padding(self, 
                                  retrieved_chunks: List[Dict],
                                  fill_pct: float,
                                  max_tokens: int = 1_000_000) -> str:
    """
    Assemble retrieved chunks and pad to match fill percentage.
    
    This is the KEY methodological control for H2.
    """
    from src.corpus.padding import PaddingGenerator
    
    # Assemble retrieved chunks
    context = self.assemble_context(retrieved_chunks, max_tokens)
    
    # Pad to match fill percentage
    padder = PaddingGenerator()
    padded_context = padder.pad_to_fill_percentage(
        context, fill_pct, max_tokens
    )
    
    return padded_context
```

**Acceptance Criteria:**
- [x] **Unit Test:** `pytest tests/test_context_engineering.py::test_rag_with_padding` exits with code 0. (This test should verify that RAG output can be padded to a target fill % accurately).

### Phase 1D: Implement Minimal Runner (Day 5)

**Status:** 🔄 In Progress (Dry Run Complete)

**Task 4.1: Create pilot runner script**

**Status:** ✅ COMPLETE

File: `scripts/run_minimal_pilot.py` (NEW)
```python
#!/usr/bin/env python3
"""
Minimal pilot: Run 1 question with 2 strategies at 3 fill levels.
Total: 1 × 2 × 3 × 3 reps = 18 API calls
"""

import json
import time
from pathlib import Path
from src.models.gemini_client import GeminiClient
from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_minimal_pilot():
    # Load corpus
    with open("data/raw/pilot/hf_model_cards.json") as f:
        corpus = json.load(f)
    
    # Load question
    with open("data/questions/pilot_question_01.json") as f:
        question = json.load(f)
    
    # Initialize client
    client = GeminiClient()
    
    # Initialize assemblers
    naive = NaiveContextAssembler()
    rag = RAGPipeline()
    
    # Index corpus for RAG
    documents = [doc['content'] for doc in corpus]
    rag.chunk_documents(documents, chunk_size=512, overlap=50)
    rag.index_chunks()
    
    # Configuration
    strategies = ["naive", "rag"]
    fill_levels = [0.3, 0.5, 0.7]
    repetitions = 3
    max_tokens = 1_000_000
    
    results = []
    
    for strategy in strategies:
        for fill_pct in fill_levels:
            for rep in range(repetitions):
                logger.info(f"Running {strategy} at {fill_pct*100}% fill, rep {rep+1}")
                
                # Assemble context
                if strategy == "naive":
                    target_tokens = int(max_tokens * fill_pct)
                    context = naive.assemble(corpus, target_tokens)
                else:  # RAG
                    retrieved = rag.retrieve(question['question'], top_k=5)
                    context = rag.assemble_context_with_padding(
                        retrieved, fill_pct, max_tokens
                    )
                
                # Build prompt
                prompt = f"""Answer the following question based on the provided documentation.

Question: {question['question']}

Documentation:
{context}

Answer:"""
                
                # Make API call
                try:
                    start_time = time.time()
                    response = client.generate_content(
                        prompt,
                        temperature=0.0,
                        experiment_id="pilot",
                        session_id=f"{strategy}_fill{int(fill_pct*100)}_rep{rep}"
                    )
                    latency = time.time() - start_time
                    
                    # Record result
                    result = {
                        "question_id": question['question_id'],
                        "strategy": strategy,
                        "fill_pct": fill_pct,
                        "repetition": rep,
                        "response": response['text'],
                        "tokens_input": response['tokens_input'],
                        "tokens_output": response['tokens_output'],
                        "latency": latency,
                        "cost": response['cost'],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    results.append(result)
                    
                    logger.info(f"Success: {response['tokens_input']} input tokens, "
                               f"{response['tokens_output']} output tokens, "
                               f"{latency:.2f}s latency")
                    
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    # Continue with next call
                
                # Small delay to be polite to API
                time.sleep(2)
    
    # Save results
    output_path = Path("results/pilot_minimal_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Pilot complete! Saved {len(results)} results to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PILOT SUMMARY")
    print("="*60)
    print(f"Total API calls: {len(results)}")
    print(f"Total tokens: {sum(r['tokens_input'] for r in results):,}")
    print(f"Total cost: ${sum(r['cost'] for r in results):.4f}")
    print(f"Results saved to: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_minimal_pilot()
```

**Acceptance Criteria:**
- [x] **Dry Run:** `python scripts/run_minimal_pilot.py --dry-run` exits with code 0 and logs that it would make 18 API calls.
- [ ] **File Creation:** After a real run, `ls results/pilot_minimal_results.jsonl` successfully lists the file.
- [ ] **Result Count:** `wc -l results/pilot_minimal_results.jsonl` reports 18.
- [ ] **Resumption:** (Manual Test) After a partial run, deleting the last line of the results file and re-running the script should only execute the remaining API calls.

**Task 4.2: Manual evaluation of results (Day 5 evening)**

**Status:** ⏳ Planned

**Acceptance Criteria:**
- [ ] **Execution:** `python scripts/evaluate_pilot_manually.py` runs and prompts for user input without crashing.
- [ ] **File Creation:** After the script is run, `ls results/pilot_minimal_results_scored.jsonl` successfully lists the file.
- [ ] **Content Verification:** `grep '"correct"' results/pilot_minimal_results_scored.jsonl | wc -l` reports 18.

### Phase 1E: Review and Go/No-Go Decision (Day 6-7)

**Task 5.1: Analyze pilot results**
- [ ] Check if responses make sense
- [ ] Verify token counting is accurate
- [ ] Confirm fill % is correctly implemented
- [ ] Identify any bugs or issues

**Task 5.1a: Scaffold Analysis Pipeline**
- **Status:** ⏳ Planned
- **Action:** Use the `results/pilot_minimal_results.jsonl` as input to build the initial versions of the analysis and visualization scripts (`scripts/analyze_results.py`, etc.).
- **Goal:** This de-risks the final analysis phase. By building the pipeline early, you can generate key metrics and plots immediately after the larger experiments finish, rather than treating analysis as a separate, final step.
- **Acceptance:**
  - [ ] **Analysis Script:** `python scripts/analyze_results.py --input results/pilot_minimal_results_scored.jsonl --output-file results/analysis/pilot_summary.md` exits with code 0.
  - [ ] **Summary Verification:** `ls results/analysis/pilot_summary.md` successfully lists the file.
  - [ ] **Visualization Script:** `python scripts/generate_visualizations.py --input results/pilot_minimal_results_scored.jsonl --output-dir results/visualizations/pilot` exits with code 0.
  - [ ] **Plot Verification:** `ls results/visualizations/pilot/*.png | wc -l` reports a count of 1 or more.

**Task 5.2: Go/No-Go Decision Checklist**

✅ **GO** if:
- [ ] All 18 API calls completed successfully
- [ ] Responses are relevant to question
- [ ] Token counts are accurate (±5%)
- [ ] Fill % is correct (±5%)
- [ ] No major bugs found
- [ ] Cost is $0 (free tier)

❌ **NO-GO** if:
- [ ] More than 2 API calls failed
- [ ] Responses are gibberish
- [ ] Token counts are wildly off (>10% error)
- [ ] Fill % is not working
- [ ] Major bugs that need fixing

**If NO-GO:** Fix issues before proceeding to full pilot.

**Success Criteria:**
- Pipeline runs end-to-end without crashes
- Results are reasonable (not random garbage)
- Token counting is accurate (±5%)
- Fill % matching works (±5%)
- Decision documented in `results/pilot_go_nogo.md`

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

---

**Last Updated:** November 3, 2025  
**Version:** 2.1 (Phase 1A Complete)  
**Status:** Phase 1A Complete - Ready for Phase 1B (Data Collection)
