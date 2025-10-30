# Project Implementation Plan

**Project:** Context Engineering for LLMs - Experimental Suite  
**Timeline:** 6 Weeks  
**Start Date:** October 30, 2025  
**Expected Completion:** December 11, 2025  
**Budget:** $174 (API costs)  
**Team Size:** 1  

**Last Updated:** October 30, 2025

---

## 📊 Current Status

### ✅ Completed (Scaffolding Phase)

- [x] Created complete directory structure (8 main directories)
- [x] Organized all files into proper locations
- [x] Created all `__init__.py` files (8 packages)
- [x] Implemented rate limiting system (`src/utils/rate_limiter.py`)
- [x] Implemented token counting utilities (`src/utils/tokenizer.py`)
- [x] Implemented structured logging (`src/utils/logging.py`)
- [x] Implemented statistical analysis functions (`src/utils/stats.py`)
- [x] Created configuration management (`src/config.py`)
- [x] Set up `requirements.txt` (13 dependencies)
- [x] Created `setup.py` for package installation
- [x] Created `.env.example` for API key configuration
- [x] Created `.gitignore` with comprehensive rules
- [x] Created `.gitkeep` files for empty directories
- [x] Created skeleton files for all modules (29 Python files)
- [x] Made all scripts executable
- [x] Documented README.md with comprehensive project information
- [x] Documented PLAN.md with 6-week implementation timeline

### 🔨 In Progress (Git Setup - Personal Account)

- [ ] Configure Git with personal account credentials
- [ ] Set up repository-specific Git config (personal email/name)
- [ ] Initialize Git repository
- [ ] Create `.gitignore` verification
- [ ] Make initial commit
- [ ] Create GitHub repository (personal account)
- [ ] Add remote and push

### 🔨 Next Up (Week 1 - Foundation & Corpus)

- [ ] Set up Python virtual environment
- [ ] Install all dependencies
- [ ] Configure API access (Google AI API key)
- [ ] Download API documentation corpus (AWS, GCP, Azure)
- [ ] Download SEC financial reports
- [ ] Download arXiv academic papers
- [ ] Download Wikipedia articles for padding
- [ ] Generate 50 questions for Experiment 1
- [ ] Generate 20 questions for Experiment 2
- [ ] Generate 10 scenarios for Experiment 3
- [ ] Generate 60 questions for Experiment 4

---

## 📅 Timeline Overview

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1: Foundation & Corpus | Week 1 | All corpora collected, 200 questions ready |
| Phase 2: Context Engineering | Week 2 | All 4 strategies implemented and tested |
| Phase 3: Experiment Execution (Part 1) | Week 3 | Experiments 1-2 completed (4,200 API calls) |
| Phase 4: Experiment Execution (Part 2) | Week 4 | Experiments 3-5 completed (4,800 API calls) |
| Phase 5: Evaluation & Analysis | Week 5 | All metrics computed, hypotheses tested |
| Phase 6: Reporting & Documentation | Week 6 | Final report published, code documented |

---

## 📅 Phase 1: Foundation & Corpus Preparation (Week 1)

### Goals
- ✅ Set up project infrastructure
- ✅ Collect and preprocess corpus data
- ✅ Create evaluation question sets with ground truth
- ✅ Establish development environment

### Day 1-2: Project Setup & Environment

**Tasks:**
- [ ] Initialize Git repository
  ```bash
  git init
  git add .
  git commit -m "Initial project structure"
  ```
- [ ] Set up Python virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Configure API access
  - Get Google AI API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
  - Set up `.env` file with credentials
  - Test API connection
- [ ] Implement core utilities
  - `src/utils/tokenizer.py` - Token counting (using tiktoken)
  - `src/utils/logging.py` - Structured logging
  - `src/models/gemini_client.py` - API wrapper with rate limiting
- [ ] Write unit tests for utilities
  - Test token counting accuracy
  - Test API client retry logic
  - Test rate limiting

**Deliverables:**
- ✓ Working development environment
- ✓ API client with rate limiting (60 RPM)
- ✓ Token counting utility
- ✓ Structured logging system
- ✓ Initial tests passing

**Time Estimate:** 16 hours

---

### Day 3-4: Corpus Collection

#### Experiment 1: API Documentation (500k-1M tokens)

**Sources:**
1. **AWS Documentation**
   - Lambda, API Gateway, DynamoDB, S3
   - Download from: https://docs.aws.amazon.com
   - Method: Web scraping or use AWS docs dataset
   
2. **GCP Documentation**
   - Cloud Functions, Cloud Storage, Firestore
   - Download from: https://cloud.google.com/docs
   
3. **Azure Documentation**
   - Azure Functions, Blob Storage, Cosmos DB
   - Download from: https://docs.microsoft.com/azure

**Tasks:**
- [ ] Scrape/download API docs (use BeautifulSoup or Selenium)
- [ ] Clean HTML → plain text conversion
- [ ] Normalize formatting (remove excessive whitespace, standardize headers)
- [ ] Extract metadata (service name, category, last updated)
- [ ] Verify total token count reaches 500k-1M
- [ ] Save as organized files: `data/raw/api_docs/{service}/{document}.txt`

**Format:**
```
data/raw/api_docs/
├── aws/
│   ├── lambda_overview.txt
│   ├── lambda_api.txt
│   └── ...
├── gcp/
│   └── ...
└── azure/
    └── ...
```

#### Experiment 2: Financial Reports (50k base + 1M pollution)

**Sources:**
1. **Base Corpus:** SEC EDGAR filings
   - 5-10 company Q4 reports (2023-2024)
   - Focus: Tech companies (Apple, Google, Microsoft, Amazon, Meta)
   - Download from: https://www.sec.gov/edgar
   
2. **Pollution Corpus:**
   - Additional 20-30 reports from different companies/quarters
   - Mix of industries to ensure diversity

**Tasks:**
- [ ] Download 10-K and 10-Q filings from SEC EDGAR
- [ ] Extract text from filings (use `python-edgar` or manual download)
- [ ] Parse sections (focus on Q4 summaries, financial statements)
- [ ] Label each report: company, quarter, year, industry
- [ ] Separate base corpus (5 reports × 50k tokens each)
- [ ] Create pollution pool (1M+ tokens)

#### Experiment 4: Academic Papers (500k tokens)

**Sources:**
1. **arXiv Papers:** CS/AI domain (Machine Learning, NLP)
   - Download from: https://arxiv.org/
   - Target: 100 papers × 5k tokens each = 500k total

**Tasks:**
- [ ] Download 100 recent papers from arXiv (2022-2024)
- [ ] Convert PDFs to text (use `pdfplumber` or `PyMuPDF`)
- [ ] Parse structure: Title, Abstract, Introduction, Methods, Results, Conclusion
- [ ] Clean conversion artifacts (headers, footers, references)
- [ ] Extract metadata: authors, year, venue, citations
- [ ] Verify token count per paper (aim for 4k-6k each)

#### Padding Corpus (2M+ tokens)

**Sources:** Wikipedia articles from unrelated domains

**Categories to include:**
- History (World Wars, ancient civilizations)
- Geography (countries, cities, landmarks)
- Literature (classic novels, authors)
- Arts (painting, music, theater)
- Sports (Olympics, major leagues)
- Biology (animals, plants, ecosystems)

**Tasks:**
- [ ] Use Wikipedia API to download 500-1000 articles
  ```python
  import wikipedia
  articles = wikipedia.random(1000)
  ```
- [ ] Filter to ensure no overlap with test domains
- [ ] Clean wiki markup → plain text
- [ ] Chunk into ~2k token segments
- [ ] Total: 2M+ tokens reusable across experiments

**Deliverables:**
- ✓ `data/raw/api_docs/` - 500k-1M tokens
- ✓ `data/raw/financial_reports/` - 1M+ tokens
- ✓ `data/raw/academic_papers/` - 500k tokens
- ✓ `data/raw/padding_corpus/` - 2M+ tokens
- ✓ `data/corpus_manifest.json` - Metadata catalog

**Scripts to write:**
- `src/corpus/downloaders/scrape_aws_docs.py`
- `src/corpus/downloaders/download_sec_filings.py`
- `src/corpus/downloaders/download_arxiv_papers.py`
- `src/corpus/downloaders/download_wikipedia.py`
- `src/corpus/loaders.py` - Generic corpus loader

**Time Estimate:** 16 hours

---

### Day 5-7: Question Generation & Ground Truth

**Goal:** Create 200 total questions across 4 experiments (50 each)

#### Question Design Principles

1. **Difficulty Distribution:**
   - 40% Simple lookups (factual recall)
   - 40% Synthesis (combine multiple sources)
   - 20% Complex (multi-hop reasoning, contradictions)

2. **Answer Types:**
   - Factual (numbers, dates, names)
   - Comparative (which is better/faster/cheaper)
   - Analytical (why/how questions)

3. **Ground Truth:**
   - Human-written reference answers
   - Explicit evaluation criteria
   - Source citations (which documents contain the answer)

#### Experiment 1 Questions (API Docs)

**Examples:**

*Simple Lookup:*
```json
{
  "id": "exp1_q001",
  "question": "What is the default timeout for AWS Lambda functions?",
  "ground_truth": "3 seconds (configurable up to 15 minutes)",
  "difficulty": "simple_lookup",
  "required_docs": ["aws/lambda_configuration.txt"],
  "evaluation_criteria": "Must mention 3 seconds and maximum of 15 minutes"
}
```

*Synthesis:*
```json
{
  "id": "exp1_q015",
  "question": "Compare the authentication methods for AWS Lambda, GCP Cloud Functions, and Azure Functions.",
  "ground_truth": "AWS uses IAM roles, GCP uses service accounts, Azure uses managed identities...",
  "difficulty": "synthesis",
  "required_docs": ["aws/lambda_auth.txt", "gcp/functions_auth.txt", "azure/functions_auth.txt"],
  "evaluation_criteria": "Must mention all three methods and their key differences"
}
```

*Complex:*
```json
{
  "id": "exp1_q030",
  "question": "The AWS docs say Lambda supports 1000 concurrent executions, but the pricing page mentions soft limits. Which is correct?",
  "ground_truth": "Both are correct. 1000 is the default soft limit per region, which can be increased...",
  "difficulty": "complex_contradiction",
  "required_docs": ["aws/lambda_limits.txt", "aws/lambda_pricing.txt"],
  "evaluation_criteria": "Must resolve the apparent contradiction and explain soft vs hard limits"
}
```

**Tasks:**
- [ ] Generate 50 questions with variety
  - 20 simple lookups
  - 20 synthesis questions
  - 10 complex/contradiction questions
- [ ] Write ground truth answers (100-200 words each)
- [ ] Identify required source documents for each
- [ ] Define evaluation criteria (what makes an answer correct?)

#### Experiment 2 Questions (Financial Reports)

**Focus:** Questions answerable ONLY from the base corpus

**Examples:**
```json
{
  "id": "exp2_q001",
  "question": "What was Apple's Q4 2023 revenue?",
  "ground_truth": "$89.5 billion",
  "difficulty": "simple_lookup",
  "required_docs": ["financial_reports/apple_q4_2023.txt"],
  "base_corpus_only": true
}
```

**Tasks:**
- [ ] Generate 20 questions strictly from base corpus
- [ ] Ensure answers are NOT in pollution documents
- [ ] Write clear ground truth with numbers/facts
- [ ] Mark which company/quarter each question targets

#### Experiment 3 Questions (Multi-Turn Memory)

**Format:** Conversational scenarios with 10 turns each

**Example Scenario:**
```json
{
  "id": "exp3_scenario_001",
  "scenario": "Customer contacting support about billing issue",
  "turns": [
    {
      "turn": 1,
      "customer_says": "I was charged twice for my subscription",
      "required_context": ["customer_history", "billing_docs"],
      "expected_actions": ["Verify charge history", "Check subscription status"]
    },
    {
      "turn": 2,
      "customer_says": "Yes, on October 15 and October 16",
      "required_context": ["previous_turn", "billing_docs"],
      "expected_actions": ["Locate charges", "Explain if legitimate or error"]
    }
    // ... 8 more turns
  ]
}
```

**Tasks:**
- [ ] Create 10 multi-turn scenarios (5 turns each for simplicity)
- [ ] Define required context at each turn
- [ ] Write expected responses/actions
- [ ] Include memory retention questions ("What did I say in turn 2?")

#### Experiment 4 Questions (Academic Papers)

**Examples:**
```json
{
  "id": "exp4_q001",
  "question": "What was the sample size in the BERT paper?",
  "ground_truth": "3.3 billion words (Wikipedia + BookCorpus)",
  "difficulty": "simple_lookup",
  "required_docs": ["academic_papers/bert_2018.txt"],
  "section": "Methods"
}
```

**Tasks:**
- [ ] Generate 60 questions
  - 30 fact lookups (sample size, accuracy, parameters)
  - 20 comparisons (which papers used technique X?)
  - 10 meta-analyses (summarize findings across papers)
- [ ] Include section hints (Abstract, Methods, Results)
- [ ] Write detailed ground truth with citations

**Format for All Questions:**
```json
{
  "experiment": "exp1_needle",
  "questions": [
    {
      "id": "exp1_q001",
      "question": "...",
      "ground_truth": "...",
      "difficulty": "simple_lookup | synthesis | complex",
      "required_docs": ["doc1.txt", "doc2.txt"],
      "evaluation_criteria": "...",
      "metadata": {
        "domain": "aws_lambda",
        "answer_type": "factual | comparative | analytical"
      }
    }
  ]
}
```

**Deliverables:**
- ✓ `data/questions/exp1_questions.json` (50 questions)
- ✓ `data/questions/exp2_questions.json` (20 questions)
- ✓ `data/questions/exp3_scenarios.json` (10 scenarios × 5 turns)
- ✓ `data/questions/exp4_questions.json` (60 questions)
- ✓ Total: 200+ evaluation items

**Time Estimate:** 24 hours (most time-consuming phase)

---

## 🔧 Phase 2: Context Engineering Implementation (Week 2)

### Goals
- ✅ Implement all 4 context assembly strategies
- ✅ Build fill % padding mechanism
- ✅ Create retrieval pipelines (RAG)
- ✅ Write comprehensive tests

---

### Day 1-2: Naïve & Structured Context Assembly

#### Task 1: Naïve Context Assembler

**File:** `src/context_engineering/naive.py`

**Implementation:**
```python
class NaiveContextAssembler:
    def assemble(self, documents: List[str], target_tokens: int, 
                 fill_pct: float, padding_corpus: List[str]) -> str:
        """
        Sequential document concatenation with padding
        
        Strategy:
        1. Concatenate docs sequentially until relevant content exhausted
        2. Add padding to reach target_tokens (based on fill_pct)
        3. No structure, no metadata, simple truncation
        """
        pass
```

**Key Features:**
- Sequential concatenation
- Token-aware truncation
- Fill % padding support
- No optimization

**Tests:**
- [ ] Test token counting accuracy
- [ ] Test padding reaches exact target
- [ ] Test truncation at boundaries
- [ ] Test with various fill percentages

#### Task 2: Structured Context Assembler

**File:** `src/context_engineering/structured.py`

**Implementation:**
```python
class StructuredContextAssembler:
    def assemble(self, documents: List[Dict], target_tokens: int,
                 fill_pct: float, padding_corpus: List[str]) -> str:
        """
        Engineered context with hierarchical structure
        
        Strategy:
        1. Generate table of contents
        2. Add metadata tags to each document
        3. Include navigation instructions
        4. Organize hierarchically
        5. Add padding if needed
        """
        pass
    
    def _generate_toc(self, documents: List[Dict]) -> str:
        """Create hierarchical TOC"""
        pass
    
    def _add_metadata(self, doc: Dict, index: int) -> str:
        """Wrap document in structured tags"""
        pass
```

**Document Structure:**
```xml
<document id="doc_0">
  <metadata>
    <title>AWS Lambda Overview</title>
    <source>AWS Documentation</source>
    <topic>Serverless Computing</topic>
  </metadata>
  <content>
    [Document content here]
  </content>
</document>
```

**Tests:**
- [ ] Test TOC generation
- [ ] Test metadata tagging
- [ ] Test hierarchical organization
- [ ] Verify token overhead is <10%

**Time Estimate:** 16 hours

---

### Day 3-5: RAG Pipeline Implementation

#### Task 3: Basic RAG Pipeline

**File:** `src/context_engineering/rag.py`

**Components:**

1. **Chunking Strategy:**
```python
def chunk_documents(self, documents: List[str], 
                   chunk_size: int = 512,
                   overlap: int = 50) -> List[Dict]:
    """
    Semantic chunking with overlap
    
    Returns:
        List of dicts with 'text', 'doc_id', 'chunk_id', 'metadata'
    """
    pass
```

2. **Embedding & Indexing:**
```python
def index_chunks(self, chunks: List[Dict]):
    """
    Generate embeddings and build vector index
    
    Uses:
    - Google text-embedding-004 (768 dimensions)
    - FAISS or ChromaDB for vector store
    """
    pass
```

3. **Retrieval:**
```python
def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
    """
    Vector similarity search
    
    Returns:
        Top-k chunks with scores
    """
    pass
```

4. **Reranking (Optional):**
```python
def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Cross-encoder reranking for better precision
    
    Optional: Use Cohere rerank API or local model
    """
    pass
```

5. **Context Assembly:**
```python
def assemble_context(self, retrieved: List[Dict], 
                    max_tokens: int = 128_000) -> str:
    """
    Assemble retrieved chunks into context
    
    Strategy:
    - Order by relevance score
    - Add contextual headers
    - Track token budget
    """
    pass
```

**Dependencies to Add:**
```bash
pip install faiss-cpu  # or faiss-gpu
pip install chromadb
pip install sentence-transformers  # for local embeddings if needed
```

**Tests:**
- [ ] Test chunking preserves semantic boundaries
- [ ] Test embedding generation
- [ ] Test retrieval returns relevant chunks
- [ ] Test token budget enforcement in assembly
- [ ] End-to-end RAG pipeline test

**Time Estimate:** 24 hours

---

### Day 6-7: Advanced RAG & Integration

#### Task 4: Advanced RAG Pipeline

**File:** `src/context_engineering/advanced_rag.py`

**Advanced Features:**

1. **Hybrid Search (Dense + Sparse):**
```python
class AdvancedRAGPipeline(RAGPipeline):
    def retrieve(self, query: str, top_k: int = 10):
        # Vector search (dense)
        vector_results = self._vector_search(query, top_k*2)
        
        # BM25 search (sparse)
        bm25_results = self._bm25_search(query, top_k*2)
        
        # Reciprocal Rank Fusion
        fused = self._rrf_fusion(vector_results, bm25_results)
        
        return fused[:top_k]
```

2. **Query Decomposition:**
```python
def decompose_query(self, complex_query: str) -> List[str]:
    """
    Break multi-hop questions into sub-queries
    
    Example:
    "Compare AWS and GCP authentication" →
    ["What is AWS authentication?", "What is GCP authentication?"]
    """
    pass
```

3. **Iterative Retrieval:**
```python
def iterative_retrieve(self, query: str, max_iterations: int = 3):
    """
    Retrieve → Generate → Reflect → Retrieve again
    
    Useful for multi-hop questions
    """
    pass
```

**Integration Testing:**
- [ ] Test all 4 assemblers on same corpus
- [ ] Verify token budgets are respected
- [ ] Compare output formats
- [ ] Benchmark assembly time

**Deliverables:**
- ✓ `src/context_engineering/naive.py`
- ✓ `src/context_engineering/structured.py`
- ✓ `src/context_engineering/rag.py`
- ✓ `src/context_engineering/advanced_rag.py`
- ✓ All tests passing

**Time Estimate:** 16 hours

---

## 🧪 Phase 3: Experiment Execution Part 1 (Week 3)

### Goals
- ✅ Run baseline calibration
- ✅ Execute Experiments 1 and 2
- ✅ Log all metrics and responses

---

### Day 1: Baseline Calibration

**Purpose:** Measure Gemini Flash 2.5's intrinsic fill % degradation

**Script:** `scripts/run_calibration.py`

**Methodology:**
```python
def calibration_test():
    """
    Needle-in-haystack across fill levels and positions
    """
    fill_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    positions = ["start", "middle", "end"]
    
    results = []
    for fill in fill_levels:
        for pos in positions:
            for trial in range(10):  # 10 needles per config
                needle = f"SECRET_CODE_{random_string()}"
                haystack = create_haystack(
                    size_tokens=int(1_000_000 * fill),
                    needle=needle,
                    position=pos
                )
                
                response = model.generate(
                    f"Find the secret code in the text.\n\n{haystack}"
                )
                
                recall = needle in response
                results.append({
                    'fill_pct': fill,
                    'position': pos,
                    'trial': trial,
                    'recall': recall
                })
    
    return results
```

**Expected Results:**
- Heatmap: fill % (x) × position (y) → recall %
- Quantify degradation curve
- Establish baseline for comparison

**Tasks:**
- [ ] Implement needle-in-haystack generator
- [ ] Run 180 calibration tests (6 fills × 3 pos × 10 trials)
- [ ] Generate heatmap visualization
- [ ] Document baseline degradation pattern

**Time:** 4 hours (includes API calls)

---

### Day 2-4: Experiment 1 - Needle in Multiple Haystacks

**Configuration:**
- Questions: 50
- Conditions: 4 (Naïve 1M, Engineered 1M, RAG 128k, Advanced RAG 128k)
- Fill levels: 5 (10%, 30%, 50%, 70%, 90%)
- Repetitions: 3

**Total API Calls:** 50 × 4 × 5 × 3 = 3,000

**Implementation:**
```python
# scripts/run_experiment.py

def run_experiment_1():
    questions = load_questions('data/questions/exp1_questions.json')
    corpus = load_corpus('data/raw/api_docs/')
    padding = load_corpus('data/raw/padding_corpus/')
    
    conditions = [
        NaiveContextAssembler(),
        StructuredContextAssembler(),
        RAGPipeline(),
        AdvancedRAGPipeline()
    ]
    
    for question in questions:
        for condition in conditions:
            for fill_pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for rep in range(3):
                    # Assemble context
                    context = condition.assemble(
                        documents=corpus,
                        target_tokens=int(1_000_000 * fill_pct),
                        padding_corpus=padding
                    )
                    
                    # Query model
                    prompt = f"{context}\n\nQuestion: {question['question']}"
                    response = model.generate(prompt)
                    
                    # Log result
                    log_result({
                        'experiment': 'exp1',
                        'question_id': question['id'],
                        'condition': condition.__class__.__name__,
                        'fill_pct': fill_pct,
                        'repetition': rep,
                        'response': response['response'],
                        'tokens_input': response['tokens_input'],
                        'tokens_output': response['tokens_output'],
                        'latency': response['latency_total'],
                        'cost': calculate_cost(response)
                    })
```

**Parallelization Strategy:**
- Run 4 parallel workers (one per condition)
- Each worker processes questions sequentially
- Respect 60 RPM rate limit per worker

**Progress Tracking:**
```python
# Update progress file every 10 completions
{
    "experiment": "exp1",
    "completed": 1250,
    "total": 3000,
    "progress_pct": 41.7,
    "estimated_time_remaining": "2.5 hours",
    "api_costs_so_far": 28.50
}
```

**Tasks:**
- [ ] Implement experiment runner with progress tracking
- [ ] Run all 3,000 API calls (~8 hours with rate limiting)
- [ ] Save raw responses to `results/raw/exp1_results.jsonl`
- [ ] Monitor for errors and retry failures
- [ ] Verify all questions completed

**Time:** 16 hours (includes waiting for API)

---

### Day 5-7: Experiment 2 - Context Pollution

**Configuration:**
- Questions: 20
- Conditions: 4
- Pollution levels: 5 (50k, 200k, 500k, 700k, 950k irrelevant tokens)
- Repetitions: 3

**Total API Calls:** 20 × 4 × 5 × 3 = 1,200

**Special Handling:**
```python
def create_polluted_context(base_corpus, pollution_level, condition):
    """
    Add irrelevant content to test robustness
    """
    base_tokens = 50_000  # Relevant content
    pollutant_tokens = pollution_level
    
    # Mix in irrelevant financial reports
    pollutant_docs = sample_pollutants(pollutant_tokens)
    
    # Assemble based on condition
    if isinstance(condition, NaiveContextAssembler):
        # Just concatenate base + pollutants
        context = base_corpus + "\n\n" + pollutant_docs
    elif isinstance(condition, StructuredContextAssembler):
        # Structure helps identify relevant vs irrelevant
        context = condition.structure_with_sections(
            relevant=base_corpus,
            irrelevant=pollutant_docs
        )
    else:  # RAG conditions
        # RAG should retrieve only relevant chunks
        context = condition.retrieve_and_assemble(
            query=question,
            corpus=base_corpus + pollutant_docs
        )
    
    return context
```

**Key Metrics:**
- Accuracy vs pollution level
- False positive rate (hallucinations from wrong docs)
- Degradation curve steepness

**Tasks:**
- [ ] Implement pollution injection
- [ ] Run 1,200 API calls
- [ ] Track which documents were cited
- [ ] Measure hallucination rate
- [ ] Save results to `results/raw/exp2_results.jsonl`

**Time:** 8 hours

---

## 🧪 Phase 4: Experiment Execution Part 2 (Week 4)

### Day 1-3: Experiment 3 - Multi-Turn Memory

**Configuration:**
- Scenarios: 10
- Turns per scenario: 5
- Conditions: 4
- Repetitions: 3

**Total API Calls:** 10 × 5 × 4 × 3 = 600 (but cumulative context grows each turn)

**Special Handling - Stateful Conversation:**
```python
def run_multi_turn_experiment(scenario, condition):
    """
    Maintain conversation state across turns
    """
    conversation_history = []
    customer_history = load_customer_history()
    docs = load_support_docs()
    
    for turn in scenario['turns']:
        # Build context based on condition
        if isinstance(condition, NaiveContextAssembler):
            # Include ALL history every time (grows linearly)
            context = (customer_history + 
                      "\n\n".join(conversation_history) + 
                      docs)
        
        elif isinstance(condition, StructuredContextAssembler):
            # Structured sections for different memory types
            context = condition.assemble_multi_turn(
                customer=customer_history,
                conversation=conversation_history,
                docs=docs,
                current_turn=turn
            )
        
        elif isinstance(condition, RAGPipeline):
            # Retrieve relevant history + docs per turn
            context = condition.retrieve_contextual(
                query=turn['customer_says'],
                episodic_memory=conversation_history,
                semantic_memory=docs
            )
        
        # Generate response
        response = model.generate(
            f"{context}\n\nCustomer: {turn['customer_says']}\nAgent:"
        )
        
        # Add to history
        conversation_history.append({
            'turn': turn['turn_number'],
            'customer': turn['customer_says'],
            'agent': response['response']
        })
        
        # Evaluate
        evaluate_turn(response, turn['expected_actions'])
```

**Evaluation Metrics:**
- Coherence score (does response make sense in context?)
- Fact retention (can model recall info from turn 2 at turn 5?)
- Cumulative cost (token usage grows each turn)
- Memory compression effectiveness

**Tasks:**
- [ ] Implement stateful conversation manager
- [ ] Create 10 diverse scenarios
- [ ] Run 600 turn-level API calls
- [ ] Evaluate coherence (use GPT-4 as judge)
- [ ] Measure memory retention with targeted questions
- [ ] Track cumulative token usage per conversation

**Time:** 12 hours

---

### Day 4-5: Experiment 4 - Precision Retrieval

**Configuration:**
- Questions: 60 (30 facts, 20 comparisons, 10 meta-analyses)
- Conditions: 4
- Fill levels: 5
- Repetitions: 3

**Total API Calls:** 60 × 4 × 5 × 3 = 3,600

**Special Focus - Citation Accuracy:**
```python
def evaluate_citation_accuracy(response, ground_truth):
    """
    Check if model correctly cited source papers
    """
    cited_papers = extract_citations(response)
    required_papers = ground_truth['required_docs']
    
    precision = len(set(cited_papers) & set(required_papers)) / len(cited_papers)
    recall = len(set(cited_papers) & set(required_papers)) / len(required_papers)
    
    return {
        'citation_precision': precision,
        'citation_recall': recall,
        'f1': 2 * precision * recall / (precision + recall)
    }
```

**Tasks:**
- [ ] Load 100 academic papers with parsed structure
- [ ] Run 3,600 API calls (~9 hours with rate limiting)
- [ ] Extract citations from responses
- [ ] Measure citation accuracy
- [ ] Compute precision@K for fact retrieval
- [ ] Evaluate completeness for meta-analyses

**Time:** 16 hours

---

### Day 6-7: Experiment 5 - Cost-Latency Frontier

**No new data collection** - analysis of Experiments 1-4

**Implementation:**
```python
# scripts/analyze_frontier.py

def compute_pareto_frontier():
    """
    Find dominant strategies across quality, cost, latency
    """
    all_results = load_all_experiments()
    
    # Normalize metrics to [0, 1]
    normalized = normalize_metrics(all_results)
    
    # Compute efficiency score
    normalized['efficiency'] = (
        normalized['quality'] / 
        (normalized['cost'] * normalized['latency'])
    )
    
    # Find Pareto frontier
    # A point is on the frontier if no other point dominates it
    # on ALL three dimensions
    frontier = []
    for point in normalized:
        is_dominated = any(
            (other['quality'] >= point['quality'] and
             other['cost'] <= point['cost'] and
             other['latency'] <= point['latency'] and
             other != point)
            for other in normalized
        )
        if not is_dominated:
            frontier.append(point)
    
    return frontier
```

**Visualizations:**
1. 3D scatter plot: Quality × Cost × Latency
2. 2D projections: Quality vs Cost, Quality vs Latency
3. Efficiency score ranking
4. Cost-latency trade-off curves per quality tier

**Tasks:**
- [ ] Aggregate metrics from all experiments
- [ ] Normalize across different scales
- [ ] Compute Pareto frontier
- [ ] Generate 3D interactive plot (Plotly)
- [ ] Create 2D projections
- [ ] Rank conditions by efficiency score

**Time:** 8 hours

---

## 📊 Phase 5: Evaluation & Analysis (Week 5)

### Goals
- ✅ Compute all automated metrics
- ✅ Conduct human evaluation on sample
- ✅ Perform statistical hypothesis testing
- ✅ Generate visualizations

---

### Day 1-2: Automated Metrics Computation

**Metrics to Compute:**

1. **Correctness (LLM-as-Judge):**
```python
def llm_judge_correctness(response, ground_truth):
    """
    Use GPT-4 or Claude to score correctness
    """
    judge_prompt = f"""
    Rate the correctness of this response on a scale of 0-1.
    
    Question: {question}
    Ground Truth: {ground_truth}
    Model Response: {response}
    
    Consider:
    - Factual accuracy
    - Completeness
    - Relevance
    
    Return only a number between 0 and 1.
    """
    
    score = judge_model.generate(judge_prompt)
    return float(score)
```

2. **Citation Accuracy:**
```python
def check_citations(response, context):
    """
    Verify claims are supported by context
    """
    claims = extract_claims(response)
    
    supported = []
    for claim in claims:
        # Check if claim appears in context
        is_supported = any(
            fuzzy_match(claim, chunk) > 0.8
            for chunk in chunk_context(context)
        )
        supported.append(is_supported)
    
    return sum(supported) / len(supported)
```

3. **Hallucination Detection:**
```python
def detect_hallucinations(response, context):
    """
    Find claims in response NOT in context
    """
    claims = extract_factual_claims(response)
    
    hallucinations = [
        claim for claim in claims
        if not is_grounded_in(claim, context)
    ]
    
    return len(hallucinations) / len(claims)
```

4. **Cost & Latency:**
```python
def calculate_metrics(response_data):
    return {
        'tokens_input': response_data['tokens_input'],
        'tokens_output': response_data['tokens_output'],
        'cost_input': response_data['tokens_input'] * 0.00001875 / 1000,
        'cost_output': response_data['tokens_output'] * 0.000075 / 1000,
        'cost_total': cost_input + cost_output,
        'latency_ttft': response_data.get('latency_ttft'),
        'latency_total': response_data['latency_total']
    }
```

**Tasks:**
- [ ] Implement all metric calculators
- [ ] Run evaluation on all 9,000+ responses
- [ ] Save computed metrics to `results/metrics/all_metrics.csv`
- [ ] Generate summary statistics per condition
- [ ] Identify outliers and edge cases

**Time:** 16 hours

---

### Day 3-4: Human Evaluation

**Sample Size:** 20% of responses = ~1,800 items

**Stratified Sampling:**
- Proportional across experiments
- Proportional across conditions
- Include mix of difficulties

**Evaluation Interface:**
```python
# src/evaluation/human_eval.py

def create_evaluation_task(sample_id):
    """
    Blind evaluation - hide condition labels
    """
    item = load_sample(sample_id)
    
    # Shuffle to avoid order bias
    randomized_item = randomize_presentation(item)
    
    return {
        'id': sample_id,
        'question': item['question'],
        'response': item['response'],
        'context_snippet': truncate_context(item['context'], 1000),
        'evaluation_form': {
            'correctness': 'Rate 1-5',
            'coherence': 'Rate 1-5',
            'has_hallucinations': 'Yes/No',
            'citations_accurate': 'Yes/No',
            'comments': 'Optional text'
        }
    }
```

**If Self-Evaluating:**
- Spread over multiple days to reduce fatigue bias
- Randomize order each day
- Track time per evaluation (~2-3 min target)

**If Using External Evaluators:**
- Create clear evaluation guidelines
- Provide training examples
- Calculate inter-rater reliability (Cohen's kappa)

**Tasks:**
- [ ] Create evaluation interface (web form or CLI tool)
- [ ] Sample 1,800 items with stratification
- [ ] Conduct evaluations
- [ ] Calculate inter-rater agreement (if multiple raters)
- [ ] Save ratings to `results/metrics/human_eval_ratings.csv`

**Time:** 16 hours (highly variable based on evaluator speed)

---

### Day 5-7: Statistical Analysis

**Hypothesis Testing:**

```python
# src/utils/stats.py

def test_h1(metrics_df):
    """
    H1: Engineered 1M > Naïve 1M by ≥15%
    """
    naive = metrics_df[metrics_df.condition == 'naive_1m']
    engineered = metrics_df[metrics_df.condition == 'engineered_1m']
    
    # Paired t-test (same questions across conditions)
    t_stat, p_value = ttest_rel(
        engineered.correctness,
        naive.correctness
    )
    
    # Effect size (Cohen's d)
    effect_size = cohen_d(engineered.correctness, naive.correctness)
    
    # Mean improvement
    improvement = (engineered.correctness.mean() - 
                  naive.correctness.mean()) / naive.correctness.mean()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'improvement_pct': improvement * 100,
        'conclusion': 'SUPPORTED' if (p_value < 0.05 and improvement >= 0.15) 
                     else 'REJECTED'
    }
```

**Regression Analysis:**
```python
import statsmodels.formula.api as smf

def regression_analysis(df):
    """
    Model: quality ~ condition + fill_pct + condition:fill_pct
    """
    model = smf.ols(
        'correctness ~ C(condition) + fill_pct + C(condition):fill_pct',
        data=df
    ).fit()
    
    return model.summary()
```

**ANOVA:**
```python
from scipy.stats import f_oneway

def anova_by_condition(df):
    """
    Test if conditions have significantly different means
    """
    groups = [
        df[df.condition == cond].correctness
        for cond in df.condition.unique()
    ]
    
    f_stat, p_value = f_oneway(*groups)
    
    return {'f_statistic': f_stat, 'p_value': p_value}
```

**Tasks:**
- [ ] Run paired t-tests for H1 and H2
- [ ] Compute effect sizes (Cohen's d)
- [ ] ANOVA across all conditions
- [ ] Regression analysis with interaction terms
- [ ] Bootstrap confidence intervals (1000 samples)
- [ ] Sensitivity analysis (remove outliers)
- [ ] Generate statistical summary report

**Deliverables:**
- ✓ `results/analysis/hypothesis_tests.json`
- ✓ `results/analysis/regression_results.txt`
- ✓ `results/analysis/effect_sizes.csv`
- ✓ `results/analysis/anova_results.json`

**Time:** 24 hours

---

## 📝 Phase 6: Reporting & Documentation (Week 6)

### Goals
- ✅ Create all visualizations
- ✅ Write final comprehensive report
- ✅ Document codebase
- ✅ Prepare for publication/sharing

---

### Day 1-2: Visualizations

**Key Visualizations:**

1. **Fill % Degradation Curves:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fill_degradation():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for condition in conditions:
        data = results[results.condition == condition]
        
        ax.plot(
            data.fill_pct,
            data.correctness.mean(),
            label=condition,
            marker='o'
        )
    
    ax.set_xlabel('Context Fill Percentage')
    ax.set_ylabel('Correctness Score')
    ax.set_title('Performance Degradation vs Fill %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('results/visualizations/fill_degradation.png', dpi=300)
```

2. **Condition Comparison (Bar Charts):**
- Correctness by experiment
- Cost by experiment
- Latency by experiment
- With error bars (95% CI)

3. **Cost-Quality Scatter:**
- X-axis: Quality (correctness)
- Y-axis: Cost per query
- Color: Condition
- Size: Latency
- Pareto frontier highlighted

4. **Pollution Robustness:**
- Line plot: Accuracy vs pollution level
- Separate lines per condition
- Shows degradation curves

5. **3D Pareto Frontier:**
```python
import plotly.graph_objects as go

def plot_3d_frontier():
    fig = go.Figure(data=[go.Scatter3d(
        x=data.quality,
        y=data.cost,
        z=data.latency,
        mode='markers',
        marker=dict(
            size=5,
            color=data.efficiency,
            colorscale='Viridis',
            showscale=True
        ),
        text=data.condition
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Quality',
            yaxis_title='Cost',
            zaxis_title='Latency'
        ),
        title='3D Cost-Quality-Latency Frontier'
    )
    
    fig.write_html('results/visualizations/pareto_3d.html')
```

6. **Position Bias Heatmap:**
- From calibration data
- X: Fill %, Y: Position, Color: Recall

**Tasks:**
- [ ] Generate all 20+ visualizations
- [ ] Export high-resolution images (300 DPI)
- [ ] Create interactive Plotly versions
- [ ] Build optional dashboard (Streamlit)

**Time:** 16 hours

---

### Day 3-5: Final Report Writing

**Report Structure (FINAL_REPORT.md):**

**1. Executive Summary (1 page)**
- Key findings in 3-5 bullet points
- Hypothesis acceptance/rejection
- Top recommendations

**2. Introduction (2 pages)**
- Motivation: Why context engineering matters
- Research questions
- Hypotheses
- Contribution to field

**3. Literature Review (2 pages)**
- "Lost in the Middle" phenomenon
- RAG approaches
- Context window utilization
- Gap in current research

**4. Methodology (4 pages)**
- Experimental design overview
- Model configuration (Gemini Flash 2.5)
- Four context engineering conditions
- Fill % control mechanism
- Five experiments described
- Evaluation metrics
- Statistical methods

**5. Results (8 pages)**

**Experiment 1: Needle in Multiple Haystacks**
- Tables: Mean correctness by condition × fill %
- Figures: Degradation curves
- Statistical tests: t-tests, effect sizes
- Key finding: "Engineered 1M outperforms naive by X% at high fill %"

**Experiment 2: Context Pollution**
- Tables: Accuracy vs pollution level
- Figures: Robustness curves
- Key finding: "RAG maintains 95% accuracy even at 90% pollution"

**Experiment 3: Multi-Turn Memory**
- Tables: Coherence and retention scores
- Figures: Cumulative cost across turns
- Key finding: "Stateful RAG reduces cost by 60% vs naive"

**Experiment 4: Precision Retrieval**
- Tables: Citation accuracy, precision@K
- Figures: Performance on different question types
- Key finding: "Structured context improves citation accuracy by X%"

**Experiment 5: Cost-Latency Frontier**
- Tables: Efficiency scores
- Figures: 3D Pareto frontier, 2D projections
- Key finding: "Advanced RAG dominates on efficiency frontier"

**Hypothesis Testing:**
```
H1: Engineered 1M > Naive 1M
- Result: SUPPORTED (p < 0.001, d = 0.85, +22% quality)
- At 90% fill: Engineered maintained 82% vs Naive's 58%
- Cost savings: 18% due to better instruction following

H2: 128k RAG ≈ Naive 1M
- Result: PARTIALLY SUPPORTED
- Quality: Within 3% (not significant at p=0.05)
- Cost: 65% lower
- Latency: 1.8x (within 2x threshold)
- Note: Confounded by fill % difference
```

**6. Discussion (4 pages)**
- Interpretation of results
- Why engineering helps more at high fill %
- RAG's advantages and trade-offs
- Limitations:
  - Fill % confound in H2
  - Model-specific (Gemini Flash 2.5)
  - English-only corpus
  - Synthetic scenarios
- Practical implications for developers

**7. Recommendations (2 pages)**

**Decision Framework:**
```
IF your task requires:
- <100k tokens → Use naive (simplest)
- 100k-500k tokens + simple lookup → Use structured
- 100k-500k tokens + complex reasoning → Use basic RAG
- >500k tokens OR high pollution → Use advanced RAG

Cost-sensitive?
→ RAG (65% cost savings)

Latency-sensitive?
→ Structured (lowest latency for 1M context)

Quality-critical?
→ Advanced RAG or Engineered 1M (equivalent quality)
```

**8. Conclusion (1 page)**
- Summary of contributions
- Key takeaways
- Future research directions

**9. Appendices**
- Sample questions and responses
- Full statistical tables
- Code repository structure
- Corpus details

**Tasks:**
- [ ] Write all sections
- [ ] Insert visualizations
- [ ] Create tables
- [ ] Format citations
- [ ] Peer review (optional)
- [ ] Export to PDF

**Time:** 24 hours

---

### Day 6-7: Code Documentation & Cleanup

**Documentation Tasks:**

1. **Docstrings:**
```python
def assemble_context(documents: List[str], target_tokens: int) -> str:
    """
    Assemble documents into context string with token budget.
    
    Args:
        documents: List of document strings to include
        target_tokens: Maximum tokens to include (for fill % control)
        
    Returns:
        Assembled context string
        
    Raises:
        ValueError: If target_tokens < minimum viable context
        
    Example:
        >>> docs = load_corpus('data/raw/api_docs/')
        >>> context = assemble_context(docs, target_tokens=100_000)
        >>> count_tokens(context)
        99847  # Close to target
    """
    pass
```

2. **README Updates:**
- Add "Results" section with key findings
- Update "Quick Start" with actual commands
- Add troubleshooting section

3. **Code Cleanup:**
- [ ] Remove commented-out code
- [ ] Fix TODOs
- [ ] Run linter (black, flake8)
- [ ] Run type checker (mypy)
- [ ] Update requirements.txt with pinned versions

4. **Testing:**
- [ ] Achieve >80% code coverage
- [ ] Add integration tests
- [ ] Document test running instructions

5. **Repository Organization:**
```
# Tag final release
git tag -a v1.0.0 -m "Complete experimental suite"
git push origin v1.0.0

# Create release notes
RELEASE_NOTES.md:
- Summary of findings
- Link to final report
- Instructions for replication
```

**Deliverables:**
- ✓ All code documented
- ✓ Tests passing
- ✓ README updated with results
- ✓ Repository cleaned and organized
- ✓ Release tagged (v1.0.0)

**Time:** 16 hours

---

## 📋 Milestones & Success Criteria

| Week | Milestone | Success Criteria | Checkpoint |
|------|-----------|------------------|------------|
| 1 | Foundation | ✅ 200 questions ready, all corpora collected | Review question quality |
| 2 | Engineering | ✅ All 4 strategies implemented, tests passing | Test on sample corpus |
| 3 | Experiments 1-2 | ✅ 4,200 API calls completed, data logged | Spot-check response quality |
| 4 | Experiments 3-5 | ✅ All 9,000+ experiments done | Verify data completeness |
| 5 | Evaluation | ✅ Metrics computed, hypotheses tested | Review statistical significance |
| 6 | Final Report | ✅ Report published, code documented | Peer review findings |

---

## 🎯 Risk Management

### Identified Risks & Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **API Rate Limits** | High | High | Implement exponential backoff, parallelize with per-worker limits |
| **Budget Overrun** | Medium | High | Monitor costs daily, set hard stop at $200 |
| **Corpus Quality Issues** | Medium | Medium | Manual review of 10% sample before proceeding |
| **LLM Judge Unreliability** | Medium | Medium | Calibrate with human eval on 20% sample |
| **Unexpected Results** | Low | Medium | Document thoroughly, rerun suspicious outliers |
| **Code Bugs in Pipeline** | High | High | Write comprehensive tests, manual spot-checks |
| **Time Overrun** | Medium | Low | Prioritize core experiments, cut optional analyses |
| **API Downtime** | Low | High | Cache intermediate results, checkpoint progress |

### Budget Tracking

**Daily Cost Monitoring:**
```python
# scripts/monitor_budget.py

def check_budget():
    total_spent = calculate_total_api_costs()
    budget_limit = 174.00
    
    if total_spent > budget_limit * 0.9:
        send_alert("Budget at 90%: $%.2f / $%.2f" % (total_spent, budget_limit))
    
    if total_spent > budget_limit:
        raise BudgetExceededError("STOP: Budget limit exceeded!")
    
    return {
        'spent': total_spent,
        'remaining': budget_limit - total_spent,
        'utilization_pct': total_spent / budget_limit * 100
    }
```

**Estimated Cost Breakdown:**
```
Calibration (180 calls):          $1.50
Experiment 1 (3,000 calls):      $30.00
Experiment 2 (1,200 calls):      $12.00
Experiment 3 (600 calls):        $15.00  (larger contexts)
Experiment 4 (3,600 calls):      $36.00
LLM-as-Judge (9,000 evals):      $27.00
Buffer for reruns:               $52.50
-------------------------------------------
TOTAL:                          $174.00
```

---

## 🔄 Iteration & Adaptation Strategy

### Weekly Review Points

**End of Week 1:**
- Review question quality with fresh eyes
- Test one question end-to-end through full pipeline
- Adjust corpus if gaps found

**End of Week 2:**
- Run mini-experiment (10 questions, 1 condition)
- Verify metrics computation works
- Adjust engineering strategies if needed

**End of Week 3:**
- Analyze preliminary results from Exp 1-2
- Check if trends match hypotheses
- Decide if adjustments needed for Exp 3-5

**End of Week 4:**
- Full data quality check
- Identify and rerun any failures
- Begin preliminary statistical analysis

**End of Week 5:**
- Draft key findings
- Identify gaps in analysis
- Plan final visualizations

**End of Week 6:**
- Final review with external feedback
- Prepare for publication/sharing

### Adaptation Triggers

**If budget runs low (>80% spent before Week 4):**
- Reduce repetitions from 3 to 2
- Reduce fill levels from 5 to 3 (10%, 50%, 90%)
- Cut Experiment 5 (can be computed without new data)

**If results are unexpectedly clear (p < 0.001 everywhere):**
- Stop early and redirect resources to deeper analysis
- Add ablation studies
- Test edge cases

**If results are unexpectedly ambiguous (p > 0.1 everywhere):**
- Increase repetitions to 5
- Add more diagnostic experiments
- Check for implementation bugs

**If one condition dominates:**
- Investigate why
- Add harder test cases
- Consider different model (Claude for comparison)

---

## 📊 Data Management

### Storage Structure

```
results/
├── raw/                          # Raw API responses (NEVER DELETE)
│   ├── exp1_results.jsonl       # One JSON object per line
│   ├── exp2_results.jsonl
│   ├── exp3_results.jsonl
│   ├── exp4_results.jsonl
│   └── calibration.jsonl
│
├── metrics/                      # Computed metrics
│   ├── all_metrics.csv          # All computed metrics
│   ├── summary_stats.json       # Aggregated statistics
│   ├── human_eval_ratings.csv   # Human evaluation data
│   └── citation_analysis.json   # Citation accuracy details
│
├── analysis/                     # Statistical analysis outputs
│   ├── hypothesis_tests.json    # H1 and H2 test results
│   ├── regression_results.txt   # Regression model outputs
│   ├── effect_sizes.csv         # Cohen's d for all comparisons
│   ├── anova_results.json       # ANOVA tables
│   └── pareto_frontier.json     # Experiment 5 results
│
└── visualizations/               # All plots and charts
    ├── fill_degradation.png
    ├── condition_comparison.png
    ├── cost_quality_scatter.png
    ├── pollution_robustness.png
    ├── pareto_3d.html
    ├── position_heatmap.png
    └── ... (20+ charts)
```

### Backup Strategy

```bash
# Daily backups
rsync -av results/ ../results_backup_$(date +%Y%m%d)/

# After each experiment completion
tar -czf results_exp1_backup.tar.gz results/raw/exp1_results.jsonl
```

### Version Control

```bash
# Commit after each major milestone
git add results/metrics/
git commit -m "Add metrics for Experiment 1"

# Tag major versions
git tag -a exp1-complete -m "Experiment 1 completed: 3000 API calls"
```

---

## 🔍 Quality Assurance Checklist

### Before Each Experiment

- [ ] Test pipeline end-to-end with 1 question
- [ ] Verify API credentials are valid
- [ ] Check disk space (need ~5GB for results)
- [ ] Confirm budget remaining sufficient
- [ ] Review question set for errors
- [ ] Verify corpus loaded correctly

### During Experiment Execution

- [ ] Monitor progress every hour
- [ ] Check for API errors
- [ ] Verify responses look reasonable (spot-check 10)
- [ ] Track cost accumulation
- [ ] Save checkpoints every 100 completions

### After Each Experiment

- [ ] Verify all questions completed
- [ ] Check for any missing data
- [ ] Run data validation script
- [ ] Compute basic statistics (mean, std)
- [ ] Generate quick visualization
- [ ] Backup results immediately

### Code Quality

- [ ] All functions have docstrings
- [ ] Tests cover >80% of code
- [ ] No hardcoded paths (use config)
- [ ] Logging throughout pipeline
- [ ] Error handling for API failures
- [ ] Type hints on public functions

---

## 📞 Troubleshooting Guide

### Common Issues & Solutions

**Issue: API Rate Limit Hit**
```
Error: 429 Too Many Requests

Solution:
- Check rate_limit_rpm in config (should be 60)
- Verify parallel workers ≤ 4
- Add exponential backoff (already implemented)
- If persistent, reduce parallelism to 2 workers
```

**Issue: Out of Memory**
```
Error: MemoryError

Solution:
- Don't load all questions at once
- Process in batches of 100
- Clear context variable after each query
- Increase system swap space
```

**Issue: Token Count Mismatch**
```
Expected: 100k tokens
Actual: 87k tokens

Solution:
- tiktoken approximation differs from Gemini's tokenizer
- Accept ±10% variance
- Use actual returned token counts for metrics
```

**Issue: Low Correctness Scores**
```
All conditions getting <50% correctness

Solution:
- Check question difficulty (may be too hard)
- Review ground truth answers
- Verify context actually contains answers
- Try manual test with same context
- Adjust LLM judge prompt for fairness
```

**Issue: No Significant Differences**
```
All conditions perform similarly (p > 0.1)

Solution:
- Check if questions too easy/hard
- Verify conditions actually differ (inspect contexts)
- Increase sample size (more repetitions)
- Add more challenging questions
```

**Issue: Unexpected API Errors**
```
Error: Service Unavailable

Solution:
- Implement retry with exponential backoff (done)
- Wait 5 minutes and resume
- Check Google AI status page
- Switch to backup API key if available
```

---

## 🚀 Post-Project Extensions

### Optional Follow-Up Work

**1. Cross-Model Validation**
- Run subset on Claude Sonnet 4.5
- Compare Gemini vs Claude patterns
- Test if findings generalize

**2. Ablation Studies**
- Remove individual engineering techniques
- Test contribution of TOC, metadata, reranking separately
- Optimize chunk size and overlap

**3. Real-World Case Study**
- Partner with organization
- Test on actual production use case
- Measure business impact (user satisfaction, task completion)

**4. Expanded Corpus**
- Add multilingual documents
- Test on code repositories
- Test on conversational data

**5. Advanced Techniques**
- Test query expansion
- Try late chunking
- Experiment with contextual embeddings
- Test mixture-of-experts retrieval

**6. Publication Preparation**
- Write academic paper
- Submit to conference (NeurIPS, EMNLP, ICLR)
- Create blog post for broader audience
- Present at local meetup

---

## 📚 Resources & References

### Documentation

- **Gemini API Docs:** https://ai.google.dev/gemini-api/docs
- **Context Caching:** https://ai.google.dev/gemini-api/docs/caching
- **Rate Limits:** https://ai.google.dev/gemini-api/docs/rate-limits

### Key Papers

- Liu et al. (2023) - "Lost in the Middle" - https://arxiv.org/abs/2307.03172
- Lewis et al. (2020) - "RAG" - https://arxiv.org/abs/2005.11401
- Anthropic (2024) - "Contextual Retrieval" - https://www.anthropic.com/news/contextual-retrieval

### Tools

- **tiktoken:** Token counting - https://github.com/openai/tiktoken
- **FAISS:** Vector search - https://github.com/facebookresearch/faiss
- **ChromaDB:** Vector store - https://www.trychroma.com/
- **scipy:** Statistical analysis - https://scipy.org/
- **plotly:** Interactive viz - https://plotly.com/python/

### Community

- **r/MachineLearning:** Share results, get feedback
- **LLM Discord Servers:** Real-time troubleshooting
- **Twitter/X:** #LLMs #RAG hashtags
- **GitHub:** Open-source your repo after publication

---

## ✅ Final Deliverables Checklist

### Code & Data

- [ ] Complete source code repository
- [ ] All corpus files (or download scripts)
- [ ] Question sets with ground truth
- [ ] Raw experimental results (9,000+ responses)
- [ ] Computed metrics CSV
- [ ] Human evaluation data

### Analysis

- [ ] Statistical test results
- [ ] Effect sizes and confidence intervals
- [ ] Regression analysis outputs
- [ ] Pareto frontier data

### Visualizations

- [ ] 20+ publication-quality figures (300 DPI)
- [ ] Interactive 3D plots (Plotly HTML)
- [ ] Optional: Live dashboard

### Documentation

- [ ] Final report (20+ pages, PDF)
- [ ] README with results summary
- [ ] API documentation
- [ ] Replication instructions
- [ ] License file (MIT)

### Presentation Materials

- [ ] Executive summary (1-pager)
- [ ] Slide deck (optional, 20 slides)
- [ ] Blog post draft (optional)
- [ ] Demo video (optional)

---

## 🎓 Learning Outcomes

By completing this project, you will have:

**Technical Skills:**
- ✅ Designed and executed rigorous experiments
- ✅ Built end-to-end RAG pipelines
- ✅ Implemented context engineering techniques
- ✅ Used LLM APIs at scale (9,000+ calls)
- ✅ Performed statistical hypothesis testing
- ✅ Created publication-quality visualizations

**Domain Knowledge:**
- ✅ Deep understanding of context window limitations
- ✅ Practical experience with prompt engineering
- ✅ Knowledge of RAG architectures
- ✅ Insight into cost-quality trade-offs
- ✅ Understanding of evaluation methodologies

**Research Skills:**
- ✅ Experimental design for ML systems
- ✅ Controlling for confounding variables
- ✅ Statistical analysis and interpretation
- ✅ Scientific writing and visualization
- ✅ Reproducible research practices

**Professional Outcomes:**
- Portfolio project demonstrating ML engineering skills
- Publishable research (conference or blog)
- Open-source contribution to community
- Practical insights for production deployments
- Network connections from sharing findings

---

## 📝 Notes & Reminders

### Important Considerations

1. **Gemini Flash 2.5 Specifics:**
   - 1M token window but recall degrades
   - Free tier has lower rate limits
   - Pricing: $0.00001875 per 1k input tokens
   - No streaming TTFT metrics available

2. **Reproducibility:**
   - Set random seeds everywhere
   - Pin exact library versions
   - Save model versions (API snapshot date)
   - Document any manual decisions

3. **Ethics:**
   - Ensure no PII in corpus
   - Credit all data sources
   - Be transparent about limitations
   - Share negative results too

4. **Time Management:**
   - Build buffer for unexpected issues
   - Don't perfectionism-trap on corpus collection
   - Start writing report early (Week 4)
   - Parallelize where possible

---

## 🤝 Collaboration Opportunities

### If Working with Others

**Roles (can be split):**
- **Data Engineer:** Corpus collection and preprocessing
- **ML Engineer:** Context engineering implementations
- **Researcher:** Experiment design and statistical analysis
- **Writer:** Report and documentation
- **Designer:** Visualizations and dashboard

**Communication:**
- Weekly sync meetings (1 hour)
- Daily async updates (Slack/Discord)
- Shared progress tracker (Notion/Trello)
- Code reviews on PRs
- Pair programming for complex logic

### Getting Feedback

**During Project:**
- Share early results on Twitter/X
- Post in ML subreddits after Exp 1
- Present at local meetups mid-project

**After Completion:**
- Submit to conferences (if academic)
- Write detailed blog post
- Open-source on GitHub
- Create YouTube walkthrough
- Offer to review similar projects

---

## 🏁 Success Metrics

### How to Know You've Succeeded

**Minimum Viable Success:**
- ✅ Both hypotheses tested with p-values
- ✅ Clear conclusion (supported or rejected)
- ✅ Final report written
- ✅ Code runs without errors

**Good Success:**
- ✅ Statistically significant results (p < 0.05)
- ✅ Effect sizes > 0.3 (medium)
- ✅ Publication-quality visualizations
- ✅ Code documented and tested
- ✅ Shared publicly (GitHub + blog)

**Exceptional Success:**
- ✅ Novel insights beyond hypotheses
- ✅ Results generalize across multiple models
- ✅ Conference acceptance or high-impact blog
- ✅ Replication by others
- ✅ Practical adoption in production systems
- ✅ 100+ GitHub stars

---

## 🎯 Closing Thoughts

This is an ambitious but achievable 6-week project. The key to success is:

1. **Stay organized** - Follow this plan, adapt as needed
2. **Check quality early** - Test everything with small samples first
3. **Document as you go** - Don't leave all writing to the end
4. **Be flexible** - Results might surprise you, that's okay
5. **Share progress** - Get feedback, build in public

Remember: **Negative results are valid results.** If naïve long-context wins, that's a valuable finding too. The goal is truth, not confirmation.

Good luck! 🚀

---

**Last Updated:** October 30, 2025  
**Version:** 1.0  
**Status:** Ready for Implementation  
**Estimated Total Time:** 240 hours (6 weeks × 40 hours/week)  
**Estimated Budget:** $174 (API costs)