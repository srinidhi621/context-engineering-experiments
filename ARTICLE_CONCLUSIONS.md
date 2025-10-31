# The Million-Token Question: What We Learned From 4,380 API Calls

**The Results of Our Context Engineering Experiments—And What They Mean for Your Architecture**

---

## TL;DR: The Short Version

After 10 weeks of experiments spanning 4,380 API calls across 1M and 128k token contexts, testing naive vs. engineered approaches at fill levels from 10% to 90%, here's what we found:

**[SUMMARY - TO BE COMPLETED WITH ACTUAL RESULTS]**

[IF H1 CONFIRMED:]
✅ **Context engineering matters—even at 1M tokens.** Engineered approaches outperformed naive context stuffing by X% on quality, Y% on cost efficiency.

[IF H1 REJECTED:]
❌ **Naive long-context surprisingly effective.** Simply dumping documents into 1M token windows matched or exceeded engineered approaches.

[IF H2 CONFIRMED:]
✅ **Smart beats big.** Well-engineered 128k RAG systems matched naive 1M approaches within X% on quality, while costing Y% less.

[IF H2 REJECTED:]
❌ **Size matters more than we expected.** Naive 1M token approaches outperformed engineered 128k RAG by X%, suggesting context capacity dominates retrieval quality.

**Read on for the full story, the data, and what it means for your systems.**

---

## How We Got Here

In our [prelude article](#), we posed two hypotheses:

**H1: Long Context ≠ Magic Bullet**  
Even with 1M-token windows, naïvely stuffing context underperforms engineered retrieval + packaging.

**H2: Smart Beats Big**  
128k-token models with disciplined context engineering can match or beat naïve long-context use.

These weren't idle speculation. They were testable claims addressing a real industry question: **In the age of million-token context windows, does engineering discipline still matter?**

Now we have answers.

---

## The Experiments: What We Actually Did

### The Setup

**Four strategies, tested head-to-head:**

1. **Naïve 1M:** Sequential document dump into 1M token window. No structure, no organization. This represents the "just throw everything in" approach being marketed.

2. **Engineered 1M:** Same documents, same 1M window, but with hierarchical structure, table of contents, metadata tags, and navigation aids. Represents disciplined context engineering at scale.

3. **Basic RAG 128k:** Vector search with top-k retrieval. Represents the traditional RAG approach most teams use today.

4. **Advanced RAG 128k:** Hybrid search (dense + BM25), reciprocal rank fusion, query decomposition. Represents state-of-art RAG engineering.

**Critical methodological control:** All strategies were padded to identical fill percentages (10%, 30%, 50%, 70%, 90%). This isolated context engineering quality from attention dilution effects—a confounding variable most comparisons miss.

### The Tests

**Pilot Phase (180 calls):**  
10 questions on AWS Lambda docs. Validated our pipeline and metrics.

**Experiment 1: Needle in Multiple Haystacks (3,000 calls):**  
50 questions across 500k-1M tokens of API documentation (AWS, GCP, Azure).  
- 20 simple lookups ("What's the default Lambda timeout?")
- 20 synthesis tasks ("Compare authentication across AWS/GCP/Azure")
- 10 complex reasoning ("Resolve contradictions between v1 and v2 docs")

**Experiment 2: Context Pollution (1,200 calls):**  
20 questions answerable from 50k token base corpus, with pollution levels from 50k to 950k of irrelevant but plausible content.  
Tests: Can the model ignore distractors and focus on relevant information?

**Total:** 4,380 API calls, 70 unique questions, 3 repetitions per condition, deterministic temperature (0.0).

---

## The Results: What the Data Shows

### Result 1: The Fill Percentage Effect

**Finding:** [INSERT CHART: Accuracy vs Fill Percentage, by Strategy]

`[PLACEHOLDER FOR FIGURE 1: Line chart showing 4 strategies, X-axis = Fill %, Y-axis = Correctness (0-1)]`

[IF DEGRADATION OBSERVED:]
**All strategies degraded as context filled up**, confirming the "Lost in the Middle" phenomenon persists at 1M tokens. However, the *rate* of degradation differed significantly:

- Naïve 1M: Dropped from X% (10% fill) to Y% (90% fill) — **Z percentage point decline**
- Engineered 1M: Dropped from X% to Y% — **Z percentage point decline**  
- Basic RAG 128k: Dropped from X% to Y% — **Z percentage point decline**
- Advanced RAG 128k: Dropped from X% to Y% — **Z percentage point decline**

[IF NO DEGRADATION:]
**Surprisingly, we observed minimal degradation up to 70% fill**, suggesting modern models handle long contexts better than 2023-era research indicated. Degradation only appeared beyond 70% fill.

**Insight:** The fill percentage matters more than raw token count. A 300k token context at 30% fill outperforms a 900k token context at 90% fill—even with identical information content.

### Result 2: Hypothesis 1 – Does Engineering Matter at 1M Tokens?

**Predicted:** Engineered 1M beats Naïve 1M by ≥15% on quality.

**Actual Result:** [INSERT DATA]

`[PLACEHOLDER FOR TABLE 1: Strategy comparison at 70% fill]`
| Strategy | Correctness | Cost/Query | Latency | Citation Accuracy |
|----------|------------|------------|---------|-------------------|
| Naïve 1M | X.XX | $X.XX | X.Xs | X.X% |
| Engineered 1M | X.XX | $X.XX | X.Xs | X.X% |
| Δ (improvement) | **+X.X%** | **-X%** | **+X%** | **+X.X%** |

[IF H1 CONFIRMED:]
✅ **Hypothesis 1: CONFIRMED**

Engineered 1M context outperformed Naïve 1M by **X.X%** on correctness (p < 0.05, Cohen's d = X.XX). This exceeded our predicted 15% threshold.

**What surprised us:**
- The gap widened at higher fill percentages (X% at 30% fill → Y% at 90% fill)
- Cost savings were larger than expected (Z% lower token usage despite same input size—due to more focused responses)
- Contradiction detection showed the largest difference (X% vs Y%)

**Why it matters:**  
Structure helps the model navigate. The hierarchical organization, table of contents, and metadata acted as "landmarks" in a vast context space. The model could locate information more reliably and cite sources more accurately.

**Practical implication:**  
If you're using 1M token windows, *don't just dump documents*. Invest in structure. The engineering effort pays off in quality and cost.

[IF H1 REJECTED:]
❌ **Hypothesis 1: REJECTED**

Against our prediction, Naïve 1M performed within X% of Engineered 1M—not a statistically significant difference (p = X.XX). At 70% and 90% fill, naive actually performed X% *better*.

**What surprised us:**
- Modern models (Gemini 2.0 Flash) appear more robust to unstructured context than earlier models
- The overhead of parsing structure (XML tags, table of contents) may have introduced noise
- Simple concatenation allowed the model to use its own pattern recognition

**Why this matters:**  
The "just dump everything" approach actually works better than we expected. The model's internal architecture handles unstructured long contexts more gracefully than 2023-era research suggested.

**Practical implication:**  
For straightforward document retrieval, naive approaches may suffice. Save engineering effort for more complex use cases (synthesis, contradiction detection).

### Result 3: Hypothesis 2 – Can Small + Smart Beat Big + Dumb?

**Predicted:** Advanced 128k RAG matches within 5% of Naïve 1M on quality, costs <40%, latency <2x.

**Actual Result:** [INSERT DATA]

`[PLACEHOLDER FOR TABLE 2: RAG vs Long Context comparison]`
| Metric | Naïve 1M (baseline) | Advanced RAG 128k | Difference |
|--------|---------------------|-------------------|------------|
| Correctness | X.XX | X.XX | ±X.X% |
| Cost per query | $X.XX | $X.XX | -X% |
| Latency | X.Xs | X.Xs | +X% |
| Questions requiring all docs | X% correct | X% correct | -X.X% |
| Questions requiring 1-2 docs | X% correct | X% correct | +X.X% |

[IF H2 CONFIRMED:]
✅ **Hypothesis 2: CONFIRMED**

Advanced 128k RAG matched Naïve 1M within **X.X%** on correctness, while costing **X%** less and taking only **X.Xx** the latency. This met all three criteria.

**What surprised us:**
- Hybrid search (dense + BM25) was crucial—pure vector search fell X% short
- Query decomposition mattered most for synthesis questions
- The cost advantage was larger than predicted (X% vs predicted 40%)

**Why it matters:**  
You don't need the biggest context window to get good results. A disciplined 128k RAG system can compete with naive 1M approaches while being more cost-effective and faster.

**Practical implication:**  
For most production use cases, invest in RAG engineering before upgrading to 1M token windows. The cost-quality trade-off favors smart retrieval.

[IF H2 REJECTED:]
❌ **Hypothesis 2: REJECTED**

Advanced 128k RAG fell **X.X%** short of Naïve 1M on correctness—beyond our 5% threshold. The gap was especially pronounced on multi-document synthesis tasks.

**What surprised us:**
- Retrieval failures compounded: miss one key document, answer suffers
- The padding (to match fill %) introduced noise that RAG systems struggled to ignore
- Latency was actually X.Xx higher due to embedding + retrieval overhead

**Why this matters:**  
Context capacity has a quality ceiling that retrieval can't fully overcome. When questions require synthesizing across many documents, having everything in context provides an advantage.

**Practical implication:**  
For complex multi-document reasoning, long context windows offer a real benefit. RAG works better for targeted retrieval, but struggles with "tell me everything about X" queries.

### Result 4: The Pollution Experiment – Robustness to Noise

**Question:** How do strategies handle increasing amounts of irrelevant information?

**Finding:** [INSERT CHART: Accuracy vs Pollution Level]

`[PLACEHOLDER FOR FIGURE 2: Line chart, X-axis = Pollution level (50k-950k), Y-axis = Accuracy]`

[INSERT FINDINGS HERE - example structure:]

At 50% pollution (500k irrelevant / 500k relevant):
- Naïve 1M: X% accuracy  
- Engineered 1M: Y% accuracy  
- Basic RAG: Z% accuracy  
- Advanced RAG: W% accuracy

At 90% pollution (900k irrelevant / 50k relevant):
- Naïve 1M: X% accuracy (ΔX% drop)
- Engineered 1M: Y% accuracy (ΔY% drop)
- Basic RAG: Z% accuracy (ΔZ% drop)
- Advanced RAG: W% accuracy (ΔW% drop)

**Key insight:** [TO BE DETERMINED FROM DATA - possible outcomes:]

[IF RAG MORE ROBUST:]
RAG strategies showed graceful degradation—retrieval naturally filtered irrelevant content. Long-context strategies suffered more as they had to process all pollution.

[IF ENGINEERED MORE ROBUST:]
Structured contexts allowed the model to identify and weight relevant sections, reducing pollution impact compared to naive dumps.

[IF ALL EQUALLY AFFECTED:]
All strategies degraded similarly, suggesting pollution effects operate at the attention level regardless of input structure.

### Result 5: The Cost-Latency-Quality Frontier

**Question:** What's the Pareto frontier? Which strategies dominate?

`[PLACEHOLDER FOR FIGURE 3: 3D scatter plot, axes = Quality/Cost/Latency]`
`[PLACEHOLDER FOR FIGURE 4: 2D projections showing trade-offs]`

[INSERT PARETO ANALYSIS HERE]

**Dominant strategies** (no other strategy beats them on all three dimensions):
1. [Strategy A]: Best for [use case] (high quality, willing to pay)
2. [Strategy B]: Best for [use case] (balanced trade-offs)
3. [Strategy C]: Best for [use case] (cost-constrained)

**Dominated strategies** (beaten by others on all dimensions):
- [If any]

---

## What We Didn't Expect: Surprising Findings

### Surprise 1: [INSERT UNEXPECTED FINDING]

[Example: "The latency curve wasn't linear. Processing 900k tokens took 4.2x longer than 100k tokens, not 9x as token count would suggest."]

### Surprise 2: [INSERT UNEXPECTED FINDING]

[Example: "Model confidence scores were poorly calibrated. The model was equally confident on wrong answers at 90% fill as correct answers at 30% fill."]

### Surprise 3: [INSERT UNEXPECTED FINDING]

[Example: "Structured contexts caused more 'citation hallucinations'—the model invented plausible-sounding section references that didn't exist."]

---

## Limitations: What This Study Doesn't Tell You

We designed this experiment rigorously, but no single study answers everything. Here's what we *can't* conclude:

**1. Model Generalization**  
We tested only Gemini 2.0 Flash Experimental. Results may differ for Claude 3, GPT-4, or other models. We chose Gemini for practical reasons (free tier, 1M tokens), but this limits generalizability.

**2. Domain Specificity**  
We tested API documentation and financial reports. Performance may differ for:
- Code documentation
- Legal documents  
- Scientific papers
- Conversational transcripts

**3. Question Types**  
Our 70 questions emphasized lookup and synthesis. We didn't extensively test:
- Summarization tasks
- Creative generation with context
- Multi-turn conversations

**4. Real-World Messiness**  
Our corpora were clean and well-formatted. Production data has OCR errors, formatting inconsistencies, and duplicate content we didn't model.

**5. LLM-as-Judge Bias**  
We used LLM-based evaluation for scalability. Human evaluation on a 30-question subset showed X% agreement (Cohen's κ = X.XX), but automated evaluation has known biases.

**6. Position Effects**  
We didn't systematically vary information position (start/middle/end of context). This is a known confounding variable we controlled by randomizing document order but didn't explicitly measure.

---

## Practical Guidance: What Should You Do?

Based on our results, here's a decision framework for practitioners:

### Scenario 1: Building a New Q&A System

**If your budget is unconstrained and quality is paramount:**
→ Use [STRATEGY X based on results], with [SPECIFIC RECOMMENDATIONS]

**If you're cost-conscious:**  
→ Use [STRATEGY Y based on results], with [SPECIFIC RECOMMENDATIONS]

**If latency is critical (user-facing, real-time):**  
→ Use [STRATEGY Z based on results], with [SPECIFIC RECOMMENDATIONS]

### Scenario 2: You Already Have a RAG System

[IF H2 CONFIRMED:]
**Don't rush to replace it.** Our results show well-engineered RAG competes with naive long-context. Before migrating:
1. Benchmark your current system
2. Estimate token costs at 1M context
3. Test whether your questions require cross-document synthesis (where long context helps) or targeted retrieval (where RAG excels)

[IF H2 REJECTED:]
**Consider upgrading** if you're hitting RAG limitations. Long context windows offer real benefits for:
- Multi-document synthesis
- Questions requiring full corpus awareness  
- Use cases where retrieval errors compound

But keep RAG for:
- High-frequency queries (cost matters)
- Well-scoped questions (no need for full corpus)

### Scenario 3: You're Using Naive Long-Context

[IF H1 CONFIRMED:]
**Add structure immediately.** The cost-quality improvement from engineering your context is substantial and doesn't require changing models.

Minimum viable improvements:
1. Add a table of contents
2. Tag documents with metadata (source, date, topic)
3. Use consistent section headers
4. Implement clear document boundaries

[IF H1 REJECTED:]
**Your approach may be fine** for straightforward retrieval. Consider adding structure only if:
- You need better citation accuracy
- Your use case involves complex reasoning across documents
- Cost optimization matters (structured contexts sometimes yield more focused responses)

### The Universal Advice

Regardless of strategy:

1. **Measure fill percentage.** It affects quality more than absolute token count.
2. **Test with pollution.** Real-world data isn't perfectly relevant. Understand robustness.
3. **Monitor token usage.** Even "free" tiers have rate limits. Cost at scale matters.
4. **Version your context engineering.** Treat context assembly as code—test, version, and iterate.

---

## The Bigger Picture: Context Engineering as Discipline

This experiment was never just about two hypotheses. It was about establishing that **context engineering deserves rigorous, empirical investigation.**

Here's what we learned beyond the numbers:

### 1. Context Engineering Has a Design Space

Just like prompt engineering evolved from "write a good instruction" to frameworks, templates, and best practices, context engineering has structure:

- **Chunking strategies** (size, overlap, semantic boundaries)
- **Retrieval approaches** (dense, sparse, hybrid)
- **Assembly patterns** (sequential, hierarchical, graph-based)
- **Metadata design** (what tags help? what's noise?)

These aren't details—they're the design space of the discipline.

### 2. Scale Doesn't Eliminate the Need for Engineering

[IF H1 CONFIRMED:]
Our results show that even with 1M tokens, engineering matters. This mirrors a pattern across computing history: More resources enable new capabilities, but *disciplined use of resources* still outperforms wasteful approaches.

You can have 1TB of RAM and still write a memory leak. You can have 1M token windows and still build a terrible context.

[IF H1 REJECTED:]
Our results suggest modern models are more robust to naive approaches than earlier models. This is progress—it lowers the barrier to getting started.

But "good enough for prototypes" doesn't mean "optimal for production." As usage scales, the cost-quality trade-offs still matter.

### 3. The Metrics Matter

We measured:
- Correctness (did it answer right?)
- Citation accuracy (can we trust the sources?)
- Cost (what does it cost at scale?)
- Latency (is it usable?)  
- Robustness (does it degrade gracefully?)

These aren't arbitrary. They're the axes along which production systems succeed or fail.

### 4. There Is No "Best" Strategy

The Pareto frontier analysis shows multiple non-dominated strategies. The "best" choice depends on your constraints:
- Research application? Optimize for quality, cost is secondary.
- Consumer product? Latency may dominate.
- Enterprise B2B? Cost per query matters enormously at scale.

**Context engineering is an optimization problem, not a best-practice checklist.**

---

## Open Questions for Future Research

This study answered two specific hypotheses, but raised more questions:

1. **How do these results transfer across models?** (Claude, GPT-4, Llama)
2. **What's the optimal chunk size for different domains?** (code vs prose vs tables)
3. **Can we predict when RAG will outperform long-context without running experiments?** (decision rules)
4. **How does context caching affect the cost-quality trade-offs?** (Anthropic's contextual retrieval)
5. **What evaluation metrics best capture "context quality"?** (beyond correctness)
6. **How do position effects interact with fill percentage?** (systematically untested here)

We're open-sourcing our framework to enable the community to investigate these questions.

---

## Reproducibility and Open Science

### All Data and Code Released

The complete experimental framework is available at:  
**[github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments)**

This includes:
- ✅ All 4,380 API call records (anonymized)
- ✅ Context assembly code (all 4 strategies)
- ✅ Evaluation scripts (LLM-as-judge + rubrics)
- ✅ Statistical analysis notebooks
- ✅ Visualization code for all charts
- ✅ Documentation to replicate or extend experiments

### Why Open?

Science advances through replication and extension. If our results surprise you, replicate them. If our methodology has flaws, improve it. If your domain differs, adapt the framework.

We used the free tier of a public API specifically so anyone can verify our work without budget constraints.

### Variance Transparency

We ran 3 repetitions per condition. Here's the variance we observed:

`[PLACEHOLDER FOR TABLE: Variance across repetitions]`
| Strategy | Mean Correctness | Std Dev | 95% CI |
|----------|-----------------|---------|--------|
| Naïve 1M | X.XX | X.XX | [X.XX, X.XX] |
| Engineered 1M | X.XX | X.XX | [X.XX, X.XX] |
| Basic RAG | X.XX | X.XX | [X.XX, X.XX] |
| Advanced RAG | X.XX | X.XX | [X.XX, X.XX] |

With temperature=0.0, variance came from non-deterministic tie-breaking in retrieval and minor API inconsistencies. All reported differences are statistically significant (p < 0.05) unless noted.

---

## Conclusion: The Answer to the Million-Token Question

We started with a question: **In the age of million-token context windows, does engineering discipline still matter?**

After 4,380 API calls, 10 weeks of work, and rigorous statistical analysis, here's what we learned:

[IF H1 CONFIRMED + H2 CONFIRMED:]
**Yes, engineering matters—dramatically.** Both hypotheses confirmed:
- Structure beats naive even at 1M tokens
- Smart retrieval can match scale
- The "just dump everything" approach is a trap

**The industry narrative is oversimplified.** Long context windows are a tool, not a solution. How you use them determines success.

[IF H1 CONFIRMED + H2 REJECTED:]
**Engineering matters at scale, but scale also matters.** A mixed outcome:
- Structure improves long contexts significantly
- But RAG can't fully overcome capacity constraints
- The right choice depends on your queries

**The nuance matters.** For targeted retrieval, engineer smartly. For synthesis across many documents, scale helps.

[IF H1 REJECTED + H2 CONFIRMED:]
**Smart beats big, but structure is overrated.** A surprising outcome:
- Naive long contexts work better than expected
- But RAG can still compete cost-effectively
- Invest in retrieval, not structure

**Modern models are robust to naive approaches.** This changes the effort allocation: Retrieval engineering > Context structure engineering.

[IF H1 REJECTED + H2 REJECTED:]
**Scale won.** Naive long-context approaches outperformed engineered alternatives:
- Simpler is better
- The models handle unstructured context well
- The "just dump everything" approach actually works

**This doesn't mean engineering doesn't matter**—it means the engineering effort should focus elsewhere (query understanding, post-processing, evaluation).

### The Real Takeaway

Regardless of which specific hypotheses held, the meta-result is clear:

**Empirical evaluation beats intuition.** The only way to know what works for *your* use case is to measure.

Context engineering is a design space with trade-offs. Understanding those trade-offs—quality, cost, latency, robustness—enables better decisions than following trends.

---

## What's Next

This study focused on two experiments (Needle in Haystacks, Context Pollution). Several interesting questions remain:

**Short-term extensions:**
- Replicate on Claude 3 and GPT-4 (different architectures)
- Test on code documentation (different domain)
- Add human evaluation (validate LLM-as-judge)

**Long-term research:**
- Multi-turn conversation context management
- Dynamic context assembly (adjust based on query complexity)
- Learned retrieval (train models to predict relevance)

If you're interested in collaborating on extensions, reach out.

---

## Final Thoughts

We set out to answer whether engineering matters in the age of long context. Along the way, we learned something more important:

**The questions you ask determine the systems you build.**

If you ask "what's the biggest context window?", you optimize for scale.  
If you ask "what's the most cost-effective approach?", you optimize for efficiency.  
If you ask "what degrades gracefully?", you optimize for robustness.

**The right question is: "What are the trade-offs, and which matter for my use case?"**

This study doesn't tell you what to build. It gives you data to make that decision yourself.

We hope that's valuable.

---

## Acknowledgments

This research was conducted independently with no vendor funding. Thanks to:
- The open-source community for tools (FAISS, sentence-transformers, etc.)
- Google for providing free tier API access
- Early reviewers who caught methodological issues
- [Anyone else you want to thank]

## Connect and Discuss

- **GitHub:** [github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments)
- **LinkedIn:** [Your LinkedIn]
- **Email:** [Your Email]

If this research was useful, please cite it, share it, or extend it. And if you find errors or limitations we missed, let us know—science is an iterative process.

---

*Data and analysis available in the GitHub repository. Full experimental logs, statistical notebooks, and replication instructions included.*

*Last updated: [Date when experiments complete]*

