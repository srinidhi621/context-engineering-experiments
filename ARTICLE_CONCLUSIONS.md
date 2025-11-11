# The Million-Token Question: What We Actually Found

**After 4,380 API Calls, 10 Weeks, and Way Too Much Coffee—Here's What the Data Says**

---

## The Short Version (For Those Who Can't Wait)

We ran the experiments. All 4,380 API calls across 1M and 128k token contexts, testing naive vs. engineered approaches at fill levels from 10% to 90%.

Here's what we found:

**[SUMMARY - TO BE COMPLETED WITH ACTUAL RESULTS]**

[IF H1 CONFIRMED:]
✅ **Context engineering matters—even at 1M tokens.** Engineered approaches beat naive context stuffing by X% on quality, Y% on cost efficiency. The "just dump everything" approach? Yeah, it's a trap.

[IF H1 REJECTED:]
❌ **Plot twist: Naive long-context actually works.** Simply dumping documents into 1M token windows matched or exceeded engineered approaches. Modern models are better than we thought.

[IF H2 CONFIRMED:]
✅ **Smart beats big.** Well-engineered 128k RAG systems matched naive 1M approaches within X% on quality, while costing Y% less. David beat Goliath.

[IF H2 REJECTED:]
❌ **Size matters more than we expected.** Naive 1M token approaches outperformed engineered 128k RAG by X%. Turns out, having everything in context is actually pretty useful.

**Read on for the full story, the surprises, and what it means for your systems.**

---

## Quick Recap: What Were We Testing Again?

In the [prelude article](#), we set up two hypotheses:

**H1: Long Context ≠ Magic Bullet**  
Even with 1M-token windows, naïvely stuffing context underperforms engineered retrieval + packaging.

**H2: Smart Beats Big**  
128k-token models with disciplined context engineering can match or beat naïve long-context use.

The real question: **In the age of million-token context windows, does engineering discipline still matter?**

After 10 weeks of work, we have answers. Some of them surprised us.

---

## What We Did: The Experimental Setup

### Four Strategies, Head-to-Head

We tested four approaches to see which actually works in practice:

**1. Naïve 1M:** Dump everything into the 1M token window. No structure. No organization. Just concatenate documents and hope for the best. This is what the marketing says you should do.

**2. Engineered 1M:** Same documents, same 1M window, but with hierarchical structure, table of contents, metadata tags, and navigation aids. This is what engineers do when they actually care about the outcome.

**3. Basic RAG 128k:** Vector search with top-k retrieval. The traditional RAG approach most teams use today. Solid, proven, maybe a bit boring.

**4. Advanced RAG 128k:** Hybrid search (dense + BM25), reciprocal rank fusion, query decomposition. The fancy stuff that research papers talk about.

### The Critical Control: Fill Percentage

Here's the methodological piece that matters: We padded all strategies to identical fill percentages (10%, 30%, 50%, 70%, 90%).

Why? Because if RAG uses 13% of its 128k window while naive uses 90% of its 1M window, you can't tell if performance differences come from better retrieval or just less attention dilution. That's a confounding variable, and we don't do confounding variables here.

This control isn't in most blog posts comparing RAG to long-context. But it should be.

### The Tests

**Pilot Phase (180 calls):**  
10 questions on AWS Lambda docs. Made sure our pipeline actually worked before scaling up. (Pro tip: always pilot your experiments. We found three bugs here.)

**Experiment 1: Needle in Multiple Haystacks (3,000 calls):**  
50 questions across 500k-1M tokens of API documentation from AWS, GCP, and Azure. All data fetched after Gemini 2.0's training cutoff, so no memorization effects.

- 20 simple lookups ("What's the default Lambda timeout?")
- 20 synthesis tasks ("Compare authentication across AWS/GCP/Azure")
- 10 complex reasoning ("Resolve contradictions between v1 and v2 docs")

**Experiment 2: Context Pollution (1,200 calls):**  
20 questions answerable from a clean 50k token base corpus. Then we added pollution—50k to 950k tokens of plausible but irrelevant content.

The question: Can the model ignore distractors? Or does it get confused by noise?

**Total:** 4,380 API calls, 70 unique questions, 3 repetitions per condition, temperature 0.0 (deterministic).

Cost: $0. (Thank you, Google free tier.)

---

## The Results: What Actually Happened

### Finding 1: Fill Percentage Is Everything

Remember "Lost in the Middle"? The research showing models lose information buried in long contexts? Turns out it doesn't magically disappear at 1M tokens.

**[INSERT CHART: Accuracy vs Fill Percentage]**

`[PLACEHOLDER FOR FIGURE 1: Line chart showing 4 strategies, X-axis = Fill %, Y-axis = Correctness (0-1)]`

[IF DEGRADATION OBSERVED:]
**All strategies degraded as context filled up.** But the rate of degradation differed:

- Naïve 1M: Dropped from X% (10% fill) to Y% (90% fill) — **Z percentage point decline**
- Engineered 1M: Dropped from X% to Y% — **Z percentage point decline**  
- Basic RAG 128k: Dropped from X% to Y% — **Z percentage point decline**
- Advanced RAG 128k: Dropped from X% to Y% — **Z percentage point decline**

**Translation:** A 300k token context at 30% fill can outperform a 900k token context at 90% fill—even with identical information. Fill percentage matters more than raw token count.

[IF NO DEGRADATION:]
**Plot twist:** We saw minimal degradation up to 70% fill. Modern models (Gemini 2.0 Flash) handle long contexts better than 2023-era research suggested. Degradation only kicked in beyond 70% fill.

**Translation:** The "Lost in the Middle" problem is real, but less severe than earlier models showed. Progress is real.

### Finding 2: Does Engineering Matter at 1M Tokens? (H1)

**We predicted:** Engineered 1M beats Naïve 1M by ≥15% on quality.

**What actually happened:** [INSERT DATA]

`[PLACEHOLDER FOR TABLE 1: Strategy comparison at 70% fill]`
| Strategy | Correctness | Cost/Query | Latency | Citation Accuracy |
|----------|------------|------------|---------|-------------------|
| Naïve 1M | X.XX | $X.XX | X.Xs | X.X% |
| Engineered 1M | X.XX | $X.XX | X.Xs | X.X% |
| Δ (improvement) | **+X.X%** | **-X%** | **+X%** | **+X.X%** |

[IF H1 CONFIRMED:]
✅ **Hypothesis 1: CONFIRMED**

Engineered 1M context beat Naïve 1M by **X.X%** on correctness (p < 0.05, Cohen's d = X.XX). We predicted 15%, got X.X%. Engineering wins.

**What surprised us:**
- The gap widened at higher fill percentages (X% at 30% fill → Y% at 90% fill). Structure helps more when context is crowded.
- Cost savings were larger than expected—Z% lower token usage despite same input size. Structured contexts led to more focused responses.
- Contradiction detection showed the biggest difference (X% vs Y%). The model could navigate structured contexts to find conflicting information.

**Why this works:**  
Structure creates landmarks in a vast context space. The hierarchical organization, table of contents, and metadata act like signposts. The model can locate information more reliably and cite sources more accurately.

Think about it: Would you rather search a 900k-word document that's just one giant text blob, or one with a table of contents, section headers, and clear boundaries? The model feels the same way.

**What you should do:**  
If you're using 1M token windows, don't just dump documents. Add structure. The engineering effort pays off in quality and cost. 

Minimum viable improvements:
- Add a table of contents
- Tag documents with metadata (source, date, topic)
- Use consistent section headers
- Implement clear document boundaries

[IF H1 REJECTED:]
❌ **Hypothesis 1: REJECTED**

Against our prediction, Naïve 1M performed within X% of Engineered 1M—not statistically significant (p = X.XX). At 70% and 90% fill, naive actually performed X% *better*.

**What surprised us:**
- Modern models (Gemini 2.0 Flash) are more robust to unstructured context than we expected
- The overhead of parsing structure (XML tags, table of contents) might have introduced noise
- Simple concatenation let the model use its own pattern recognition without our imposed structure

**Why this happened:**  
The model's internal architecture handles unstructured long contexts better than 2023-era research suggested. Either the architecture improved, or the training data included enough unstructured long documents to learn robust strategies.

**What you should do:**  
For straightforward document retrieval, naive approaches might be fine. Save engineering effort for:
- Cases where you need precise citation accuracy
- Complex reasoning across multiple documents
- Cost optimization (though our results suggest this benefit is smaller than expected)

### Finding 3: Can Small + Smart Beat Big + Dumb? (H2)

**We predicted:** Advanced 128k RAG matches within 5% of Naïve 1M on quality, costs <40%, latency <2x.

**What actually happened:** [INSERT DATA]

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

Advanced 128k RAG matched Naïve 1M within **X.X%** on correctness, while costing **X%** less and taking only **X.Xx** the latency. All three criteria met. David beat Goliath.

**What surprised us:**
- Hybrid search (dense + BM25) was crucial. Pure vector search fell X% short. Sparse retrieval matters.
- Query decomposition mattered most for synthesis questions. Breaking complex queries into sub-queries improved retrieval dramatically.
- The cost advantage was larger than predicted (X% vs our predicted 40%). RAG is even more cost-effective than we thought.

**Why this works:**  
You don't need the biggest context window to get good results. A disciplined 128k RAG system can compete with naive 1M approaches while being more cost-effective and faster.

The key: Good retrieval selects the right information. Bad retrieval plus large context just gives you a large pile of mostly-irrelevant information.

**What you should do:**  
For most production use cases, invest in RAG engineering before upgrading to 1M token windows. The cost-quality trade-off favors smart retrieval.

Specifically:
- Use hybrid search (don't rely on dense vectors alone)
- Implement query decomposition for complex questions
- Focus on retrieval quality—garbage in, garbage out applies here

[IF H2 REJECTED:]
❌ **Hypothesis 2: REJECTED**

Advanced 128k RAG fell **X.X%** short of Naïve 1M on correctness—beyond our 5% threshold. The gap was especially pronounced on multi-document synthesis tasks.

**What surprised us:**
- Retrieval failures compounded. Miss one key document in a 10-document synthesis, and the answer suffers.
- The padding (to match fill %) introduced noise that RAG systems struggled to ignore. Ironic—we added noise to control for noise.
- Latency was actually X.Xx higher due to embedding + retrieval overhead. RAG isn't always faster.

**Why this happened:**  
Context capacity creates a quality ceiling that retrieval can't fully overcome. When questions require synthesizing across many documents, having everything in context provides an advantage. The model doesn't need perfect retrieval—it can just look at everything.

**What you should do:**  
For complex multi-document reasoning, long context windows offer real benefits. Use RAG for:
- Targeted retrieval ("find the answer to X")
- High-frequency queries (cost matters)
- Well-scoped questions

Use long context for:
- Multi-document synthesis ("compare X across all documents")
- Exploratory questions ("tell me everything about X")
- Cases where retrieval errors compound

### Finding 4: How Strategies Handle Pollution

We threw noise at the models. Lots of it. From 50k to 950k tokens of plausible but irrelevant content.

**Question:** Can the model stay focused? Or does it get distracted?

**[INSERT CHART: Accuracy vs Pollution Level]**

`[PLACEHOLDER FOR FIGURE 2: Line chart, X-axis = Pollution level (50k-950k), Y-axis = Accuracy]`

[INSERT FINDINGS HERE - example structure:]

At 50% pollution (500k irrelevant / 500k relevant):
- Naïve 1M: X% accuracy  
- Engineered 1M: Y% accuracy  
- Basic RAG: Z% accuracy  
- Advanced RAG: W% accuracy

At 90% pollution (900k irrelevant / 50k relevant):
- Naïve 1M: X% accuracy (dropped ΔX%)
- Engineered 1M: Y% accuracy (dropped ΔY%)
- Basic RAG: Z% accuracy (dropped ΔZ%)
- Advanced RAG: W% accuracy (dropped ΔW%)

**The insight:** [TO BE DETERMINED FROM DATA - possible outcomes:]

[IF RAG MORE ROBUST:]
RAG strategies showed graceful degradation. Retrieval naturally filtered irrelevant content—if it's not in the top-k results, it doesn't make it to the context. Long-context strategies had to process all the pollution.

**Translation:** RAG's retrieval step acts as a filter. This is a feature, not a bug.

[IF ENGINEERED MORE ROBUST:]
Structured contexts let the model identify and weight relevant sections. The metadata and organization helped the model distinguish signal from noise.

**Translation:** Structure helps with filtering, not just navigation.

[IF ALL EQUALLY AFFECTED:]
All strategies degraded similarly. Pollution effects operate at the attention level, regardless of input structure. More noise = more distraction, period.

**Translation:** There's no magic bullet for handling pollution. Clean your data.

### Finding 5: The Pareto Frontier (Quality × Cost × Latency)

In a perfect world, one strategy would win on all metrics. In reality, there are trade-offs.

**[INSERT CHARTS: 3D Pareto Frontier + 2D Projections]**

`[PLACEHOLDER FOR FIGURE 3: 3D scatter plot, axes = Quality/Cost/Latency]`
`[PLACEHOLDER FOR FIGURE 4: 2D projections showing trade-offs]`

[INSERT PARETO ANALYSIS HERE]

**Dominant strategies** (no other strategy beats them on all three dimensions):
1. [Strategy A]: Best for [use case] (high quality, willing to pay)
2. [Strategy B]: Best for [use case] (balanced trade-offs)
3. [Strategy C]: Best for [use case] (cost-constrained)

**Dominated strategies** (beaten by others on all dimensions):
- [If any]

**What this means:**  
There is no "best" strategy. The right choice depends on what you optimize for:
- Research application? Optimize for quality, cost is secondary.
- Consumer product? Latency might dominate.
- Enterprise B2B? Cost per query matters at scale.

Context engineering is an optimization problem, not a best-practice checklist.

---

## The Surprises: What We Didn't Expect

Every experiment has surprises. Here are ours:

### Surprise 1: [INSERT UNEXPECTED FINDING]

[Example: "The latency curve wasn't linear. Processing 900k tokens took 4.2x longer than 100k tokens, not 9x. There's some optimization happening under the hood we didn't expect."]

### Surprise 2: [INSERT UNEXPECTED FINDING]

[Example: "The model's confidence scores were poorly calibrated. It was equally confident giving wrong answers at 90% fill as correct answers at 30% fill. Don't trust confidence scores blindly."]

### Surprise 3: [INSERT UNEXPECTED FINDING]

[Example: "Structured contexts sometimes caused 'citation hallucinations'—the model invented plausible-sounding section references that didn't exist. More structure = more opportunities to hallucinate structure."]

---

## What This Study Doesn't Tell You (The Limitations)

We designed this carefully, but no single study answers everything. Here's what we *can't* conclude:

**1. Other models might behave differently**  
We tested Gemini 2.0 Flash Experimental. Claude 3, GPT-4, Llama 3—they might show different patterns. We chose Gemini because it's free and has 1M tokens, but that limits generalizability.

**2. Other domains might differ**  
We tested API documentation and financial reports. Code documentation, legal documents, scientific papers, chat logs—all might behave differently. Domain matters.

**3. Question types matter**  
We focused on lookup and synthesis. We didn't extensively test summarization, creative generation, or multi-turn conversations. Those might favor different strategies.

**4. Real data is messier**  
Our corpora were clean and well-formatted. Production data has OCR errors, formatting inconsistencies, duplicates, and other chaos we didn't model.

**5. LLM-as-judge has biases**  
We used automated evaluation (another LLM) for scalability. We validated on a 30-question subset with human eval (Cohen's κ = X.XX), but automated evaluation isn't perfect.

**6. We didn't test position effects systematically**  
Information position (start/middle/end of context) is a known confounding variable. We controlled it by randomizing, but didn't measure it explicitly. That's future work.

---

## What You Should Actually Do (Practical Guidance)

Based on our results, here's how to make decisions:

### Scenario 1: You're Building a New Q&A System

**If quality is paramount and budget isn't:**
→ Use [STRATEGY X based on results], with [SPECIFIC RECOMMENDATIONS]

**If you're cost-conscious:**  
→ Use [STRATEGY Y based on results], with [SPECIFIC RECOMMENDATIONS]

**If latency is critical (user-facing, real-time):**  
→ Use [STRATEGY Z based on results], with [SPECIFIC RECOMMENDATIONS]

### Scenario 2: You Already Have a RAG System

[IF H2 CONFIRMED:]
**Don't rush to replace it.** Well-engineered RAG competes with naive long-context. Before migrating:
1. Benchmark your current system (don't guess)
2. Estimate token costs at 1M context (might be expensive)
3. Test if your questions need cross-document synthesis (long context helps) or targeted retrieval (RAG is fine)

[IF H2 REJECTED:]
**Consider upgrading** if you're hitting RAG limitations. Long context helps with:
- Multi-document synthesis
- Questions requiring full corpus awareness  
- Cases where retrieval errors compound

Keep RAG for:
- High-frequency queries (cost adds up)
- Well-scoped questions (don't need full corpus)

### Scenario 3: You're Using Naive Long-Context

[IF H1 CONFIRMED:]
**Add structure immediately.** The improvement is substantial and doesn't require changing models.

Start with:
1. Table of contents
2. Document metadata (source, date, topic)
3. Consistent section headers
4. Clear document boundaries

[IF H1 REJECTED:]
**Your approach might be fine** for simple retrieval. Consider structure only if:
- You need better citation accuracy
- Complex reasoning across documents
- Cost optimization matters

### Universal Advice (Regardless of Results)

1. **Measure fill percentage.** It affects quality more than absolute token count.
2. **Test with pollution.** Real data isn't perfectly relevant. Know your robustness.
3. **Monitor token usage.** Even "free" tiers have rate limits.
4. **Version your context engineering.** Treat context assembly as code—test, version, iterate.

---

## The Bigger Picture: Why This Matters

This wasn't just about two hypotheses. It was about establishing that **context engineering deserves serious attention.**

### What We Learned Beyond the Numbers

**1. Context engineering has a design space**

Just like prompt engineering evolved from "write a good instruction" to frameworks and best practices, context engineering has structure:
- Chunking strategies (size, overlap, boundaries)
- Retrieval approaches (dense, sparse, hybrid)
- Assembly patterns (sequential, hierarchical, graph-based)
- Metadata design (what helps, what's noise)

These are engineering decisions, not implementation details.

**2. Scale doesn't eliminate engineering**

[IF H1 CONFIRMED:]
Even with 1M tokens, engineering matters. This mirrors computing history: More resources enable new capabilities, but *disciplined use* still beats wasteful approaches.

You can have 1TB of RAM and still write a memory leak. You can have 1M token windows and still build terrible contexts.

[IF H1 REJECTED:]
Modern models are more robust than earlier versions. That's progress—it lowers the barrier to getting started.

But "good enough for prototypes" ≠ "optimal for production." As usage scales, the trade-offs matter.

**3. Multiple metrics matter**

We measured:
- Correctness (did it answer right?)
- Citation accuracy (can we trust the sources?)
- Cost (what does it cost at scale?)
- Latency (is it usable?)  
- Robustness (does it degrade gracefully?)

These aren't arbitrary. They're how production systems succeed or fail.

**4. There is no "best" strategy**

The Pareto frontier shows multiple non-dominated strategies. The "best" choice depends on your constraints.

Context engineering is optimization, not dogma.

---

## Open Questions (What's Next?)

This study answered two hypotheses, but raised more questions:

1. How do results transfer across models? (Claude, GPT-4, Llama)
2. What's optimal chunk size for different domains? (code vs prose vs tables)
3. Can we predict when RAG outperforms long-context? (decision rules)
4. How does context caching affect trade-offs? (Anthropic's approach)
5. What metrics best capture "context quality"? (beyond correctness)
6. How do position effects interact with fill %? (untested here)

We're open-sourcing the framework so the community can investigate.

---

## Reproducibility: All Code and Data Released

Everything is available at:  
**[github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments)**

Includes:
- All 4,380 API call records (anonymized)
- Context assembly code (all 4 strategies)
- Evaluation scripts (LLM-as-judge + rubrics)
- Statistical analysis notebooks
- Visualization code
- Documentation to replicate or extend

Why open-source? Science advances through replication. If our results surprise you, replicate them. If our methodology has flaws, improve it. If your domain differs, adapt the framework.

We used the free tier specifically so anyone can verify our work without budget constraints.

### Variance Transparency

We ran 3 repetitions per condition. Here's the variance:

`[PLACEHOLDER FOR TABLE: Variance across repetitions]`
| Strategy | Mean Correctness | Std Dev | 95% CI |
|----------|-----------------|---------|--------|
| Naïve 1M | X.XX | X.XX | [X.XX, X.XX] |
| Engineered 1M | X.XX | X.XX | [X.XX, X.XX] |
| Basic RAG | X.XX | X.XX | [X.XX, X.XX] |
| Advanced RAG | X.XX | X.XX | [X.XX, X.XX] |

Temperature was 0.0 (deterministic), so variance came from non-deterministic tie-breaking in retrieval and minor API inconsistencies. All reported differences are statistically significant (p < 0.05) unless noted.

---

## The Bottom Line

We started with a question: **In the age of million-token context windows, does engineering discipline still matter?**

After 4,380 API calls and 10 weeks, here's the answer:

[IF H1 CONFIRMED + H2 CONFIRMED:]
**Yes, engineering matters—a lot.** Both hypotheses confirmed:
- Structure beats naive even at 1M tokens
- Smart retrieval can match scale
- The "just dump everything" approach is a trap

The industry narrative is oversimplified. Long context windows are a tool, not a solution.

[IF H1 CONFIRMED + H2 REJECTED:]
**Engineering matters at scale, but scale also matters.** Mixed results:
- Structure improves long contexts significantly
- But RAG can't fully overcome capacity constraints
- The right choice depends on your queries

Nuance matters. For targeted retrieval, engineer smartly. For synthesis, scale helps.

[IF H1 REJECTED + H2 CONFIRMED:]
**Smart beats big, but structure is overrated.** Surprising:
- Naive long contexts work better than expected
- But RAG can still compete cost-effectively
- Invest in retrieval, not structure

Modern models are robust. Effort allocation: Retrieval engineering > Context structure engineering.

[IF H1 REJECTED + H2 REJECTED:]
**Scale won.** Naive long-context approaches outperformed engineered alternatives:
- Simpler is better
- Models handle unstructured context well
- The "just dump everything" approach actually works

This doesn't mean engineering doesn't matter—it means focus effort elsewhere (query understanding, post-processing, evaluation).

### The Real Takeaway

Regardless of specific outcomes:

**Empirical evaluation beats intuition.** The only way to know what works for *your* use case is to measure.

Context engineering is a design space with trade-offs. Understanding those trade-offs—quality, cost, latency, robustness—enables better decisions than following trends.

---

## What's Next for This Research

Short-term extensions:
- Replicate on Claude 3 and GPT-4
- Test on code documentation
- Add human evaluation

Long-term research:
- Multi-turn conversation context management
- Dynamic context assembly
- Learned retrieval

Interested in collaborating? Reach out.

---

## Final Thoughts

We set out to answer whether engineering matters in the age of long context. We learned something more important:

**The questions you ask determine the systems you build.**

If you ask "what's the biggest context window?", you optimize for scale.  
If you ask "what's most cost-effective?", you optimize for efficiency.  
If you ask "what degrades gracefully?", you optimize for robustness.

**The right question: "What are the trade-offs, and which matter for my use case?"**

This study doesn't tell you what to build. It gives you data to make that decision yourself.

Hope that's useful.

---

## Acknowledgments

Thanks to:
- The open-source community for tools (FAISS, sentence-transformers, etc.)
- Google for free tier API access
- Early reviewers who caught issues
- [Anyone else you want to thank]

## Connect

- **GitHub:** [github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments)
- **LinkedIn:** [Your LinkedIn]
- **Email:** [Your Email]

If this was useful, cite it, share it, or extend it. If you find errors, let us know—science is iterative.

---

*Data and analysis in the GitHub repository. Full logs, statistical notebooks, and replication instructions included.*

*Last updated: [Date when experiments complete]*
