# The Million-Token Question: Does More Context Actually Make LLMs Smarter?

**An Empirical Investigation into Context Engineering for Long-Context Language Models**

---

## The Decision

The reasons to write this and then build the subsequent codebase were motivated by two things:
1. In a specific conversation with a team member, we were discussing how a 128k context window can limit what we can do. I tried to make the point that a well engineered 128k context window can stand up to a 1M context window, all other variables being equal.
2. There are models out there with 1 million and 2 million tokens, and there is an inherent curiosity to see if these actually made a difference to shipping a real product.

Now, back in the grand old days pre-gen AI, there was this notion amongst practitioners that the model (prediction, classification etc) itself was a very small part of what made a good system. I believe the same can be said for building systems that use LLMs - the large majority of work lies in the data, the problem definition, and the engineering of the system.

So, I thought why don't we actually test it out—Google Gemini has a fairly capable model that can be used for free, plus Cursor and Claude came out with new versions of their coding tools. So we could plan and run some experiments and figure out what this context engineering actually means for building systems that use LLMs.

For context, 100k tokens is about the size of a small book. With 1M tokens you could fit nearly all of the 7 Harry Potter books.  

So let's ask the questions that we are actually testing:

### Question 1: Does adding 900k tokens into a context window actually work better than a well-engineered 100k token context?

The "Lost in the Middle" research (Liu et al., 2023) showed that models struggle to recall information buried in long contexts at 32k tokens. **Does the problem magically disappear at 1M tokens? Would be nice if it did—save us all a whole lot of trouble.** Or does it get worse?

### Question 2: Can a smaller context window with disciplined engineering match or beat a naive long-context approach?

If you have 128k tokens and use them wisely across retrieval, ranking, structure, etc—can you compete with someone dumping 1M tokens of unstructured text?

### Question 3: What about cost, latency, and robustness?

Even if long-context *works*, is it practical?
- Processing 1M tokens costs 50x more than 20k tokens
- Latency scales with context size
- What happens when 50% of your context is irrelevant noise?

These aren't just technical questions. **They're architecture decisions affecting millions of dollars and user experience.**

If you're still reading, you're probably wondering what we're actually testing. Let's get down to it.

## What We're Actually Testing


### Hypothesis 1: Long Context ≠ Magic Bullet

> *"Even with 1M-token windows, naïvely stuffing context underperforms engineered retrieval + packaging."*

**Concrete prediction:**
- A well engineered 1M context beats naïve 1M context by ≥15% on quality
- At high context pollution levels (≥50% irrelevant content—the information kind, not the air quality kind), the well engineered context maintains >90% accuracy vs <70% for naïve
- Cost per query is ≥20% lower for engineered approaches (this one should be obvious) 

**Translation:** More tokens don't automatically mean better results. How you organize those tokens matters; even at massive scale.

### Hypothesis 2: Smart Beats Big

> *"128k-token models, with disciplined context engineering, can match or beat naïve long-context use on practical tasks."*

**Concrete prediction:**
- A well engineered 128k RAG matches within 5% of naïve 1M on quality
- And costs <40% of naïve 1M per query  
- And has <2x latency of naïve 1M (this one is a bit more interesting)

**Translation:** You might not need the biggest context window. Smart beats big.

---

## Why These Hypotheses Matter

If **H1 is true**, it means the game hasn't changed as much as vendors claim. Context engineering still matters. The "just dump everything" approach is a trap.

If **H2 is true**, it means smaller, well-engineered systems can compete with bigger, naive ones. This changes cost models, architecture decisions, and vendor selection.

If **both are false**, it means long-context models with naive approaches genuinely work better. That would be surprising—and valuable to know.

**Either way, we get data instead of opinions.**

## Why Now? The Industry Moment

We're living through a remarkable shift in LLM capabilities. In just 18 months, context windows exploded from 4k tokens (GPT-3) to 32k (GPT-4), then 128k, and now 1-2 million tokens for models like Gemini 1.5 Pro and Claude 3.

The narrative has shifted from "be surgical with your context" to "just dump everything in." The marketing is confident. The blog posts are enthusiastic.

**But has anyone actually measured whether this works?** That's what we're doing here.

Another interesting experiment would be to see how a smaller, less capable model (say GPT-4o) fares against a larger model (GPT-5). This test would pit context length against model size and capability (plus ~2 years of model development). But we'll save that for another time.

**A side note on measuring "quality":** This is tricky. We'll use the model's ability to answer questions accurately, cite sources correctly, and synthesize information from multiple documents. "Correctness" is a good proxy for quality, but it's not the only metric.

---

## The Experiment Design: Isolating What Actually Matters

Here's the challenge: How do you fairly compare a 1M token context with a 128k token context? If you just test them as-is, you're confounding *context engineering quality* with *attention dilution effects*. In other words, if the RAG approach uses 13% of its 128k window while the naive approach uses 90% of its 1M window, you can't tell if performance differences come from better retrieval or just less attention dilution.

**Our solution:** Pad all contexts to the same fill percentage.

```
At 70% fill (700k tokens):
- Naïve 1M:  700k tokens (documents + padding)
- RAG 128k:  Retrieve 90k relevant + pad to 700k with irrelevant content

Result: Both strategies face identical attention strain. 
Differences now reflect engineering quality, not fill %.
```

### Four Strategies, Five Fill Levels, Two Experiments

**Strategies being tested:**

1. **Naïve 1M:** Sequential document concatenation. No structure. Just dump and pray.

2. **Engineered 1M:** Hierarchical structure with table of contents, metadata, navigation aids. Same documents, organized.

3. **Basic RAG 128k:** Vector search, top-k retrieval, focused context assembly.

4. **Advanced RAG 128k:** Hybrid search (dense + sparse), reciprocal rank fusion, query decomposition.

**Fill levels:** 10%, 30%, 50%, 70%, 90% (from barely-used to nearly-full)

**Experiments:**

- **Experiment 1: Needle in Multiple Haystacks** - 50 questions across 500k-1M tokens of API documentation (AWS, GCP, Azure). Tests multi-document reasoning and synthesis. Data fetched is outside of the cut-off date of the Gemini 2.0 Flash model.

- **Experiment 2: Context Pollution** - 20 questions answerable from 50k base tokens, with pollution ramping from 50k to 950k irrelevant content. Tests robustness to distraction.

**Total API calls:** 4,380 (10 pilot + 3,000 + 1,200)  
**Model:** Gemini 2.0 Flash Experimental (free tier, 1M token window)  
**Repetitions:** 3 per condition (statistical power)  
**Temperature:** 0.0 (deterministic, reproducible)

### What We're Measuring

**Quality metrics:**
- Correctness: Did the model answer accurately? (0-1 scale, LLM-as-judge)
- Citation accuracy: Are claims grounded in provided context?
- Completeness: For synthesis questions, did it integrate multiple sources?

**Efficiency metrics:**
- Cost per query (token usage)
- Latency (total response time)
- Robustness (performance degradation vs pollution level)

**The Pareto frontier:**
We'll map the 3D trade-off space (quality × cost × latency) to find dominant strategies—approaches where no other strategy beats them on all three dimensions.

---

## What We're NOT Testing (And Why)

**We're not testing:**
- Multiple model versions within the same provider (Gemini Flash vs Pro models) or model families (OpenAI vs Anthropic vs Google). We're focusing on Gemini 2.0 Flash across all experiments for consistency and control. 
- Position effects: Placement of the "needle" in the context haystack (interesting, but secondary to core hypotheses)

---

### The Questions That Follow

If our hypotheses hold, several follow-up questions become critical:

- What's the optimal chunk size for different domains?
- How should structure be represented (XML? JSON? Markdown headers?)
- When does hybrid search beat pure vector search?
- How do we evaluate context quality before querying?
- What's the cost-quality trade-off curve for different strategies?

These aren't just technical curiosities. **They're engineering decisions that separate production-grade systems from demos.**

---

## What You'll See in the Follow-Up

After we run these experiments (estimated 10-12 weeks), we'll publish a follow-up article with:

- **The data:** All 4,380 API calls, anonymized and available for replication
- **The results:** Whether H1 and H2 hold, with statistical significance tests
- **The insights:** What we learned that we didn't expect to learn
- **The decision framework:** A practical guide for choosing strategies based on your constraints

We'll also open-source the entire experimental framework, so you can run variations for your own use cases.

---

Stay tuned.

---

## Methodology Note

This research is being conducted independently, with no vendor funding or affiliation. We're using Google's Gemini API on the free tier because:
1. It offers 1M token context windows (needed for H1)
2. It's genuinely free (we can run 4,380 calls at $0 cost)
3. The results will be publicly verifiable

We acknowledge this limits generalization to other models, but we believe the principles will transfer. If the community finds these results valuable, we'll consider expanding to Claude, GPT-4, and others.

---

## About This Research

This experiment is part of a broader project investigating context engineering for LLMs. The complete experimental framework, including all code, data, and analysis, will be open-sourced at [github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments).

**Timeline:** 10-12 weeks from experiment start  
**Follow-up article:** Expected Q1 2026  
**Replication encouraged:** All methodology, code, and data will be public

---

*If you're interested in following this research, connect on [LinkedIn](#) or watch the GitHub repository for updates.*

*Have thoughts on the experimental design? Found a confounding variable we missed? Reach out—we're still in the implementation phase and open to feedback.*

