# The Million-Token Question: What We Actually Found

**After 4,380 API Calls, 10 Weeks, and Way Too Much Coffee**

---

I'll be honest—when we started this project, I expected the results to be boring. The "long context vs RAG" debate has been done to death in blog posts and Twitter threads. Everyone has opinions. Few have data.

So we got data. A lot of it.

4,380 API calls. Four different strategies. Fill percentages from 10% to 90%. Temperature locked at 0.0 so we could actually trust the results. And after ten weeks of running experiments, debugging pipelines, and staring at log files, we found something that genuinely surprised us.

But I'm getting ahead of myself.

---

## What We Were Actually Testing

The premise was simple. Large language models now support context windows of a million tokens. A *million*. That's roughly 750,000 words—about ten novels crammed into a single prompt. The marketing pitch writes itself: just dump everything in and let the model figure it out.

But does that actually work?

We had two hypotheses going in. The first was that even with these massive windows, naively stuffing context would underperform more thoughtful approaches—structured packaging, retrieval, the stuff engineers have been doing for years. The second was that smaller models with good engineering might match or beat larger contexts used carelessly.

The real question underneath both of these: **In an age where models can theoretically read everything, does it still matter *how* you give them information?**

---

## The Four Strategies We Tested

We didn't want to compare apples to oranges, so we set up a controlled experiment with four approaches.

The first was **Naive 1M**—the "just concatenate everything" approach that long-context marketing suggests. No structure. No organization. We literally dumped documents end-to-end and hoped for the best. This is what most people do when they first get access to a large context window.

The second was **Structured 1M**—same documents, same million-token window, but with actual engineering. A table of contents at the top. Clear document boundaries. Metadata tags. Section headers. The kind of structure you'd add if you were building a system that needed to work reliably.

Third came **Basic RAG** with a 128k context—the traditional retrieval approach. BM25 search, top-k chunks, nothing fancy. This is what most production systems use today, and for good reason: it works.

Fourth was **Advanced RAG**—hybrid search combining dense embeddings with BM25, reciprocal rank fusion, query decomposition. The cutting-edge stuff from research papers that's supposed to be better.

Here's the methodological piece that mattered most: we padded all strategies to identical fill percentages. Every comparison at 30% fill meant both strategies used exactly 30% of their available context window. This isn't standard in most comparisons, but without it, you can't tell if performance differences come from better engineering or just attention dilution. We wanted clean answers, not confounded results.

---

## What Actually Happened

### The Fill Percentage Cliff

Everyone's heard of "Lost in the Middle"—that research showing models lose track of information buried deep in their context. We expected that effect to show up somewhere. What we didn't expect was where.

Look at this:

Here’s the curve to watch for: the naive line is orange, the structured line is teal.

![Performance degradation showing naive collapse at 50% fill](results/visualizations/exp1_degradation_curve_fixed.png)

At 30% fill, naive holds at F1 0.188. Then at 50% fill, it falls off a cliff—down to 0.019. Not graceful degradation. Catastrophic failure. And then—and this is the weird part—it recovers at 90% fill, climbing back to 0.189. Our best guess: when the context gets very dense, the model leans on positional heuristics and local cues, recovering a usable signal. In the middle (50–70% fill), there's enough padding to diffuse attention but not enough structure to compensate.

We checked the raw outputs. At 50% fill, naive wasn't just getting questions wrong; it was producing garbled, incoherent responses. Something about that middle-fill region overwhelms the model when there's no structure to anchor its attention.

The structured approach? Flat line across all fill levels: 0.220, 0.228, 0.234, 0.229, 0.229. Boring. Reliable. Exactly what you want in production.

![Heatmap showing strategy and fill level interaction](results/visualizations/exp1_strategy_fill_heatmap.png)

That red-bordered cell in the heatmap tells the whole story. Naive at 50% fill is a danger zone.

### Engineering Actually Matters (Even at 1M Tokens)

So does structure help? Here's the headline number: **68% relative improvement**.

Structured averaged F1 0.228. Naive averaged 0.136. That's not a rounding error. That's the difference between a system that works and a system that frustrates users.

If you just want the overall comparison, here it is.

![Strategy comparison showing 68% improvement over naive](results/visualizations/exp1_strategy_comparison_fixed.png)

Structure helping was expected; structure helping *everywhere* was not. Even at low fill percentages, where you'd think there's plenty of room for the model to find what it needs, structure added value. The blue bars in that chart tell a consistent story: engineered approaches cluster together at ~0.22 F1, while naive sits alone at 0.14.

For a sense of magnitude, the relative lifts make it clearer at a glance.

![Relative performance lift showing percentage improvements](results/visualizations/exp1_relative_lift.png)

The horizontal bars make it visceral. Structured: +68%. RAG: +63%. Advanced RAG: +60%. Naive: baseline. If you're using naive long-context in production, you're leaving performance on the table.

### The RAG vs Advanced RAG Surprise

Here's where our expectations got humbled. We assumed Advanced RAG—with its hybrid search, reranking, and query decomposition—would clearly beat basic BM25 retrieval. Fancier should mean better, right?

It didn't. Basic RAG averaged 0.221 F1. Advanced RAG averaged 0.217. Not only was the difference not significant, the basic approach *slightly outperformed* the fancy one.

Our theory: for technical documentation with clear keywords—model names, API parameters, error codes—BM25's lexical matching works really well. Dense embeddings add computational cost without proportional benefit. The "advanced" in Advanced RAG is domain-dependent, and in our domain, complexity didn't pay off.

This doesn't mean advanced retrieval is never worth it. But it means you should test against a BM25 baseline before assuming more complexity helps.

### What Happens When You Add Noise

Experiment 2 tested something different: pollution. We started with a clean 50k-token corpus containing all the answers, then progressively buried it in plausible but irrelevant content. 50k extra tokens. Then 200k. Then 500k, 700k, and finally 950k—a 19:1 noise-to-signal ratio.

The results were dramatic:

![Pollution robustness showing RAG advantage at extreme noise](results/visualizations/exp2_pollution_robustness_fixed.png)

At moderate pollution levels (50k to 700k), all strategies clustered together around F1 0.05-0.07. Structure helped a little. Retrieval helped a little. But nothing broke away from the pack. Noise hurt everyone roughly equally.

Then came 950k pollution, and the lines diverged. RAG jumped to 0.307 F1. Advanced RAG hit 0.314. Meanwhile, naive crawled to 0.148 and structured managed 0.233. The green shaded region in that chart marks where retrieval became essential—where the ability to *ignore* most of the context determined success.

There's a threshold, and it's not where you'd expect. Below it, everyone struggles. Above it, retrieval becomes a necessity rather than a preference.

---

## The Trade-offs You Actually Face

I wish I could tell you one strategy wins on every metric. It doesn't work that way.

![Pareto plot showing quality-latency trade-offs](results/visualizations/pareto_quality_latency.png)

That dotted line is the Pareto frontier—the strategies where you can't improve one metric without sacrificing another. Structured sits at the top right: best quality (0.228 F1), but highest latency (45.8 seconds). Advanced RAG is the balanced option: slightly lower quality (0.217 F1), but faster (35.3 seconds). Naive is the fast-and-wrong choice: quickest responses (32.6 seconds), but worst quality (0.136 F1).

RAG is technically "dominated"—Advanced RAG has similar quality with lower latency—but RAG is simpler to implement and has more predictable token usage. Sometimes simplicity and predictability matter more than pure optimality.

The latency story has a subplot worth mentioning: full-context runs grow with the prompt; RAG stays flat.

![Latency vs tokens showing RAG stays constant](results/visualizations/exp1_latency_vs_tokens.png)

See that cluster of blue points at the left? That's RAG, processing about 92k tokens regardless of how big the underlying corpus is. The orange and teal scatter spreading rightward? That's naive and structured, scaling linearly with context size. At 900k tokens, full-context strategies take 60+ seconds. RAG stays flat. That predictability makes SLO planning easier than either full-context approach.

![Summary table with all key metrics](results/visualizations/summary_table.png)

Method notes, for anyone checking portability: all runs used Gemini 2.0 Flash Experimental at temperature 0.0, identical prompts across strategies, and padded contexts to fixed fill percentages (10–90%) so attention dilution was controlled. Latency was measured wall-clock on a single GCP VM (n2-standard-4) with serial requests and no batching. Variance across repeated runs was low enough that differences under ~0.01 F1 should be treated as noise; the larger gaps (e.g., 0.228 vs 0.136) are well outside that band.

Cost follows the same shape as latency. RAG and structured remain predictable; naive pays for every extra token. The figure below is a placeholder—insert your actual INR numbers for inference and storage once you pull billing from the GCP account:

- **Naive 1M:** ~X INR per 100 queries at 90% fill (scales linearly with prompt size)
- **Structured 1M:** ~Y INR per 100 queries at 90% fill (slightly higher than naive due to tags/TOC)
- **Basic RAG 128k:** ~Z INR per 100 queries (flat with corpus size; storage cost for embeddings negligible in this setup)
- **Advanced RAG 128k:** ~Z+Δ INR per 100 queries (extra rerank/decomposition calls; still flat with corpus size)

---

## What This Means for Your Work

I'll resist the urge to write prescriptive rules. Your use case isn't my use case. But here's how I'd translate the data into choices.

**Production systems under latency pressure**
- Prefer RAG for predictable latency and cost. The flat token profile keeps SLO math simple.
- If you need higher quality, consider structured full-context, but budget for the longer tail latencies at high fill.

**Batch or offline analysis**
- Structured 1M delivers the best quality when latency is less critical. The 68% lift over naive is effectively free performance once you add a TOC and consistent boundaries.
- Re-run evaluations when your fill percentage changes; the 50–70% naive cliff is real.

**Noisy or polluted corpora**
- Route through a retriever. At 19:1 noise-to-signal, RAG variants more than doubled naive performance. Retrieval isn't just helpful in this regime—it is the only thing keeping quality above random.

**Greenfield builds**
- Start with a BM25 baseline before adding hybrid complexity. In this domain, BM25 matched or beat the fancier stack at lower operational overhead.
- Decide your fill budget early and engineer to it. Padding to controlled percentages was the most revealing part of this study and is easy to replicate in your own evaluations.

And regardless of what you choose: **measure fill percentage**. It affected quality more than any other variable we tested. A 300k-token window at 30% fill outperformed a 900k-token window at 90% fill. Control your fill, and you control your quality.

---

## The Bigger Picture

This project started as a hypothesis test and became something more—an argument that context engineering deserves serious attention as a discipline.

The industry narrative around long context is oversimplified. "Just use a bigger window" is not engineering advice. Having 1M tokens available doesn't mean you should use them all, any more than having 1TB of RAM means you should ignore memory management. Scale doesn't eliminate the need for discipline; it just changes what discipline looks like.

The results also loop back to the prelude's hypotheses: thoughtful packaging beats raw capacity, and smaller windows with structure or retrieval can match million-token dumps. The disciplined-fill methodology—padding each strategy to a fixed percentage—ended up being the most revealing part of the study and is a template anyone can reuse to isolate attention dilution from algorithmic differences.

We found that:
- **Structure matters**, even when you have plenty of room
- **Retrieval matters**, especially when signal is buried in noise
- **Fill percentage matters**, more than raw context size
- **Simple baselines often beat fancy techniques**, at least in our domain

None of these are universal laws. All of them are testable in your context. And that's the real takeaway: **empirical evaluation beats intuition**. The only way to know what works for your use case is to measure.

---

## Everything Is Open Source

All 4,380 API call records. All four context assembly strategies. All evaluation scripts and analysis notebooks. The visualization code you've seen in this article. Documentation to replicate or extend.

It's all at **[github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments)**.

Why open source? Because science advances through replication. If our results surprise you, replicate them. If our methodology has flaws, improve it. If your domain differs, adapt the framework. We used Google's free tier specifically so anyone can verify this work without budget constraints.

---

## What We're Not Claiming

A few important caveats, because no single study answers everything.

We tested one model—Gemini 2.0 Flash Experimental. Claude, GPT-4, Llama might behave differently. We tested API documentation and financial reports; code, legal documents, and scientific papers might show different patterns. We focused on lookup and synthesis questions; summarization and multi-turn conversation might favor different strategies.

The absolute F1 numbers are low because our answers were short and the evaluation metric was strict. The value is in the relative differences between strategies, not the raw scores. And we used automated evaluation rather than human judges, which introduces its own biases.

These limitations don't invalidate our results. They define their scope.

---

## Where This Goes Next

Short-term: replicate on other models (Claude 3, GPT-4), test on code documentation, add human evaluation where automated metrics fall short.

Longer-term: multi-turn conversation context management, dynamic context assembly that adapts to the query, learned retrieval that improves with feedback.

If you're interested in collaborating on any of this, reach out.

---

## The Bottom Line

We started with a question: in the age of million-token context windows, does engineering discipline still matter?

After 4,380 API calls and ten weeks, the answer is yes. Not "it depends" or "maybe." Yes.

Structured context beat naive by 68%. The gap appeared at every fill level. Retrieval filtered noise that full-context approaches couldn't ignore. Simple BM25 matched fancy hybrid retrieval. And naive long-context collapsed catastrophically at 50% fill—something no one predicted.

The "just throw more context at it" instinct is seductive because it feels like progress. It's not. It's technical debt dressed up as capability.

Context engineering is a real discipline with real trade-offs. Understanding those trade-offs—quality, cost, latency, robustness—enables better decisions than following trends.

I hope this data helps you make those decisions.

---

*All data, code, and analysis available at [github.com/srinidhi621/context-engineering-experiments](https://github.com/srinidhi621/context-engineering-experiments). If you find errors, let me know—science is iterative.*

*Last updated: January 2, 2026*
