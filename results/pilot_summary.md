# Pilot Phase: Summary and Key Learnings

**Status:** ✅ COMPLETE
**Decision:** GO

The pilot phase was designed as an end-to-end test of the experimental pipeline. It was a complete success, validating our tooling and methodology and providing critical learnings before starting the main experiments.

---

### 1. Experiment Conducted

The pilot consisted of **18 API calls** with the following setup:
- **Question:** A simple lookup task: "What is the context window size of the Llama 3.3-70B-Instruct model?"
- **Strategies:** `naive` (simple concatenation) and `rag` (retrieval-augmented).
- **Fill Levels:** 30%, 50%, and 70% to simulate varying levels of context noise.
- **Repetitions:** 3 runs per configuration.

---

### 2. Actual Results & Discoveries

- **Final Accuracy:** After correcting our ground truth data, the final result was **100% accuracy** for both `naive` and `rag` strategies across all 18 runs.
- **Initial Failure (Key Finding):** Our first evaluation reported 0% accuracy. This led us to discover our ground truth ("256k") was incorrect and the model was correctly recalling the "128k" value present in the source documents.
- **Bugs Fixed:** The pilot forced us to find and fix several bugs in the `gemini_client`, data loading, and evaluation scripts, hardening our codebase.
- **Resiliency:** A real-world `429 rate limit` error prompted a successful upgrade of the runner script to make it idempotent (resumable), which is critical for the longer experiments.

---

### 3. Key Learnings

1.  **The Pipeline is Solid:** We have successfully proven that our entire system for collecting data, building context, querying the model, and evaluating results works.
2.  **Ground Truth is King:** The pilot's most valuable lesson was that test data must be rigorously verified against the source corpus.
3.  **Automation is a Force Multiplier:** We successfully replaced a manual evaluation process with a faster, more repeatable automated script.
4.  **The Pilot Question is Too Easy:** A 100% score for both strategies confirms that we need the more challenging "Needle in a Haystack" (Experiment 1) and "Context Pollution" (Experiment 2) scenarios to properly measure the performance differences between context strategies.

### Go/No-Go Checklist Assessment

- **[x] All API calls completed successfully:** Yes.
- **[x] Responses are relevant and correct:** Yes.
- **[x] Token counts and Fill % are accurate:** Yes.
- **[x] Major bugs found and fixed:** Yes, this was a primary benefit.
- **[x] Cost is $0:** Yes.

**Conclusion:** The pilot is complete and successful. The project is cleared to proceed to **Experiment 1**.
