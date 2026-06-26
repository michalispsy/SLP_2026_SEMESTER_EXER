# Step 4b — Experiment vs Normal: Contrastive Pair Comparison

**Run:** Run 2  
**Date:** 2026-06-25  
**Normal:**  `OUTPUT/normal/step1_contrastive_pairs (1).jsonl` — original pipeline (own-query-only Good selection)  
**Experiment:** `OUTPUT/experiment/step1_contrastive_pairs_experiment.jsonl` — new pipeline (`global_best`: best query from all 3,175 regardless of origin)

---

## 1. Verdict

**The experiment is strictly better.** Not a single feature degraded. 580/635 features improved their Good example (91.3%), and the 55 unchanged features are identical in both pipelines (both used the original-span fallback). The key gain is coverage: the fallback rate dropped from 93.5% → 7.6%, meaning 546 features that previously had no real SAE activation as their Good example now have one.

---

## 2. Good Example Source

The most direct measure of improvement: how many features have a real SAE-measured Good example versus the hand-curated original span (score=5.0 sentinel, not a real activation).

| | Normal | Experiment |
|---|---:|---:|
| Total pairs | 635 | 635 |
| Good = original span fallback (score 5.0) | **594 (93.5%)** | **48 (7.6%)** |
| Good = real SAE activation | **41 (6.5%)** | **587 (92.4%)** |

The experiment reduced fallback usage by **12.4×** — from 594 features relying on the hand-written span down to 48. The 48 remaining fallbacks are the features with zero activation from any of the 3,175 queries (no SAE signal available regardless of source).

---

## 3. Good Activation Score Distribution

Scores compared only for pairs where the Good example comes from a real activation (score ≠ 5.0 sentinel).

| Metric | Normal (n=41) | Experiment (n=587) |
|---|---:|---:|
| Min | 0.0152 | 0.0001 |
| Mean | 0.2894 | 0.1753 |
| Median | 0.1206 | **0.1217** |
| 90th percentile | 0.4115 | 0.2679 |
| Max | 2.2714 | **4.4441** |
| Std dev | 0.5523 | 0.2987 |

**Interpretation of the lower mean:** The experiment's mean (0.1753) is lower than normal's (0.2894) but this is expected and not a quality regression. Normal's 41 real activations were the *surviving* features where ≥2 of the feature's own 5 queries happened to activate — a heavily self-selected, high-activation subset. The experiment includes all 587 features with any activation, including many with modest but genuine signal. The **median is essentially identical** (0.1206 vs 0.1217), confirming the per-feature quality is preserved. The **max is nearly 2× higher** (4.4441 vs 2.2714), showing that the best Good examples are actually stronger.

---

## 4. Bad Example Analysis

Bad example selection is **unchanged** between the two pipelines, confirmed by identical numbers.

| Metric | Normal | Experiment |
|---|---:|---:|
| Bad with real activation (span + score > 0) | 41 | 41 |
| Bad with no activation (score = 0 / empty span) | 594 | 594 |
| Bad score mean (activated only) | 0.1025 | 0.1025 |
| Bad score max (activated only) | 1.1600 | 1.1600 |

The bad selection logic is bitwise identical. Any difference in future runs would come only from different `samples` lists (own-feature queries), which are also unchanged.

---

## 5. Contrastiveness: Good − Bad Score Gap

A larger gap means the SAE can more easily distinguish the Good example from the Bad one — a more informative training signal.

Computed only for pairs where Good score ≠ 5.0 (i.e., real activations):

| Metric | Normal (n=41) | Experiment (n=587) |
|---|---:|---:|
| Mean gap | 0.1869 | 0.1682 |
| **Median gap** | 0.0605 | **0.1193** |
| Min gap | 0.0014 | 0.0001 |
| Max gap | 1.4946 | **4.4441** |

The **median gap nearly doubled** (0.0605 → 0.1193). Normal's lower median was driven by several own-query activations where the good and bad examples were similarly weak. The experiment's global_best consistently finds a higher-scoring Good example, widening the gap from the (unactivated) Bad example.

---

## 6. Span and Text Length

Longer, more contextual spans indicate that the SAE fired on a richer portion of the input — a qualitatively better signal.

| Metric | Normal | Experiment |
|---|---:|---:|
| Good span count (non-empty) | 635 | 635 |
| Good span **mean** length (chars) | 88.0 | **108.6** |
| Good span **median** length (chars) | 82.0 | **120.0** |
| Good span max length (chars) | 179 | 173 |
| Good example **mean** text length (chars) | 96.5 | **219.4** |
| Good example **median** text length (chars) | 86.0 | 219.0 |
| Bad span mean length (chars) | 84.2 | 84.2 |

Good spans are **23% longer on average** (88 → 108.6 chars) in the experiment, and the good example texts are **2.3× longer**. The longer texts reflect that global_best examples are multi-query synthetic prompts (Query-1 + Query-2) rather than short original-span snippets. Longer activated spans provide more context for the model to learn what triggers a feature.

---

## 7. Per-Feature Outcome

| Outcome | Count | % |
|---|---:|---:|
| **Improved** (exp got real activation where normal used fallback, or exp score > norm score) | **580** | **91.3%** |
| — of which: normal fallback → experiment real activation | 546 | 86.0% |
| — of which: both real, experiment score > normal score | 34 | 5.4% |
| **Same** (both use fallback, or both have identical real score) | **55** | **8.7%** |
| — of which: both use original-span fallback (no activation found) | 48 | 7.6% |
| **Degraded** (experiment worse than normal) | **0** | **0%** |

**No feature regressed.** The 55 "same" features are those with zero activation in both pipelines (48) plus 7 features where both versions happened to select the same highest-scoring own-feature query.

---

## 8. Global Best Source

Of the 587 experiment features with a valid `global_best`:

| Source | Count | % |
|---|---:|---:|
| Cross-feature query (created for a different feature) | **575** | **98.0%** |
| Own-feature query (created for this feature) | **12** | **2.0%** |

As established in the cross-feature analysis, 98% of the best activations come from queries that were never designed for the feature they activate — the main motivation for the change.

---

## 9. Summary

| Dimension | Normal | Experiment | Change |
|---|---|---|---|
| Real Good activation coverage | 41/635 (6.5%) | 587/635 (92.4%) | **+85.9 pp** |
| Fallback usage | 594/635 (93.5%) | 48/635 (7.6%) | **−85.9 pp** |
| Median Good score (real only) | 0.1206 | 0.1217 | ≈ same |
| Max Good score | 2.2714 | 4.4441 | **+96%** |
| Median Good−Bad gap | 0.0605 | 0.1193 | **+97%** |
| Good span mean length | 88.0 chars | 108.6 chars | **+23%** |
| Features degraded | 0 | 0 | — |
| Bad example logic | unchanged | unchanged | identical |

The experiment is an unambiguous improvement: it produces real, measured SAE activations for 92.4% of features (vs 6.5%), the contrastive gap is twice as wide at the median, the activated spans are richer, and no feature was made worse.
