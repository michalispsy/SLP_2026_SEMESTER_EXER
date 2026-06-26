# Step 4d — Experiment vs Normal: Filtering Results Comparison

**Run:** Run 2  
**Date:** 2026-06-26  
**Normal:** `OUTPUT/normal/` — queries generated from original contrastive pairs (own-span Good examples)  
**Experiment:** `OUTPUT/experiment/` — queries generated from `global_best` contrastive pairs (real SAE-measured Good examples)  
**Config (both):** DELTA=0.0, COLLECT_THRESHOLD=0.0, TOP_M=1, MODEL=llama

Step 4d filters the step 4c generated queries by SAE activation score. It runs two strategies:
- **Per-feature filter**: considers only the 2 queries generated *for* each feature; keeps any that pass the activation threshold.
- **Global filter**: considers all 1,270 queries from step 4c across all features and assigns to each feature the single best-activating query, regardless of origin. This is the primary output.

---

## 1. Verdict

**Normal has better overall feature coverage; experiment has better per-feature specificity.** Normal covers 11 more features globally (589 vs 578) and leaves 11 fewer features with no activation at all (46 vs 57). The experiment's advantage is that 97 features had their own generated query activate the right neuron (vs only 79 in normal) — a direct payoff of the step 4b improvement. However this per-feature gain does not translate into better global coverage, because the thematic bleed introduced in step 4c (address/stalking theme from cross-feature Good examples) left several non-address feature types under-represented in the experiment query pool. The two outputs have **zero content overlap** and are fully complementary.

---

## 2. Coverage — Primary Metrics

| Metric | Experiment | Normal |
|---|---:|---:|
| Input features | 635 | 635 |
| **Global: features covered** | **578 (91.0%)** | **589 (92.8%)** |
| Global: features uncovered (no activation anywhere) | 57 | **46** |
| Per-feature: features where own query passed threshold | **97** | 79 |
| Features rescued by global (failed per-feature, saved globally) | 481 | **510** |
| Global rescue rate | 89.4% | **91.7%** |
| Features only in per-feature (global missed them) | 0 | 0 |

Normal covers **11 more features** in total and rescues a higher fraction of per-feature failures. This is the most important metric for downstream use.

---

## 3. The Counter-Intuitive Finding: Better Per-Feature Specificity but Lower Global Coverage

The experiment produces **97** features where the query generated *for* that feature passes the per-feature threshold — vs only **79** in normal (+18, a 23% gain). This is the direct payoff of the step 4b improvement: real SAE-calibrated Good examples guided the generator toward queries that activate their own target feature more reliably.

Yet globally, the experiment covers **11 fewer** features. The thematic bleed documented in the step 4c analysis explains this. Experiment step 4c queries absorbed the address/stalking theme from cross-feature Good examples into ~49.5% of affected generations. At step 4d, this manifests as topic imbalance:

| Topic | Experiment global | Normal global |
|---|---:|---:|
| Address / stalking | **28.0%** | 23.6% |
| Violence / weapons | 12.8% | **16.0%** |
| Underage / alcohol | 17.7% | **19.6%** |
| Hacking / privacy | 8.4% | **10.3%** |
| Drugs | 11.2% | 11.2% |
| Theft / shoplifting | 13.1% | 13.1% |
| Financial fraud | 5.8% | 5.3% |
| Sexual / coercion | 4.9% | **6.0%** |

The experiment over-represents address/stalking and under-represents violence, underage, hacking, and sexual/coercion categories. Features in those under-represented categories ended up with no activating query in the experiment pool — becoming the 11 extra uncovered features.

---

## 4. File-Level Counts

| File | Experiment | Normal |
|---|---:|---:|
| `4d_global_filtered.queries.tsv` (unique queries) | 429 | 419 |
| `4d_perfeature_filtered.queries.tsv` | 97 | 79 |
| Label distribution (all label=1) | 429 / 0 | 419 / 0 |

All retained queries carry label=1. No label=0 queries passed the filter in either version.

---

## 5. Cross-Assignment and Query Reuse

Both pipelines are almost entirely cross-assigned at the global level — the vast majority of features are best served by a query that was generated for a *different* feature.

| Metric | Experiment | Normal |
|---|---:|---:|
| Cross-assigned (best Q from a different feature) | 554 **(95.8%)** | 579 **(98.3%)** |
| Own-feature assignment (best Q was its own) | 24 | 10 |
| Unique queries written to global file | 429 | 419 |
| Features covered per unique query (avg) | 1.35 | **1.41** |
| Queries serving multiple features (reused) | 149 | 170 |

Normal is more efficient at cross-assignment — each unique query covers 1.41 features on average vs 1.35, meaning normal's step 4c queries activated a broader range of features when tested globally. This reflects the more topically diverse step 4c output in normal (less address-dominated).

---

## 6. Query Length

| Metric | Experiment | Normal |
|---|---:|---:|
| Min (chars) | 123 | 95 |
| Mean | **340.5** | 360.0 |
| Median | 326.0 | 352.0 |
| 90th percentile | 465 | 498 |
| Max | 770 | 733 |
| Std dev | 95.3 | 103.3 |

Experiment queries are consistently shorter (~20 chars at the median). This reflects the step 4c generation style difference: experiment Good examples were real synthetic queries (concise, coded language), while normal Good examples were longer original spans with more context.

---

## 7. Style Markers

| Marker | Experiment | Normal |
|---|---:|---:|
| Euphemistic quoted language (`"creative"` etc.) | **51.7%** (avg 1.24/query) | 44.2% (avg 0.91/query) |
| Contains question mark | 94.4% | 91.6% |
| First-person framing (I / my / me) | **100%** | 99.0% |
| Roleplay / character prompt | 11.2% | **21.2%** |

The experiment queries are more euphemistic and coded; normal queries include nearly twice as many roleplay/jailbreak constructions (`"You're a master of..."`, `"Play as..."`, `"Write a scenario where..."`). Both styles are valid and represent different attack surfaces for the safety training signal.

---

## 8. Content Overlap Between the Two Versions

| Metric | Value |
|---|---:|
| Shared queries (identical text in both global files) | **0** |
| Experiment-only queries | 429 |
| Normal-only queries | 419 |
| Combined unique pool | **848** |
| Diversity gain from merging | **+419 unique queries** |

Not a single query is identical between the two global outputs. They explore entirely different corners of the feature space with different styles and angles. Merging both into a single pool of **848 unique queries** would be the strongest option, giving maximum feature coverage and stylistic diversity.

---

## 9. Summary Table

| Dimension | Experiment | Normal | Edge |
|---|---|---|---|
| Global feature coverage | 578 (91.0%) | 589 (92.8%) | **Normal +11 features** |
| Features uncovered | 57 | 46 | **Normal** |
| Per-feature specificity | **97** own-query passes | 79 | **Experiment +18** |
| Global rescue rate | 89.4% | 91.7% | **Normal** |
| Query reuse efficiency | 1.35×/query | 1.41×/query | **Normal** |
| Topic balance | address-heavy | more balanced | **Normal** |
| Euphemistic style | ✓ higher (51.7%) | 44.2% | Experiment |
| Roleplay diversity | 11.2% | ✓ higher (21.2%) | Normal |
| Query length | shorter (340 chars) | longer (360 chars) | — |
| Content overlap | **0 shared queries** | — | **Fully complementary** |

---

## 10. Recommendation

The two outputs address different strengths:

- **Use Normal** when global feature coverage is the priority — it covers 11 more features and has better topic balance across harmful categories.
- **Use Experiment** when per-feature precision matters — 97 features have a query that genuinely activates their own neuron, which produces cleaner training signal for those features.
- **Merge both** for the best overall result: 848 unique queries, zero redundancy, maximum feature coverage, and complementary stylistic diversity (euphemistic/coded from experiment, roleplay/direct from normal).
