# Step 4b — SAE Scoring & Contrastive Pairs: Analysis Report

**Run:** Run 2  
**Date:** 2026-06-25  
**Model:** `meta-llama/Llama-3.1-8B-Instruct` (4-bit quantized) + SAE `Zhongzhi1228/sae_llama_l16_h65536` (layer 16, TopK-7, 65k features)

---

## 1. Input Overview

| Item | Value |
|---|---|
| Target features (missing features) | **635** |
| Synthetic queries generated (Phase 4a) | **3,175** (5 per feature, sequential assignment) |
| Textspans file total rows (all SAE activations) | **2,868,069** |
| Unique neurons activated across all queries | **41,916** |
| Score range (full textspans) | 0.0000 — 8.5011 |

The 3,175 queries are assigned to features sequentially: feature at position `i` in the ordered feature list receives `text_id` values `[i*5, ..., i*5+4]`.

---

## 2. Own-Feature Activation: How Well Did Queries Activate Their Own Feature?

Each feature had exactly 5 queries generated specifically for it. The table below shows how many of those queries actually produced a nonzero SAE activation on their own target feature neuron.

| Metric | Count |
|---|---|
| Features where ≥1 own query activated the feature | **106 / 635 (16.7%)** |
| Features where 0 own queries activated the feature | **529 / 635 (83.3%)** |
| Total own-feature (NeuronID, TextID) activations with score > 0 | **11** |
| Own-query activation score min / mean / max | 0.0002 / 0.2001 / 2.2714 |

Only **11 out of 7,930** total target-feature activations (0.1%) came from a query that was actually designed for that feature. This means the abliterated model's synthetic queries almost never directly activate their intended neuron — but other neurons fire abundantly on them.

---

## 3. Cross-Feature Activations: Queries Activating Features They Were Not Created For

### 3.1 Overall Numbers

After filtering the textspans to rows where:
- `NeuronID` is one of the 635 target features
- `TextID` belongs to one of the 3,175 synthetic queries
- `Score > 0`

| Metric | Count | % of total |
|---|---|---|
| Total (NeuronID, TextID) activations | **7,930** | 100% |
| Own-feature activations | **11** | 0.1% |
| Cross-feature activations | **7,919** | **99.9%** |

### 3.2 Coverage

| Metric | Count |
|---|---|
| Unique queries involved in ≥1 cross-feature activation | **1,547 / 3,175** |
| Target features that received ≥1 cross-feature query | **519 / 635** |
| Target features with 0 cross-feature activations | **116 / 635** |
| Target features with 0 activations of any kind | **116 / 635** |

### 3.3 Distribution of Unique Cross-Feature Queries per Feature

Among the 519 features that received at least one cross-feature activation:

| Statistic | Value |
|---|---|
| Mean | 15.26 |
| Std | 54.55 |
| Min | 1 |
| 25th percentile | 2 |
| Median (50th) | 5 |
| 75th percentile | 13 |
| 90th percentile | 30 |
| 95th percentile | 46 |
| Max | 999 |

### 3.4 Top 10 Features by Unique Cross-Feature Queries Received

| Feature ID | Unique cross-feature queries |
|---|---|
| 61055 | 999 |
| 35672 | 493 |
| 43021 | 299 |
| 11310 | 258 |
| 8483 | 235 |
| 14114 | 183 |
| 35693 | 124 |
| 50778 | 108 |
| 15578 | 107 |
| 35669 | 90 |

### 3.5 Cross-Feature Quality vs Own-Feature Quality

| Metric | Value |
|---|---|
| Features where best cross-feature score > best own-feature score | **515 / 519** |
| Features with no own-query activation but ≥1 cross-feature activation | **508 / 635** |

For virtually every feature that had any activation at all, a query from a *different* feature produced a higher activation score than the feature's own generated queries.

---

## 4. Contrastive Pairs — State Before Changes (v1)

The original pipeline (`analyze_step1_synthetic_data.py` + `merge_step1_failed_cases.py`) only considered the 5 own-feature queries per feature when selecting the Good example.

| Metric | Count |
|---|---|
| Total contrastive pairs output | **635** |
| Good example: original span from missing_features.tsv (fallback, score=5.0) | **594 / 635 (93.5%)** |
| Good example: real synthetic query activation | **41 / 635 (6.5%)** |
| Real Good activation score min / mean / max | 0.0152 / 0.2894 / 2.2714 |

**93.5% of Good examples** fell back to the manually-curated original span because the feature's own 5 queries failed to produce a sufficient activation (≥2 activated own queries required). Only 41 features had enough own-query signal to select a real synthetic Good example.

---

## 5. Code Changes (v2)

### 5.1 Motivation

Given that:
- 99.9% of all target-feature activations come from cross-feature queries
- 508 features have no own-query activation at all but do receive cross-feature activations
- For 515/519 activated features the cross-feature best score outperforms the own-query best

…the previous Good-example selection (restricted to own queries) missed the vast majority of high-quality signal available in the textspans. The change extends the Good example search to the full 3,175-query pool.

### 5.2 New Files

| File | Role |
|---|---|
| `fac_synthesis/step1_contrastive_pair_construction/analyze_step1_synthetic_data_2.py` | Extended analysis script — computes `global_best` per feature |
| `fac_synthesis/step1_contrastive_pair_construction/merge_step1_failed_cases_2.py` | Updated merge script — uses `global_best` for Good example |

### 5.3 `analyze_step1_synthetic_data_2.py` — What Changed

**Added** after loading `all_texts` (~line 91): a new block that computes `global_best_by_feature`.

```python
# For each feature, find the single best-activating query across ALL text IDs (not just own assigned ones).
global_best_by_feature = {}
valid_text_ids = set(all_texts.keys())
active_spans = spans_df[(spans_df['NeuronID'].isin(matching_ids_set)) & (spans_df['Score'] > 0)]
active_spans = active_spans[active_spans['TextID'].isin(valid_text_ids)]
for fid, group in active_spans.groupby('NeuronID'):
    best_idx = group['Score'].idxmax()
    best_row = group.loc[best_idx]
    tid = int(best_row['TextID'])
    global_best_by_feature[int(fid)] = {
        'text_id': tid,
        'text': all_texts.get(tid, ''),
        'span': str(best_row['Span']),
        'score': float(best_row['Score'])
    }
```

**Modified** the `json.dumps` write call to include the new field:

```python
outj.write(json.dumps({
    'feature_id': fid,
    'samples': samples,          # unchanged — own 5 queries (used for Bad selection)
    'global_best': global_best_by_feature.get(int(fid))   # NEW
}, ensure_ascii=False) + '\n')
```

Everything else in the script is identical to v1.

### 5.4 `merge_step1_failed_cases_2.py` — What Changed

**Good example selection** now uses `global_best`:

```python
global_best = data.get("global_best")

# Good: highest-scoring query for this feature from ANY source
good = global_best if (global_best and is_activated_sample(global_best)) else None

# Bad: unchanged — lowest-scoring activated own-feature query (requires >=2 own activations)
activated = [s for s in samples if is_activated_sample(s)]
bad = None
if len(activated) >= 2:
    srt = sorted(activated, key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    bad = srt[-1]
```

**Three output cases** (replacing the original two):

| Case | Good source | Bad source |
|---|---|---|
| `global_best` valid AND ≥2 own activations | `global_best` (any query, highest score) | Lowest activated own query |
| `global_best` valid, <2 own activations | `global_best` (any query, highest score) | Lowest own query (may be score=0) |
| No `global_best` (116 features, zero activations) | Original span from `missing_features.tsv` (score 5.0) | Lowest own query |

The **Bad example logic is identical** to v1 in all cases. The fallback for features with no activation of any kind is also identical to v1 (original span as Good).

### 5.5 CLI Usage (v2)

```bash
# Step 1: analyze (adds global_best to each record)
python analyze_step1_synthetic_data_2.py \
  --final-decision-file /content/missing_features.tsv \
  --textspans-file .../textspans_group0.tsv \
  --synthetic-queries-file .../step1_queries.queries.tsv \
  --output-jsonl .../4b_step1_analyzed_v2.jsonl

# Step 2: merge (uses global_best for Good example)
python merge_step1_failed_cases_2.py \
  --final-decision-file /content/missing_features.tsv \
  --triplets-file .../4b_step1_analyzed_v2.jsonl \
  --output-file .../step1_contrastive_pairs_v2.jsonl
```

---

## 6. Expected Impact of Changes

| Metric | v1 (before) | v2 (expected) |
|---|---|---|
| Good examples using original span fallback | 594 (93.5%) | ~116 (18.3%) |
| Good examples using real SAE activation | 41 (6.5%) | ~519 (81.7%) |
| Good examples sourced from own queries | up to 41 | up to 41 (subset of the 519) |
| Good examples sourced from cross-feature queries | 0 | up to 519 |
| Features still using fallback (no activation at all) | 594 | 116 |

The new approach reduces the fallback rate from **93.5% → ~18.3%** by drawing Good examples from the full 3,175-query pool instead of only the 5 own-feature queries.
