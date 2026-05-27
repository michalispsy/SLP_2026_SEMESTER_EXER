# Step 5 Progress Report — Toxicity Detection Finetuning

> **Status:** ✅ Ready to run on Colab  
> **Last updated:** 2026-05-27  
> **Notebook:** [step5_finetuning.ipynb](step5_finetuning.ipynb)

---

## What Step 5 Does

Step 5 is the **downstream evaluation** of the entire FAC-Synthesis pipeline. We take the synthetic toxic queries generated in Steps 4a→4b→4c and use them to finetune a **binary toxicity classifier** on top of Llama-3.1-8B-Instruct.

The central claim of the paper ("Less is Enough") is tested here: **200 targeted synthetic samples** should outperform other methods that generate more, but less targeted, data.

```
Steps 1–3   SAE Feature Analysis
    → 318 missing toxic features identified

Step 4a     Candidate Query Generation (abliterated Llama)
    → 1,590 raw toxic queries

Step 4b     SAE Scoring + Contrastive Pair Construction
    → Ranking per feature, good/bad pairs

Step 4c     Refined Query Generation (Round 2)
    → 636 refined toxic queries  ← our SYNTHETIC_TOXIC_TSV

Step 5      ← YOU ARE HERE
    → Train classifier: 200 synthetic toxic + 1,000 safe samples
    → Evaluate: AUPRC on ToxicChat
```

---

## The Training Data

### What goes into training

| Role | Source | Label | Count | File |
|---|---|---|---|---|
| **Safe queries (t=0)** | HH-RLHF helpful-base train split | 0 | 1,000 (sampled) | `safe_hh_rlhf.tsv` |
| **Toxic queries (t=1)** | FAC Synthesis Step 4c output | 1 | 200 (cut from 636) | `step2_queries.queries.tsv` |
| **Total train** | | | **~1,200** | |

The safe queries were extracted from HH-RLHF `helpful-base/train.jsonl.gz` by taking the first 3 human turns of each conversation, joining them with ` | `, and stripping all newlines. All texts are plain queries — no `H:` / `A:` prefixes. The chat template is applied by the script at tokenization time.

The 200 toxic cut is intentional: the paper matches all methods to the same sample budget for fair comparison. Our pipeline generates 636, but only 200 are used.

### Evaluation data (NOT seen during training)

| Split | Source | Rows | Toxic | File |
|---|---|---|---|---|
| **Validation** (early stopping) | ToxicChat 0124 train split | 5,082 | 384 (7.6%) | `valid.tsv` |
| **Test** (final AUPRC reported) | ToxicChat 0124 test split | 5,083 | 362 (7.1%) | `test.tsv` |

> ⚠️ **Known data quality note:** 196 queries appear in both valid and test (from the original ToxicChat dataset split). This is not our bug — the paper uses the same splits. Since all methods in Table 1 share this same bias, comparisons remain fair.

---

## The Model & Hyperparameters

All parameters match **paper Appendix I** exactly:

| Parameter | Value | Paper reference |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | Appendix H.2 |
| Task | Sequence Classification (2 labels) | Appendix H.1 |
| Quantization | 4-bit NF4 (QLoRA) | *Colab adaptation — paper uses bf16* |
| LoRA rank `r` | 8 | Appendix I |
| LoRA alpha `α` | 16 | Appendix I |
| LoRA dropout | 0.1 | Appendix I |
| Target modules | `q,k,v,o,gate,up,down` _proj | Appendix I |
| `modules_to_save` | `score` (classification head) | Appendix I |
| Learning rate | 5e-5 | Appendix I |
| Epochs | 3 | Appendix I |
| Per-device batch | 4 | Appendix I |
| Gradient accum | 4 → effective batch = **16** | Appendix I |
| Precision | bf16 | Appendix I |
| Max length | 512 (right truncation) | Appendix I |
| Seeds | 42, 43, 44 | Standard |

> **Why 4-bit quantization?** Llama-3.1-8B in bf16 requires ~16 GB VRAM. A T4 GPU (Colab free tier) has exactly 16 GB, leaving no room for activations. 4-bit NF4 reduces model memory to ~5 GB. The paper ran on H100/A100 clusters where this isn't needed.

---

## What We Compare Against (Paper Table 1)

The same finetuning script is run with different `SYNTHETIC_TOXIC_TSV` files — one per synthesis method — all matched to 200 samples:

| Method | Strategy | AUPRC (LoRA) | AUPRC (head-only) |
|---|---|---|---|
| Baseline | No synthetic augmentation | 38.97 ± 2.74 | 38.97 ± 2.74 |
| Full Dataset | Entire HH-RLHF (no budget) | 49.59 ± 2.29 | 44.31 ± 1.14 |
| Alpaca | Self-instruct | 50.59 | 44.15 |
| Evol-Instruct | Instruction evolution | 49.47 | 45.07 |
| Magpie | Alignment generation | 44.18 | 37.97 |
| CoT-Self-Instruct | CoT expansion | 50.86 | 43.68 |
| SAO | Self-alignment optimization | 50.51 | 42.76 |
| Prismatic Synthesis | Alignment-objective | 52.11 | 45.43 |
| SynAlign | Alignment-constrained | 58.83 | 42.68 |
| **Ours (FAC)** | **Missing-feature targeted** | **62.60 ± 4.41** | **49.12 ± 0.49** |

We run only **our method**. Baseline numbers come from the paper. The gap vs Baseline is **+23.63 AUPRC** in the LoRA setting.

---

## Output Structure

Everything is written to Google Drive to survive Colab disconnections:

```
/content/drive/MyDrive/fac_synthesis/step_5/output/
├── seed42/
│   ├── training_log.jsonl     ← full event log (loss, AUPRC per step, final test)
│   ├── progress.txt           ← status=completed | seed=42 | test_auprc=X.XX
│   ├── logits/
│   │   ├── eval_logits.tsv    ← validation logits (overwritten each eval step)
│   │   └── test_logits.tsv    ← final test logits (the important one)
│   ├── checkpoint-50/
│   └── checkpoint-100/        ← best 2 kept; best loaded at end
├── seed43/
└── seed44/
```

The key output per seed is **`test_logits.tsv`** — from this we compute the final AUPRC reported in our results.

---

## How to Run

```python
# In Colab:

# 1. Clone the repo (all input data is already in it)
!git clone https://github.com/<your-repo>/FAC-Synthesis.git

# 2. Mount Drive (for output persistence)
from google.colab import drive
drive.mount('/content/drive')

# 3. Run the notebook cells in order
# All paths are pre-configured in cell 7 (Configuration)
```

**Estimated runtime per seed:**
- T4 GPU (free Colab): ~15–25 minutes  
- A100 (Colab Pro): ~3–5 minutes

Total for 3 seeds: ~45–75 minutes on T4.

---

## Bugs Fixed During Setup

A full audit is in [STEP5_BUGS_FIXES_REPORT.md](STEP5_BUGS_FIXES_REPORT.md). Key fixes relevant to running the notebook:

| # | Bug | Impact | Fixed |
|---|---|---|---|
| 4 | Synthetic data was overwritten and never trained on | Silent training failure | ✅ |
| 10 | `OUTPUT_DIR` in ephemeral `/content/` | Checkpoints lost on disconnect | ✅ |
| 11 | **Model shared across seeds** | Seeds 43/44 started from already-trained weights — results wrong | ✅ |
| 12 | No fault tolerance in Colab | No resume, no logs, no checkpoints | ✅ |
| 8 | Sample counts hardcoded | Cannot vary budget without editing source | ✅ |

The most critical was **bug #11** (shared model): without the fix, the three seeds would produce dependent, cumulative results instead of three independent replications from the same base checkpoint.

---

## File Map

```
our_work/synthesis/synthesis_data/step5/
├── final_validate_test_datasets/
│   ├── safe_hh_rlhf.tsv          ← t=0 training data (1,000 safe queries from HH-RLHF)
│   ├── valid.tsv                  ← validation set (ToxicChat 0124 train, 5,082 rows)
│   └── test.tsv                   ← test set      (ToxicChat 0124 test,  5,083 rows)
├── hh-rlhf/                       ← raw HH-RLHF download (source for safe_hh_rlhf.tsv)
└── toxic-chat/                    ← raw ToxicChat download (source for valid/test)

our_work/synthesis/synthesis_data/steps_4a,b,c/step4c/OUTPUT/
└── step2_queries.queries.tsv      ← t=1 training data (636 synthetic toxic queries)

our_work/synthesis/code/step5/
├── step5_finetuning.ipynb         ← THE NOTEBOOK TO RUN
├── STEP5_EXPLAINED.md             ← pipeline explanation
├── STEP5_BUGS_FIXES_REPORT.md     ← all bugs found and fixed
└── STEP5_PROGRESS_REPORT.md       ← this file
```
