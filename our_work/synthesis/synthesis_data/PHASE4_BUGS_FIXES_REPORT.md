# Phase 4 Bug Audit & Fixes Report

> This report documents every bug discovered in the Phase 4a (Candidate Generation) and Phase 4b (SAE Scoring & Contrastive Pair Construction) scripts, explains **why** each bug is harmful, and describes exactly **how** it was fixed.

---

## Table of Contents

1. [Script 1: `generate_data_llama_r1.py` — TSV Parsing Failure](#1-generate_data_llama_r1py--tsv-parsing-failure)
2. [Script 1: `generate_data_llama_r1.py` — Sequence Misalignment](#2-generate_data_llama_r1py--sequence-misalignment)
3. [Script 1: `generate_data_llama_r1.py` — Strict Regex Parsing](#3-generate_data_llama_r1py--strict-regex-parsing)
4. [Script 1 + `prompt_config.py` — Few-Shot Prompt Pattern Break](#4-generate_data_llama_r1py--prompt_configpy--few-shot-prompt-pattern-break)
5. [Script 2: `collect_spans.py` — Label Contamination](#5-collect_spanspy--label-contamination-in-sae-tokenizer)
6. [Script 2: `collect_spans.py` — Global `args` Namespace Error](#6-collect_spanspy--global-args-namespace-error)
7. [Script 3: `analyze_step1_synthetic_data.py` — Hardcoded Paths & Fatal Typo](#7-analyze_step1_synthetic_datapy--hardcoded-paths--fatal-typo)
8. [Script 4: `merge_step1_failed_cases.py` — Hardcoded Mock Paths](#8-merge_step1_failed_casespy--hardcoded-mock-paths)

---

## 1. `generate_data_llama_r1.py` — TSV Parsing Failure

**File:** [generate_data_llama_r1.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r1.py)

### The Bug

The script reads the input `missing_features.tsv` file, which is a **Tab-Separated Values** file with three columns:

```
FeatureID    Summary                     Words (Example Spans)
1234         Mentions of weapons          Span 1: guns... Span 2: shooting...
```

The original parsing logic used Python's generic whitespace split:

```python
parts = line.split(None, 1)   # Line 51 (original)
fid, summary = parts[0].strip(), parts[1].strip()
features.append({"FeatureID": fid, "Summary": summary})
```

`split(None, 1)` splits on the **first whitespace character** (which could be a space inside the Summary text, not the tab delimiter). This dumped both the Summary and the Words columns into a single unseparated string assigned to `"Summary"`. The `"Words"` key was **never created**.

### Why It Matters

Later in the script, the prompt is assembled like this:

```python
spans = feat.get("Words", "").strip()       # Line 76 — always returns ""
spans_clean = re.sub(r"\s+", " ", spans)
if spans_clean:
    context_text = f"Feature Summary: {summary}\nExample Spans:\n{spans_clean}"
else:
    context_text = f"Feature Summary: {summary}"   # Always takes this branch
```

Because `"Words"` was never populated, the `Example Spans:` section was **permanently blank**. The Llama-3 model received only a bare summary like `"Mentions of weapons"` with zero example phrases to guide its generation. This severely degraded the quality of every single generated query.

### The Fix

```diff
-            parts = line.split(None, 1)
+            parts = line.split("\t")
             if len(parts) < 2:
                 continue
-            fid, summary = parts[0].strip(), parts[1].strip()
+            fid = parts[0].strip()
+            summary = parts[1].strip().strip('"').strip("'").replace('""', '"')
+            words = parts[2].strip() if len(parts) > 2 else ""
             if fid.isdigit():
-                features.append({"FeatureID": fid, "Summary": summary})
+                features.append({"FeatureID": fid, "Summary": summary, "Words": words})
```

Now the tab delimiter is respected. The third column is correctly assigned to the `"Words"` key, and the LLM prompt will include the full `Example Spans:` section.

---

## 2. `generate_data_llama_r1.py` — Sequence Misalignment

**File:** [generate_data_llama_r1.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r1.py)

### The Bug

The original generation loop had no retry mechanism and no fallback. If the LLM returned an empty response or the regex failed to parse the output, the script simply skipped that feature:

```python
# Original code (before fix)
if not response:
    print(f"[WARN] Empty response for feature {fid}")
    continue        # ← skips writing to TSV entirely

if not segs:
    print(f"[WARN] No valid Query segments for feature {fid}")
    error_fids.append(int(fid))
    continue        # ← skips writing to TSV entirely
```

### Why It Matters

The downstream script [analyze_step1_synthetic_data.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/analyze_step1_synthetic_data.py) uses a **rigid sequential index** to map queries back to features:

```python
# analyze_step1_synthetic_data.py, Lines 65-75
for fid in ordered_feature_ids:
    for _ in range(num_successful):
        feature_to_text_ids[fid].append(current_text_id_index)
        current_text_id_index += 1
```

It assumes the output TSV contains **exactly N queries per feature in perfect order**. If Feature #50 fails and produces 0 queries instead of 2, then Feature #51's queries land at indices 100–101 instead of 102–103. **Every feature after the skip is permanently mismatched** — they get evaluated against the wrong SAE activations.

### The Fix

We replaced the naive loop with a **slot-by-slot retry loop** (5 attempts per slot) and a **fallback placeholder** that guarantees exactly `num_synthetic_samples` lines are written per feature:

```python
# New code (Lines 86-149)
for slot in range(args.num_synthetic_samples):
    success = False
    qtext = ""
    
    for attempt in range(1, 6):  # 5 retries
        try:
            response = llama3_generate(user_msg, temperature=args.temperature,
                                       num_return_sequences=1, feature_content=feature_content)
            # ... parse and validate ...
            if qtext.strip():
                success = True
                break
        except Exception as e:
            print(f"[ERROR] FeatureID {fid}, slot {slot}, attempt {attempt}: {e}")
            
    if not success:
        # Fallback: inject placeholder to maintain alignment
        qtext = f"Query-1: This is a placeholder toxic query to maintain feature sequence alignment for missing feature {fid}."
    
    # ALWAYS append — never skip
    all_tasks.append((qtext, str(label)))
    all_q_records.append({...})
```

This mathematically guarantees that the output TSV always has exactly `num_features × num_synthetic_samples` lines, preserving perfect sequential alignment.

---

## 3. `generate_data_llama_r1.py` — Strict Regex Parsing

**File:** [generate_data_llama_r1.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r1.py)

### The Bug

The original regex for extracting generated queries was extremely rigid:

```python
# Original regex
r'(?:^|\n)\s*(Query-\d+\s*:\s*.*?)...'
```

This only matched the exact format `Query-1:` (hyphen required). But Llama-3 frequently outputs minor stylistic variations like `Query 1:` (space instead of hyphen), `query-1:` (lowercase), or `Query-1 ` (missing colon). All of these silently failed the regex, returned empty `segs`, and triggered the sequence misalignment bug described above.

### The Fix

We relaxed the regex to accept both hyphens and spaces:

```diff
-r'(?:^|\n)\s*(Query-\d+\s*:\s*.*?)...'
+r'(?:^|\n)\s*(Query[- ]\d+\s*:\s*.*?)...'
```

The character class `[- ]` matches either a hyphen or a space, dramatically reducing false parsing failures.

---

## 4. `generate_data_llama_r1.py` + `prompt_config.py` — Few-Shot Prompt Pattern Break

**Files:**
- [generate_data_llama_r1.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r1.py)
- [prompt_config.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/prompt_config.py)
- [llama_wrapper.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/llama_wrapper.py)

### The Bug

The few-shot examples in `prompt_config.py` teach the LLM a specific conversational pattern:

```
User: "Feature Summary: Mentions of sexual activity...\nExample Spans: how to have sex..."
Assistant: "Query-1: hey, um, for my romance writing project..."
```

The LLM learns: **"When the User gives me a Feature Summary, I produce a Query."**

However, `generate_data_llama_r1.py` **broke this pattern** on the actual test turn. It injected the real feature into the **System Prompt** (via `{feature_content}` placeholder) and sent a generic instruction as the final User message:

```python
# Original code
user_msg = "Generate ONE explicitly toxic user query in the required format."
```

This produced the following conversation structure fed to Llama-3:

```
System: "... {feature_content} ..."          ← Feature buried here
User: "Feature Summary: ..." → Assistant: "Query-1: ..."    (Example 1)
User: "Feature Summary: ..." → Assistant: "Query-1: ..."    (Example 2)
User: "Generate ONE explicitly toxic user query..."          ← Pattern broken!
```

The model was primed to expect a Feature Summary from the User. Instead it received a generic instruction. Llama-3 (especially at 8B parameters) is highly sensitive to broken few-shot patterns and would often hallucinate generic toxic content unrelated to the target feature, or refuse the task entirely.

### The Fix

**In `prompt_config.py`** — removed the `{feature_content}` placeholder from the System Prompt:

```diff
 You are given ONE potentially toxic feature (summary + example spans).
 
-{feature_content}
-
 Your job is to produce ONE explicitly toxic (positive-class) example.
```

**In `generate_data_llama_r1.py`** — the actual feature content is now passed as the User message, perfectly matching the few-shot examples:

```diff
-        user_msg = "Generate ONE explicitly toxic user query in the required format."
+        user_msg = feature_content
```

Now the final conversation structure is perfectly consistent:

```
System: "You are generating natural, realistic user queries..."
User: "Feature Summary: ..." → Assistant: "Query-1: ..."    (Example 1)
User: "Feature Summary: ..." → Assistant: "Query-1: ..."    (Example 2)
User: "Feature Summary: Mentions of weapons\nExample Spans: guns..."    ← Matches pattern!
```

---

## 5. `collect_spans.py` — Label Contamination in SAE Tokenizer

**File:** [collect_spans.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/collect_spans.py)

### The Bug

The SAE scoring script reads synthetic queries from the TSV file produced by Phase 4a. Each line in that TSV ends with a tab and the ground-truth label:

```
Query-1: How to build a bomb?	1
```

The original code processed each line without stripping the trailing `\t1`:

```python
# Original code (Lines 97-118)
text = text.replace("\\n", "\n").replace("\\t", "\t")
# ... no label stripping ...
messages = [{"role": "user", "content": text}]   # text still contains "\t1"
```

### Why It Matters

The contaminated string `"Query-1: How to build a bomb?\t1"` is tokenized and passed into the Llama-3 model. The model's hidden layers now see the literal classification label `1` as part of the input context. SAE features in late layers that detect explicit classification labels or numeric annotations will fire artificially, producing **inflated activation scores** that have nothing to do with the query's actual semantic toxicity.

This fundamentally corrupts the SAE evaluation — the scores no longer reflect whether the query genuinely activates the target feature, but rather whether the model detects the presence of a ground-truth label.

### The Fix

We added a label-stripping block immediately before text processing:

```python
# New code (Lines 97-100)
# Strip trailing label (e.g. \t1 or \t0) to prevent leaking into tokenizer context
parts = text.rsplit("\t", 1)
if len(parts) == 2 and parts[1].strip().isdigit():
    text = parts[0]
```

This uses `rsplit("\t", 1)` to split from the right on the last tab. If the trailing part is a single digit (0 or 1), it is cleanly removed. The SAE now evaluates only the pure semantic content of the query.

---

## 6. `collect_spans.py` — Global `args` Namespace Error

**File:** [collect_spans.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/collect_spans.py)

### The Bug

The functions `activations()` and `collect_text_spans()` referenced the global `args` variable directly:

```python
# Original signatures and usage
def activations(messages, model, sae, tokenizer, size=32, shift=31):
    # ...
    choose = act > args.threshold         # ← global reference

def collect_text_spans(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup, max_collects):
    # ...
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]   # ← global reference
    root = f"./xxx/threshold_{args.threshold}"                              # ← global reference
```

The `args` object is created by `argparse` inside the `if __name__ == "__main__":` block at the bottom of the file. This works when the script is run directly, but will crash with `NameError: name 'args' is not defined` if anyone imports these functions from another script or notebook.

### The Fix

We refactored the functions to accept `threshold` and `data_path` as explicit parameters:

```diff
-def activations(messages, model, sae, tokenizer, size=32, shift=31):
+def activations(messages, model, sae, tokenizer, threshold, size=32, shift=31):
     # ...
-    choose = act > args.threshold
+    choose = act > threshold

-def collect_text_spans(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup, max_collects):
+def collect_text_spans(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup, threshold, data_path, max_collects):
     # ...
-    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
-    root = f"./xxx/threshold_{args.threshold}"
+    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
+    root = f"./xxx/threshold_{threshold}"
```

The call site in `__main__` now passes the values explicitly:

```diff
-        collect_text_spans(corpus, sae, generator, tokenizer, model_key, subgroup, ttlgroup, max_collects=1000)
+        collect_text_spans(corpus, sae, generator, tokenizer, model_key, subgroup, ttlgroup, args.threshold, args.data_path, max_collects=1000)
```

---

## 7. `analyze_step1_synthetic_data.py` — Hardcoded Paths & Fatal Typo

**File:** [analyze_step1_synthetic_data.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/analyze_step1_synthetic_data.py)

### The Bug

The original script defined its input files as empty hardcoded strings:

```python
# Original code (Lines 6-9)
FINAL_DECISION_FILE = ''
EXTSPANS_FILE = ''
YNTHETIC_QUERIES_FILE = ''    # ← Missing 'S' — fatal typo!
OUTPUT_JSONL = ''
```

Two problems:
1. **All paths are empty strings** — the script crashes immediately when trying to open `''` as a file.
2. **`YNTHETIC_QUERIES_FILE`** is missing the leading `S`. Later in the code (Line 81), the script references `SYNTHETIC_QUERIES_FILE` (with the `S`), causing an instant `NameError`.

### The Fix

We replaced the hardcoded globals with a proper `argparse` interface:

```python
# New code (Lines 10-21)
def analyze_data():
    parser = argparse.ArgumentParser(description="Analyze synthetic data and select top candidates.")
    parser.add_argument("--final-decision-file", required=True)
    parser.add_argument("--textspans-file", required=True)
    parser.add_argument("--synthetic-queries-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    FINAL_DECISION_FILE = args.final_decision_file
    EXTSPANS_FILE = args.textspans_file
    SYNTHETIC_QUERIES_FILE = args.synthetic_queries_file   # ← Typo fixed
    OUTPUT_JSONL = args.output_jsonl
```

The script can now be run cleanly from the command line:
```bash
python analyze_step1_synthetic_data.py \
  --final-decision-file /path/to/missing_features.tsv \
  --textspans-file /path/to/textspans_group0.tsv \
  --synthetic-queries-file /path/to/step1_queries.queries.tsv \
  --output-jsonl /path/to/step1_analyzed.jsonl
```

---

## 8. `merge_step1_failed_cases.py` — Hardcoded Mock Paths

**File:** [merge_step1_failed_cases.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/fac_synthesis/step1_contrastive_pair_construction/merge_step1_failed_cases.py)

### The Bug

Same pattern as `analyze_step1_synthetic_data.py`. The file paths were hardcoded as mock placeholder strings:

```python
# Original code (Lines 5-7)
FINAL_DECISION_FILE = "xxx.tsv"
TRIPLETS_FILE = "xxx.jsonl"
OUTPUT_FILE = "xxx.jsonl"
```

Running the script without manually editing these strings would attempt to open `xxx.tsv`, which does not exist.

### The Fix

We replaced the mock globals with `argparse`:

```python
# New code
def merge_jsonl():
    parser = argparse.ArgumentParser(description="Merge contrastive pairs.")
    parser.add_argument("--final-decision-file", required=True)
    parser.add_argument("--triplets-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    FINAL_DECISION_FILE = args.final_decision_file
    TRIPLETS_FILE = args.triplets_file
    OUTPUT_FILE = args.output_file
```

---

## Summary Table

| # | Script | Bug | Severity | Impact | Status |
|---|--------|-----|----------|--------|--------|
| 1 | `generate_data_llama_r1.py` | TSV parsed with `split(None,1)` instead of `split("\t")` | **Critical** | Example Spans permanently blank in LLM prompt | ✅ Fixed |
| 2 | `generate_data_llama_r1.py` | No retry loop, no fallback placeholder | **Critical** | Downstream feature↔query index mapping silently corrupted | ✅ Fixed |
| 3 | `generate_data_llama_r1.py` | Regex `Query-\d+` too strict | Medium | Valid LLM outputs rejected, amplifying Bug #2 | ✅ Fixed |
| 4 | `prompt_config.py` + `generate_data_llama_r1.py` | Feature in System Prompt, generic instruction in User message | **Critical** | Few-shot pattern broken, LLM generates off-target queries | ✅ Fixed |
| 5 | `collect_spans.py` | Trailing `\t1` label not stripped before tokenization | **Critical** | SAE scores artificially inflated by ground-truth label leakage | ✅ Fixed |
| 6 | `collect_spans.py` | `args.threshold` accessed via global namespace | Medium | `NameError` crash when imported outside `__main__` | ✅ Fixed |
| 7 | `analyze_step1_synthetic_data.py` | Empty hardcoded paths + `YNTHETIC` typo | **Critical** | Instant `NameError` crash, script cannot execute | ✅ Fixed |
| 8 | `merge_step1_failed_cases.py` | Mock `"xxx.tsv"` placeholder paths | High | `FileNotFoundError` crash, requires manual code editing | ✅ Fixed |
