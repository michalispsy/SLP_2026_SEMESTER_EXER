# Steps 1, 2 & 3 Bug Audit & Fixes Report

> This report documents every bug discovered in the SAE Feature Annotation Pipeline (Steps 1–3), explains **why** each bug is harmful, and describes exactly **how** it was fixed.

---

## Table of Contents

1. [`groupby_textspans.py` — Hardcoded Placeholder Output Path](#1-groupby_textspanspy--hardcoded-placeholder-output-path)
2. [`groupby_textspans.py` — Raw File Loaded Instead of Deduplicated Output](#2-groupby_textspanspy--raw-file-loaded-instead-of-deduplicated-output)
3. [`annotate_explanations.py` — Global `KEY` Used Instead of Constructor Parameter](#3-annotate_explanationspy--global-key-used-instead-of-constructor-parameter)
4. [`annotate_explanations.py` — Dead Code in `format()` & Method Never Called](#4-annotate_explanationspy--dead-code-in-format--method-never-called)
5. [All Step 3 Scripts — Missing `import os` Crash](#5-all-step-3-scripts--missing-import-os-crash)
6. [`annotate_toxicity.py` — `format()` Method Never Called](#6-annotate_toxicitypy--format-method-never-called)
7. [All Step 3 Sibling Scripts — "Subjective" Instead of "Objective" in Prompt](#7-all-step-3-sibling-scripts--subjective-instead-of-objective-in-prompt)
8. [`annotate_toxicity.py` — Rigid Regex for `Query-\d+:` Prefix Stripping](#8-annotate_toxicitypy--rigid-regex-for-query-d-prefix-stripping)
9. [`OpenaiAPI.py` — Infinite Retry Loop, `max_retry` Never Enforced](#9-openaiapipy--infinite-retry-loop-max_retry-never-enforced)

---

## Pipeline Architecture & File Mapping

```
sae_feature_analysis/
├── interpret_features/
│   ├── groupby_textspans.py       ← [STEP 1] Group and deduplicate spans
│   ├── annotate_explanations.py   ← [STEP 2] Generate feature summaries via GPT
│   └── OpenaiAPI.py               ← API Wrapper used by Step 2
│
└── identify_task_relevant_features/
    ├── annotate_toxicity.py            ← [STEP 3] Identify toxicity-relevant features
    ├── annotate_helpfulness.py         ← [STEP 3] Helpfulness annotation
    ├── annotate_instruction_following.py ← [STEP 3] Instruction-following annotation
    ├── annotate_steering_survival.py   ← [STEP 3] Survival steering annotation
    ├── annotate_steering_sycophancy.py ← [STEP 3] Sycophancy steering annotation
    └── OpenaiAPI.py                    ← API Wrapper used by Step 3
```

---

## 1. `groupby_textspans.py` — Hardcoded Placeholder Output Path

**File:** [groupby_textspans.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/groupby_textspans.py)

### The Bug

Line 74 uses the `%` string operator to interpolate the filename into the output path, but the string literal `"xxx.tsv"` contains **no `%s` format specifier**:

```python
# Line 73-74
print("./%s.tsv" % file.replace("textspans", "TopAct"))   # ← debug print works correctly
with open("xxx.tsv" % file.replace("textspans", "TopAct"), "w", encoding="utf8") as f:
#          ^^^^^^^^ — literal string with no placeholder; % has no effect here
```

### Why It Matters

The output file is always written to the hardcoded literal `xxx.tsv` regardless of the input directory. Every run overwrites the same file, making it impossible to process multiple threshold directories without manual renaming. The debug `print()` on line 73 shows the correct intended path, but the actual `open()` ignores it entirely.

### The Fix

```diff
- with open("xxx.tsv" % file.replace("textspans", "TopAct"), "w", encoding="utf8") as f:
+ with open("./%s.tsv" % file.replace("textspans", "TopAct"), "w", encoding="utf8") as f:
```

---

## 2. `groupby_textspans.py` — Raw File Loaded Instead of Deduplicated Output

**File:** [groupby_textspans.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/groupby_textspans.py)

### The Bug

The script performs a full deduplication pass (lines 55–65), writing clean records to `full_deduplicated.tsv`. But the very next line instantiates the `Reader` using the **original un-deduplicated** file:

```python
# Line 67
reader = Reader(folder + "/full.tsv")   # ← should be full_deduplicated.tsv
```

### Why It Matters

The deduplication step is completely negated. The `Reader` that performs span grouping for all 65,536 features loads the full redundant dataset, causing inflated span counts, repeated activation examples, and larger context windows for every downstream GPT call in Step 2.

### The Fix

```diff
- reader = Reader(folder + "/full.tsv")
+ reader = Reader(folder + "/full_deduplicated.tsv")
```

---

## 3. `annotate_explanations.py` — Global `KEY` Used Instead of Constructor Parameter

**File:** [annotate_explanations.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/annotate_explanations.py)

### The Bug

Both `TextSpanExplainer` and `TextSpanJudge` accept `key` (lowercase) as a constructor parameter, but their `__init__` bodies reference the global uppercase variable `KEY` instead:

```python
class TextSpanExplainer:
    def __init__(self, key):          # ← receives 'key'
        # ...
        self.model = Chatting.GPT4oMini(KEY, cache=False, ...)  # ← uses global 'KEY'

class TextSpanJudge:
    def __init__(self, key):          # ← receives 'key'
        # ...
        self.model = Chatting.GPT4oMini(KEY, system=instruct, ...)  # ← uses global 'KEY'
```

`KEY` is only defined inside `if __name__ == "__main__":` at line 99 — immediately before the constructors are called there. The constructor `key` parameter is silently ignored.

### Why It Matters

This is the same class of bug as Phase 4 Bug #6 (global `args` namespace error). The code works only when run as a standalone script. Importing either class in a notebook or another script and calling `TextSpanExplainer("sk-...")` will immediately raise:

> `NameError: name 'KEY' is not defined`

### The Fix

```diff
 class TextSpanExplainer:
     def __init__(self, key):
         # ...
-        self.model = Chatting.GPT4oMini(KEY, cache=False, ...)
+        self.model = Chatting.GPT4oMini(key, cache=False, ...)

 class TextSpanJudge:
     def __init__(self, key):
         # ...
-        self.model = Chatting.GPT4oMini(KEY, system=instruct, ...)
+        self.model = Chatting.GPT4oMini(key, system=instruct, ...)
```

---

## 4. `annotate_explanations.py` — Dead Code in `format()` & Method Never Called

**File:** [annotate_explanations.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/interpret_features/annotate_explanations.py)

### The Bug

The `TextSpanExplainer.format()` method has an early `return raw` that makes its second line permanently unreachable:

```python
def format(self, raw):
    raw = raw.replace("\\n", "\n").replace("<s>[INST]", "").strip()
    return raw                                             # ← always returns here
    return "\nSpan".join(raw.split("\nSpan")[:4])         # ← DEAD CODE, never executes
```

Furthermore, the `__call__` method never invokes `format()` at all — it sends spans directly to the batch API:

```python
def __call__(self, cases):
    # ...
    return list(map(self.clean, self.model.batch_call(cases)))  # ← format() bypassed
```

### Why It Matters

Without `format()` being called:
1. Escaped newline literals `\n` and `\t` are not converted to real whitespace before being sent to GPT.
2. Mistral tokenizer artifacts like `<s>[INST]` leak into the prompt.
3. The span truncation to the top-4 (`[:4]`) is skipped, inflating token usage for features with 10+ spans.

### The Fix

```diff
     def __call__(self, cases):
         if isinstance(cases, str):
             cases = [cases]
         if not isinstance(cases, (tuple, list)):
             cases = list(cases)
-        return list(map(self.clean, self.model.batch_call(cases)))
+        formatted = [self.format(c) for c in cases]
+        return list(map(self.clean, self.model.batch_call(formatted)))

     def format(self, raw):
         raw = raw.replace("\\n", "\n").replace("<s>[INST]", "").strip()
-        return raw
         return "\nSpan".join(raw.split("\nSpan")[:4])
```

---

## 5. All Step 3 Scripts — Missing `import os` Crash

**Files:**
- `sae_feature_analysis/identify_task_relevant_features/annotate_toxicity.py`
- `sae_feature_analysis/identify_task_relevant_features/annotate_helpfulness.py`
- `sae_feature_analysis/identify_task_relevant_features/annotate_instruction_following.py`
- `sae_feature_analysis/identify_task_relevant_features/annotate_steering_survival.py`
- `sae_feature_analysis/identify_task_relevant_features/annotate_steering_sycophancy.py`

### The Bug

Every Step 3 script reads the API key with `os.environ.get()` in the `__main__` block:

```python
if __name__ == "__main__":
    KEY = os.environ.get("OPENAI_API_KEY", "<YOUR_API_KEY>")
```

But **none** of these five scripts import `os`. Their import blocks contain `time`, `re`, `string`, `json`, `concurrent`, `multiprocessing`, `tqdm`, and `from OpenaiAPI import Chatting` — but no `import os`.

### Why It Matters

This is the same class of crash as Phase 4 Bug #7 (fatal typo / unresolvable name). Running any of these five scripts from the command line or a Colab cell produces an immediate:

> `NameError: name 'os' is not defined`

before any model is loaded or any data is processed. Note that `annotate_explanations.py` (Step 2) does **not** have this bug — it correctly imports `os` at line 2.

### The Fix

Add `import os` to the top of all five scripts:

```diff
 import time
 import re
 import string
 import json
 import concurrent
 import multiprocessing
+import os
 import tqdm
 from OpenaiAPI import Chatting
```

---

## 6. `annotate_toxicity.py` — `format()` Method Never Called

**File:** [annotate_toxicity.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/identify_task_relevant_features/annotate_toxicity.py)

### The Bug

The `HarmfulJudge` class defines a detailed `format()` method that:
1. Unescapes `\n` and `\t` to real whitespace
2. Strips `<s>[INST]` tokenizer artifacts
3. Strips `Query-N:` prefixes from individual spans
4. Truncates to at most 4 spans
5. Builds the structured `"Feature Name: ...\nExample Text Spans:\n..."` prompt

However, `__call__()` bypasses it entirely with a raw string join:

```python
def __call__(self, cases):
    cases = ["%s\n%s" % (c[0], c[1]) for c in cases]   # ← raw join, no format()
    cases = self.model.batch_call(cases)
    return list(map(self.clean, cases))
```

All four sibling scripts (`annotate_helpfulness.py`, `annotate_instruction_following.py`, `annotate_steering_survival.py`, `annotate_steering_sycophancy.py`) correctly call `format()`:

```python
def __call__(self, cases):
    cases = map(self.format, cases)   # ← correct pattern in sibling scripts
    cases = self.model.batch_call(cases)
```

### Why It Matters

Because `format()` is skipped in `annotate_toxicity.py` only:
- Escaped characters `\n`, `\t` appear literally in the GPT prompt
- `<s>[INST]` tokenizer artifacts contaminate span text
- `Query-N:` generation prefixes are not stripped from span content
- All spans are sent instead of just the top 4, increasing token cost
- The prompt is an unstructured blob instead of the designed `Feature Name / Example Text Spans` format

This makes toxicity annotations systematically lower quality than the sibling task annotations.

### The Fix

```diff
 def __call__(self, cases):
-    cases = ["%s\n%s" % (c[0], c[1]) for c in cases]
+    cases = list(map(self.format, cases))
     cases = self.model.batch_call(cases)
     return list(map(self.clean, cases))
```

---

## 7. All Step 3 Sibling Scripts — "Subjective" Instead of "Objective" in Prompt

**Files:**
- `annotate_helpfulness.py` (Line 19)
- `annotate_instruction_following.py` (Line 18)
- `annotate_steering_survival.py` (Line 22)
- `annotate_steering_sycophancy.py` (Line 19)
- `annotate_explanations.py` — `TextSpanJudge` (Line 73)

### The Bug

The system prompts in all four Step 3 sibling scripts (and the Step 2 `TextSpanJudge`) instruct the LLM:

```python
"Please do not be too mean but be as subjective as possible."
```

Only `annotate_toxicity.py` has the correct word — `"objective"`.

### Why It Matters

"Subjective" means based on personal feelings or opinions; "objective" means based on facts and impartial analysis. Instructing GPT-4o-mini to be "as subjective as possible" is the semantic opposite of the intended behavior — it encourages opinionated, inconsistent, and non-reproducible annotations rather than guideline-grounded evaluations.

### The Fix

```diff
- "Please do not be too mean but be as subjective as possible."
+ "Please do not be too mean but be as objective as possible."
```

Apply to: `annotate_helpfulness.py`, `annotate_instruction_following.py`, `annotate_steering_survival.py`, `annotate_steering_sycophancy.py`, and `TextSpanJudge` in `annotate_explanations.py`.

---

## 8. `annotate_toxicity.py` — Rigid Regex for `Query-\d+:` Prefix Stripping

**File:** [annotate_toxicity.py](file:///home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/sae_feature_analysis/identify_task_relevant_features/annotate_toxicity.py)

### The Bug

The `format()` method strips `Query-N:` markers from spans using an exact regex that only matches a hyphen delimiter:

```python
# Line 45
span = re.split(r"Query-\d+:\ ", span)[-1]
```

### Why It Matters

This is the same strictness bug as Phase 4 Bug #3. The upstream LLM (`generate_data_llama_r1.py`) can emit either `Query-1:` (hyphen) or `Query 1:` (space). When a space variant is present, the regex fails to split and the `Query 1:` prefix leaks into the span text that GPT-4o-mini receives, polluting its context with generation artifacts.

> **Note:** Bug #6 (format() never called) means this bug is currently latent. Once Bug #6 is fixed, this regex will become active and must also be corrected.

### The Fix

```diff
- span = re.split(r"Query-\d+:\ ", span)[-1]
+ span = re.split(r"Query[- ]\d+:\s*", span)[-1]
```

---

## 9. `OpenaiAPI.py` — Infinite Retry Loop, `max_retry` Never Enforced

**Files:**
- `sae_feature_analysis/interpret_features/OpenaiAPI.py`
- `sae_feature_analysis/identify_task_relevant_features/OpenaiAPI.py`

### The Bug

The `_APISetup.create()` method tracks the number of failed attempts in `tries` but never checks it against `self._retry`:

```python
def create(self, *args, **kwrds):
    tries = 0
    report = False
    while True:                              # ← infinite loop
        try:
            return self._api.create(model=self._model, **kwrds)
        except Exception as e:
            if not report:
                print(("Unkown Error: %s" % e).replace("\n", "\\n"))
                report = True
        time.sleep(self._cool)
        tries += 1                           # ← incremented but never compared to self._retry
```

`self._retry` is stored in `__init__` but referenced nowhere else.

### Why It Matters

If the API returns a persistent error — expired key, model deprecation, bad parameters, or a sustained rate-limit lockout — the script enters an infinite sleep-retry loop. This silently burns Colab session time and API quota with no escape path, requiring a manual kernel interrupt.

### The Fix

```diff
         time.sleep(self._cool)
         tries += 1
+        if self._retry is not None and tries >= self._retry:
+            raise RuntimeError(f"OpenAI API call failed after {tries} retries")
```

---

## Summary Table

| # | Script / Module | Bug | Severity | Impact | Status |
|---|--------|-----|----------|--------|--------|
| 1 | `groupby_textspans.py` | Output path hardcoded `"xxx.tsv"` — missing `%s` placeholder | **High** | Output always overwrites same file; downstream steps can't locate it | 🔍 Audited |
| 2 | `groupby_textspans.py` | `Reader` loads `full.tsv` after dedup instead of `full_deduplicated.tsv` | **Medium** | Deduplication step has zero effect; inflated spans reach Step 2 | 🔍 Audited |
| 3 | `annotate_explanations.py` | Global `KEY` used instead of constructor parameter `key` | **High** | `NameError` crash when imported outside `__main__` | 🔍 Audited |
| 4 | `annotate_explanations.py` | Dead `return raw` in `format()` + method never called in pipeline | **Medium** | Spans sent raw to GPT: no unescape, no artifact removal, no truncation | 🔍 Audited |
| 5 | All 5 Step 3 scripts | Missing `import os` | **Critical** | Instant `NameError` crash at startup for all task annotation scripts | 🔍 Audited |
| 6 | `annotate_toxicity.py` | `format()` defined but `__call__()` bypasses it with raw string join | **Critical** | Toxicity annotations degrade: raw spans with artifacts sent to GPT | 🔍 Audited |
| 7 | 4 of 5 Step 3 scripts + Step 2 Judge | Prompt says "subjective" instead of "objective" | **Medium** | LLM produces opinionated, inconsistent annotations | 🔍 Audited |
| 8 | `annotate_toxicity.py` | Regex `Query-\d+:` only matches hyphen, not space variant | **Medium** | `Query N:` prefix leaks into span text (latent until Bug #6 is fixed) | 🔍 Audited |
| 9 | `OpenaiAPI.py` (both copies) | `max_retry` stored but never enforced — infinite `while True` loop | **High** | Script hangs forever on persistent API errors | 🔍 Audited |
