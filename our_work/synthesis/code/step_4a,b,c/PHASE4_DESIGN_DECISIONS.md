# Phase 4 Design Decisions

> Αυτό το έγγραφο καταγράφει τις **σχεδιαστικές αποφάσεις** που πάρθηκαν στα Phase 4a (Candidate Generation), 4b (SAE Scoring) και 4c (Contrastive Query Generation) — δηλαδή επιλογές που αντικατοπτρίζουν **πρόθεση**, όχι διορθώσεις σφαλμάτων.

---

## Πίνακας Περιεχομένων

1. [`merge_step1_failed_cases.py` — `srt[-1]` ως "bad" sample στα contrastive pairs](#1-merge_step1_failed_casespy--srt-1-ως-bad-sample-στα-contrastive-pairs)
2. [`generate_data_llama_r1.py` — 2 queries ανά feature στο Step 4a](#2-generate_data_llama_r1py--2-queries-ανά-feature-στο-step-4a)
3. [`llama_wrapper.py` — Uncensored (abliterated) Llama αντί για standard Instruct](#3-llama_wrapperpy--uncensored-abliterated-llama-αντί-για-standard-instruct)
4. [`collect_spans.py` — Αφαίρεση `Query-N:` prefix πριν το SAE scoring](#4-collect_spanspy--αφαίρεση-query-n-prefix-πριν-το-sae-scoring)
5. [`generate_data_llama_r2.py` — Resume / checkpoint mechanism στο Step 4c](#5-generate_data_llama_r2py--resume--checkpoint-mechanism-στο-step-4c)

---

## 1. `merge_step1_failed_cases.py` — `srt[-1]` ως "bad" sample στα contrastive pairs

**File:** [merge_step1_failed_cases.py](../../../../fac_synthesis/step1_contrastive_pair_construction/merge_step1_failed_cases.py)  
**Commit:** `0cab441` (zoetsouroufli)

### Η Απόφαση

Κατά την κατασκευή των contrastive pairs (good/bad), τα activated samples ταξινομούνται φθίνοντα κατά activation score. Αρχικά ως "bad" επιλεγόταν το **δεύτερο** στη λίστα (`srt[1]`). Αποφασίστηκε να χρησιμοποιείται το **τελευταίο** (`srt[-1]`):

```python
# Πριν:
good, bad = srt[0], srt[1]   # best vs. 2nd best
s
# Μετά:
good, bad = srt[0], srt[-1]  # best vs. worst
```

### Γιατί

Ο στόχος των contrastive pairs είναι να μάθει το μοντέλο να **διακρίνει** — η μέγιστη αντίθεση επιτυγχάνεται όταν το "good" sample έχει το υψηλότερο activation score για το feature και το "bad" έχει το χαμηλότερο, όχι απλώς το αμέσως χαμηλότερο.

---

## 2. `generate_data_llama_r1.py` — 5 queries ανά feature στο Step 4a

**File:** [generate_data_llama_r1.py](../../../../fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r1.py) ΓΙΑ ΤΗΝ ΑΚΡΙΒΕΙΑ ΣΤΟ COLLAB ΟΡΙΖΕΤΑΙ
**Commit:** `bd9f7b4` (michalis.psy)

### Η Απόφαση

Το script τρέχει με `--ratio 1.0` (επεξεργάζεται και τα **318 features** χωρίς τυχαία δειγματοληψία) και `--num_synthetic_samples 5` (παράγει **5 synthetic queries ανά feature**):

```bash
python generate_data_llama_r1.py \
  --ratio 1.0 \
  --num_synthetic_samples 5 \
  ...
```

Αποτέλεσμα: **636 queries** συνολικά στο output TSV.

### Γιατί

Η επιλογή `2` αποτελεί ισορροπία μεταξύ ποικιλίας δεδομένων (περισσότερα samples) και πρακτικού χρόνου εκτέλεσης σε GPU. Το `--ratio 1.0` εξασφαλίζει πλήρη κάλυψη όλων των features που εντοπίστηκαν ως "missing" από τα προηγούμενα βήματα.

---

## 3. `llama_wrapper.py` — Uncensored (abliterated) Llama αντί για standard Instruct

**Files:**  
- [fac_synthesis llama_wrapper.py](../../../../fac_synthesis/step1_contrastive_pair_construction/llama_wrapper.py)  
- [behavior_steering llama_wrapper.py](../../../../behavior_steering/llama_wrapper.py)  

**Commit:** `385f87a` (michalis.psy)

### Η Απόφαση

Αλλαγή του base model από το επίσημο `meta-llama/Llama-3.1-8B-Instruct` στο `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`:

```python
# Πριν:
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Μετά:
model_name = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
```

### Γιατί

Το standard `Llama-3.1-8B-Instruct` απορρίπτει συστηματικά τα requests για παραγωγή toxic/harmful queries — αρνείται να ακολουθήσει το few-shot prompt όταν το ζητούμενο περιεχόμενο αφορά βίαιες, παράνομες ή επιβλαβείς ενέργειες. Το "abliterated" variant (από τη μέθοδο [abliteration](https://huggingface.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated)) αφαιρεί τους safety refusal mechanisms διατηρώντας ταυτόχρονα τις instruction-following ικανότητες του base model.

Ισχύει τόσο για το Step 4a όσο και για το Step 4c.

---

## 4. `collect_spans.py` — Αφαίρεση `Query-N:` prefix πριν το SAE scoring

**File:** [collect_spans.py](../../../../sae_feature_analysis/interpret_features/collect_spans.py)  
**Commits:** `9e73df9`, `8c0a950` (zoetsouroufli)

### Η Απόφαση

Πριν το κείμενο κάθε synthetic query δοθεί στο SAE για scoring, αφαιρείται το generation prefix (`Query-1:`, `Query 2:` κτλ.):

```python
# Αφαίρεση Query-N: prefixes (ίδιο pattern με annotate_toxicity.py)
text = re.sub(r"Query[- ]?\d+\s*:\s*", "", text)
```

### Γιατί

Το `Query-1:` είναι **formatting artifact** του generation pipeline — δεν αποτελεί μέρος του σημασιολογικού περιεχομένου της ερώτησης. Αν παραμείνει, το SAE αξιολογεί τα tokens `Query`, `-`, `1`, `:` αντί για το ίδιο το περιεχόμενο, με κίνδυνο να ενεργοποιηθούν features που ανιχνεύουν formatting patterns αντί για τοξικότητα.

---

## 5. `generate_data_llama_r2.py` — Resume / checkpoint mechanism στο Step 4c

**File:** [generate_data_llama_r2.py](../../../../fac_synthesis/step1_contrastive_pair_construction/generate_data_llama_r2.py)  
**Commit:** `fa334cf` (michalis.psy)

### Η Απόφαση

Στο Step 4c (που επεξεργάζεται τα features ένα-ένα σε GPU), υλοποιήθηκε **checkpoint / resume** μηχανισμός:

```python
progress_file = args.out + ".progress.txt"

# Φόρτωση ήδη ολοκληρωμένων features
completed_fids = set()
if os.path.exists(progress_file):
    with open(progress_file, "r") as pf:
        for line in pf:
            if line.strip():
                completed_fids.add(line.strip())

# Skip αν ήδη ολοκληρωμένο
if fid in completed_fids:
    continue

# Καταγραφή μετά από κάθε feature
pf.write(f"{fid}\n")
pf.flush()
```

Τα output αρχεία ανοίγουν σε `append` mode αν υπάρχει ήδη progress, διαφορετικά σε `write`.

### Γιατί

Η εκτέλεση του Step 4c πάνω σε 318+ features με LLM inference διαρκεί αρκετές ώρες. Διακοπή (Colab timeout, OOM error, network loss) χωρίς checkpoint σημαίνει επανέναρξη από μηδέν. Με τον μηχανισμό αυτό η εκτέλεση συνεχίζεται ακριβώς από το feature που σταμάτησε.

---

## Σύνοψη

| # | Αρχείο | Απόφαση | Αιτιολογία |
|---|--------|---------|-----------|
| 1 | `merge_step1_failed_cases.py` | `srt[-1]` αντί `srt[1]` ως "bad" sample | Μέγιστη αντίθεση στα contrastive pairs |
| 2 | `generate_data_llama_r1.py` | 2 queries/feature, ratio 1.0 (318 features) | Ισορροπία ποικιλίας και χρόνου εκτέλεσης |
| 3 | `llama_wrapper.py` (4a + 4c) | Abliterated Llama αντί standard Instruct | Safety filters εμπόδιζαν παραγωγή toxic queries |
| 4 | `collect_spans.py` | Strip `Query-N:` prefix πριν SAE scoring | Καθαρή σημασιολογική αξιολόγηση χωρίς formatting artifacts |
| 5 | `generate_data_llama_r2.py` | Resume/checkpoint mechanism | Fault tolerance σε μακρόχρονες GPU εκτελέσεις |
