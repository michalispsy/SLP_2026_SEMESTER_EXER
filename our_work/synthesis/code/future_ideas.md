# Future Ideas & Architectural Improvements

Αυτό το έγγραφο καταγράφει προτάσεις και ιδέες για τη μελλοντική βελτίωση της αρχιτεκτονικής του pipeline (FAC-Synthesis), οι οποίες προέκυψαν κατά τη διάρκεια της ανάλυσης του κώδικα και της διόρθωσης των bugs.

---

## 1. Cross-Neuron Query Selection (Global Pool για Contrastive Pairs)

**Η τρέχουσα κατάσταση:**  
Στη Φάση 4b (δηλαδή στα αρχεία όπως το `analyze_step1_synthetic_data.py`), όταν το σύστημα αναζητά το "Highest" (Good) και το "Lowest" (Bad) query για να φτιάξει το prompt ενός συγκεκριμένου νευρώνα, **περιορίζεται αυστηρά στα δικά του queries** (σε αυτά που παρήχθησαν ειδικά γι' αυτόν).

**Η ιδέα:**  
Στη θεωρία, θα ήταν εξαιρετική ιδέα να διαλέγει το καλύτερο query **απ' όλη τη δεξαμενή** των διαθέσιμων queries. Αν, για παράδειγμα, ένα query που φτιάχτηκε για τον Νευρώνα Β (π.χ. Ρατσισμός) τυχαίνει να ενεργοποιεί τον Νευρώνα Α (π.χ. Βρισιές) με τεράστιο σκορ (καλύτερο από τα δικά του), θα έπρεπε να χρησιμοποιείται αυτό ως το "Good Example" του Νευρώνα Α. 

Αυτό θα εξασφάλιζε ότι κάθε νευρώνας παίρνει το απολύτως καλύτερο παράδειγμα ενεργοποίησης (global maximum) και θα έλυνε πλήρως το πρόβλημα του "Sequence Misalignment", καθώς η επιλογή δεν θα βασιζόταν σε "τυφλά" indexes.

---

## 2. Σταθερό Context Window (32 Tokens με Ασύμμετρο Διαμοιρασμό)

**Η τρέχουσα κατάσταση:**  
Υπάρχουν ασυνέπειες στο μέγεθος του κειμένου γύρω από τα ενεργοποιημένα tokens (span truncation). Σε κάποια σημεία του κώδικα χρησιμοποιούνται 32 tokens, ενώ σε άλλα (π.χ. στο Step 2 της αρχικής pipeline) τα spans κόβονται υπερβολικά αυστηρά (10 tokens).

**Η ιδέα:**  
Να τυποποιηθεί η εξαγωγή των spans στα **32 tokens παντού** (ακόμη και εκεί που ιστορικά ο κώδικας είχε 10), αλλά με **έξυπνο, ασύμμετρο διαμοιρασμό**. Αντί να μοιράζονται συμμετρικά (π.χ. 15 πριν, 15 μετά), να υιοθετηθεί η κατανομή:

* **20 tokens πριν**
* **11 tokens μετά**  
*(Σύνολο: 20 + 11 + 1 [το ίδιο το token] = 32 tokens)*

**Γιατί;**  
Επειδή η αρχιτεκτονική των Transformers λειτουργεί με αιτιατό (causal) τρόπο, η σημασία (και άρα η ενεργοποίηση του SAE feature) βασίζεται αποκλειστικά στο **προηγούμενο** κείμενο (left context). Τα 20 tokens πριν παρέχουν πολύτιμο context στο LLM (ή στους human annotators) για να καταλάβουν **γιατί** άναψε ο νευρώνας, ενώ τα 11 μετά είναι απλώς βοηθητικά για την ολοκλήρωση της πρότασης.

---

<<<<<<< HEAD
## 3. SAE Filtering of Step 4c Candidates (Post-Synthesis SAE Filter)

**Current state:**  
`generate_data_llama_r2.py` (Step 4c) generates 2 candidates per missing feature but does **not** run the SAE on the Round 2 outputs. There is no verification that the synthesized query actually activates the target neuron — only a prompt-level proxy (the instruction to include the exact span phrase) is used.

**The idea:**  
After Step 4c synthesis, run the SAE on all generated candidates and apply proper filtering and selection:

1. For each missing feature _i_, compute g_i(x) for every candidate query x.
2. **Keep only the single best-performing query per feature** (the one with the highest g_i(x) activation score).
3. **Drop any feature where no candidate achieves g_i(x) > δ** (the activation threshold, default δ = 0.0). This means a feature is dropped entirely if synthesis failed to produce a query that reliably activates it.

**Why this matters:**  
This is what the paper describes in Section 6 and Appendix J but is currently missing from our implementation. The 200 surviving samples reported in the paper are the result of exactly this filter. Without it, we risk adding queries to the training set that do not actually activate the intended SAE neuron — they are topically related at the surface level but miss the latent representation the feature encodes. The filter also naturally keeps the training set small (at most 1 sample per missing feature) and maximally targeted.

---

## 4. SAE-Filtered Task Toxic Data in Training

**Current state:**  
The final training set is: **1,000 helpful (safe, label=0) + ≤200 synthesized toxic (label=1, for F_miss features)**. The Red-Team data that was used as the seed to compute F(Q_Z) is never included in training, which means the model sees zero toxic examples for the features in F(P_Z) ∩ F(Q_Z) — features present in both the anchor and the seed but never synthesized.

**The idea:**  
Use **500 toxic-only queries** (e.g., from the Red-Team subset) as a task dataset and apply SAE filtering to select the most informative representative per activated neuron. Specifically:

1. Take 500 Red-Team (toxic) queries as the task pool.
2. Run the SAE on each query; for each query record which task-relevant toxic features it activates and at what score (g_i(x)).
3. For each activated toxic neuron _i_, **keep only the top-scoring query** (the one with the highest g_i(x) across all 500 queries). Discard duplicates.
4. Add these filtered task-toxic queries to the training set as additional toxic (label=1) examples.

**Resulting training set:**

```
1,000 helpful (safe, label=0)
+ ≤500 filtered task toxic (label=1, best query per activated F(Q_Z) neuron)
+ ≤314 filtered synthesized toxic (label=1, best query per F_miss neuron, from Step 4c + SAE filter above)
```

**Why this matters:**  
This directly addresses the coverage gap identified in the analysis: features in F(P_Z) ∩ F(Q_Z) are currently never seen as labeled-toxic examples during training, even though the anchor confirms they are task-relevant. By selecting the best Red-Team query per activated neuron, we:
- Cover F(P_Z) ∩ F(Q_Z) with real, high-activation toxic examples.
- Keep the dataset small and focused (at most 1 example per neuron, no redundancy).
- Combine naturally with Idea 3: both the synthesized queries (F_miss) and the real queries (F(Q_Z)) are SAE-filtered to the same quality standard.

The combined training set would cover the **entire F(P_Z)** with at least one toxic example per task-relevant neuron, which is the theoretical upper bound of what the anchor tells us the model needs to learn.
=======
## 3. Head-Only Fine-Tuning ως Εναλλακτικό Evaluation Protocol

**Η τρέχουσα κατάσταση:**  
Στο Phase 5 χρησιμοποιούμε **LoRA fine-tuning** (`finetune_with_synthetic_lora.py`) για την εκπαίδευση του toxicity classifier. Αυτό ταιριάζει 100% με τα hyperparameters του paper (Appendix I).

**Η ιδέα:**  
Το paper αναφέρει ρητά δύο evaluation settings (Table 1 = LoRA, Table 9 = Head-Only) και μάλιστα λέει: *"Unless otherwise specified, we report comparisons and ablations on Toxic Detection in the head-only setting, which serves as a linear-probe protocol to directly assess whether synthetic data improves label separability."*

Υπάρχει ήδη το script `finetune_with_synthetic_head_only.py` που:
- **Παγώνει εντελώς** το backbone (freeze all, εκτός `score`/`classifier`)
- Εκπαιδεύει **μόνο** το classification head (Linear layer)
- Hyperparameters: **15 epochs**, lr=**8e-5**, batch=**1**, grad_accum=4 (effective=4), bf16

**Γιατί αξίζει:**  
Η head-only αξιολόγηση είναι **πιο «καθαρή»** ως μέτρηση ποιότητας synthetic data, γιατί δεν αλλάζει τα εσωτερικά representations του μοντέλου — δείχνει αν τα δεδομένα μας πραγματικά βελτιώνουν τη **γραμμική διαχωρισιμότητα** (label separability) στον feature space του LLM. Αν τα FAC-guided data βελτιώνουν το AUPRC στο head-only setting, σημαίνει ότι τα synthetic queries ενεργοποιούν features που πραγματικά βοηθούν τη διάκριση toxic vs safe.

**Πρακτικά:**  
Να τρέξουμε **και τα δύο** (LoRA + Head-Only) και να συγκρίνουμε τα AUPRC, ώστε να έχουμε πλήρη εικόνα — ακριβώς όπως στο paper (Table 1 vs Table 9).

> **Σημείωση:** Το `finetune_with_synthetic_head_only.py` έχει bugs (κενά paths, λάθος όνομα function `load_and_safe_toxic_dataset` αντί `load_and_sample_safe_dataset`). Πρέπει να διορθωθούν πριν τρέξει.

---

## 4. Re-run Pipeline με LLM-Synthesized Initial Blind Data (σύμφωνα με Paper)

**Η τρέχουσα κατάσταση:**  
Στο Phase 3 (`identify_fac.py`) χρειαζόμαστε δύο σύνολα features: **anchor** (από τα πραγματικά δεδομένα) και **baseline** (από τα αρχικά "τυφλά" synthetic data). Εμείς τρέξαμε τη Phase 3 χρησιμοποιώντας **πραγματικά δεδομένα** ως baseline αντί για LLM-generated synthetic data.

**Τι κάνει πραγματικά το paper:**  
Στο paper, το baseline dataset είναι **seed-initialized synthetic data** — δηλαδή toxic queries που παρήχθησαν "τυφλά" από ένα LLM (χωρίς κανέναν οδηγό SAE features). Αυτά τα τυφλά synthetic data:
1. Τρέφονται στο Phase 2 (annotation pipeline) για να πάρουμε τα baseline features
2. Στο Phase 3, αφαιρούνται τα features που **ήδη καλύπτουν** αυτά τα naive synthetic data
3. Τα missing features (F_miss) αντιπροσωπεύουν αυτά που η **naive σύνθεση αποτυγχάνει** να καλύψει
4. Στο Phase 5, τα naive synthetic data **ΔΕΝ πετιούνται** — συνδυάζονται μαζί με τα FAC-guided data στο τελικό training set

Αυτή η μεθοδολογία σημαίνει ότι τα missing features αντικατοπτρίζουν τα **πραγματικά κενά της LLM synthesis**, ενώ στη δική μας εκτέλεση αντικατοπτρίζουν τα κενά σε σχέση με πραγματικά (human-generated) δεδομένα.

**Η ιδέα — 2 φάσεις:**

**Φάση Α — Δημιουργία Initial Blind Synthetic Data:**
- Χρησιμοποιώντας το abliterated Llama (ή οποιοδήποτε LLM), παράγουμε **naive toxic queries** χωρίς SAE guidance — π.χ. με ένα generic prompt τύπου "Generate diverse toxic user queries" ή χρησιμοποιώντας ένα seed set (HH-RLHF Red-Team prompts)
- Αυτά γίνονται τα "initial blind synthetic data"

**Φάση Β — Re-run ολόκληρου του Pipeline:**
1. Phase 2 τρέχει **ξανά** πάνω στα blind synthetic data → baseline features
2. Phase 3 ξαναϋπολογίζει τα F_miss = F_anchor ∖ F_baseline (τώρα πιο ρεαλιστικά)
3. Phase 4 (a,b,c) τρέχει κανονικά → νέα FAC-guided data
4. Phase 5: training set = blind synthetic + FAC-guided + real samples

**Γιατί αξίζει:**
- Πιο **πιστή αναπαραγωγή** του paper
- Τα missing features θα αντιπροσωπεύουν τα πραγματικά κενά της LLM synthesis (και όχι "τι λείπει σε σχέση με τον άνθρωπο")
- Μεγαλύτερο training set (blind + FAC-guided μαζί)
- Δυνατότητα **iterative self-improvement** (Round 2 synthesis, όπως αναφέρει το paper)

> **Σημείωση:** Αυτό απαιτεί σημαντικό compute time (ξανατρέξιμο Phase 2 + 3 + 4), αλλά είναι η σωστή μεθοδολογία.

---

## 5. Βελτιωμένη Επιλογή των 200 Synthetic Toxic Samples (Αντί για Τυχαία Επιλογή)

**Η τρέχουσα κατάσταση:**  
Στη Φάση 5 (Finetuning), το script επιλέγει τυχαία τα 200 synthetic toxic samples από τη συνολική δεξαμενή των 636 παραγόμενων queries (μέσω shuffle).

**Η ιδέα:**  
Τα 200 δείγματα που θα αποτελέσουν το τελικό training set δεν θα πρέπει να επιλέγονται τυχαία, αλλά με στρατηγικό τρόπο ώστε να μεγιστοποιείται το diversity και η κάλυψη των missing features. Προτείνονται δύο προσεγγίσεις:

1. **Επιλογή με βάση το Score και τη "Γνησιότητα" (Ανά Feature):** Να γίνει ένα φιλτράρισμα κρατώντας τα **γνήσια synthetic** queries με τα υψηλότερα scores. Είναι κρίσιμο **να αποκλειστούν** τα queries όπου τα παραγόμενα δεδομένα απέτυχαν να ενεργοποιήσουν επαρκώς το feature και αναγκαστήκαμε να τα αντικαταστήσουμε/κάνουμε fallback στα αρχικά (human-written) spans με το τέλειο σκορ 5.0. Επιλέγοντας ένα πραγματικά συνθετικό query από κάθε ζεύγος/feature εξασφαλίζουμε και ομοιόμορφη κάλυψη, και ότι αξιολογούμε αυστηρά τη συνθετική ικανότητα του μοντέλου (χωρίς να "κλέβουμε" με πραγματικά δεδομένα).
2. **SAE Threshold Filtering & Ranking (Προσέγγιση του Paper):** Όπως αναφέρεται ρητά στο paper, τα παραγόμενα samples πρέπει να περνάνε **ξανά** μέσα από τον SAE. Τα δείγματα φιλτράρονται με ένα σταθερό activation threshold ($\delta$) και διατηρούνται μόνο όσα ενεργοποιούν ικανοποιητικά το στοχευμένο missing feature. Έπειτα, τα candidates κατατάσσονται (ranking) και επιλέγονται μόνο τα κορυφαία για να μπουν στο τελικό pool.

**Γιατί αξίζει:**  
Όπως εξηγεί η θεωρία του FAC Synthesis, αυτή η "constrained generation" εξασφαλίζει ότι τα synthetic samples περιέχουν **πράγματι** τα target features που λείπουν. Αυτό μειώνει το "estimation error caused by limited sampling" και ρίχνει το conditional entropy, καθιστώντας τα 200 τελικά samples εξαιρετικά πιο στοχευμένα και αποτελεσματικά σε σχέση με την απλή τυχαία επιλογή.
