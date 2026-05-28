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
