#!/usr/bin/env python3
"""Generate the Step 5 Colab notebook as JSON."""
import json

cells = []

def md(source, cell_id=""):
    cells.append({"cell_type": "markdown", "metadata": {"id": cell_id}, "source": source})

def code(source, cell_id=""):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"id": cell_id}, "outputs": [], "source": source})

# ── Title ──
md([
    "# Phase 5: Toxicity Detection Finetuning\n",
    "\n",
    "Αυτό το notebook εκπαιδεύει έναν **toxicity classifier** πάνω στο Llama-3.1-8B χρησιμοποιώντας τα synthetic toxic queries από τα Steps 4a-4c.\n",
    "\n",
    "### Στρατηγική\n",
    "- **LoRA finetuning** (r=8) σε όλα τα attention + MLP layers\n",
    "- Classification head (`score`) εκπαιδεύεται πλήρως\n",
    "- Training data: **200 synthetic toxic** + **1000 safe** samples\n",
    "- Αξιολόγηση: **AUPRC** (Area Under Precision-Recall Curve)\n",
    "- Multi-seed: τρέχει 3 φορές (seeds 42, 43, 44)\n",
    "\n",
    "### Εκτιμώμενος Χρόνος\n",
    "- T4 GPU: ~15-25 λεπτά ανά seed\n",
    "- A100: ~3-5 λεπτά ανά seed\n",
], "step5_intro")

# ── Cell 1: Drive + GPU ──
code([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Verify GPU\n",
    "!nvidia-smi\n",
], "step5_drive_gpu")

# ── Cell 2: Install ──
md(["## 1. Εγκατάσταση Βιβλιοθηκών"], "step5_install_header")
code([
    "!pip install -q transformers accelerate peft bitsandbytes datasets scikit-learn\n",
], "step5_install")

# ── Cell 3: HF Login ──
md(["## 2. Σύνδεση με Hugging Face"], "step5_hf_header")
code([
    "from google.colab import userdata\n",
    "from huggingface_hub import login\n",
    "\n",
    "hf_token = userdata.get('HF_TOKEN')\n",
    "login(hf_token)\n",
], "step5_hf_login")

# ── Cell 4: Config ──
md([
    "## 3. Configuration\n",
    "\n",
    "Ορίζουμε τα paths και hyperparameters. **Άλλαξε τα paths ανάλογα με τη δομή του Drive σου.**\n",
], "step5_config_header")

code([
    "import os\n",
    "\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "# ΑΛΛΑΞΕ ΑΥΤΑ ΤΑ PATHS\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "\n",
    "# Synthetic toxic queries (output από Steps 4a + 4c)\n",
    "# Μπορείς να χρησιμοποιήσεις είτε step1 (4a) είτε step2 (4c) είτε και τα δύο merged\n",
    "SYNTHETIC_TOXIC_TSV = '/content/drive/MyDrive/fac_synthesis/step_4/4c/log_files/step2_queries.queries.tsv'\n",
    "\n",
    "# Safe dataset (HH-RLHF helpful-base ή παρόμοιο)\n",
    "# Format: text<TAB>label (label=0 για safe)\n",
    "SAFE_DATA_TSV = '/content/drive/MyDrive/fac_synthesis/step_5/safe_data.tsv'\n",
    "\n",
    "# Validation & Test sets\n",
    "VALID_DATA_TSV = '/content/drive/MyDrive/fac_synthesis/step_5/valid.tsv'\n",
    "TEST_DATA_TSV  = '/content/drive/MyDrive/fac_synthesis/step_5/test.tsv'\n",
    "\n",
    "# Output directory\n",
    "OUTPUT_DIR = '/content/drive/MyDrive/fac_synthesis/step_5/output'\n",
    "\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "# HYPERPARAMETERS (defaults from the paper)\n",
    "# ═══════════════════════════════════════════════════════════════\n",
    "BASE_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'\n",
    "NUM_SAFE_SAMPLES = 1000     # πόσα safe samples να χρησιμοποιηθούν\n",
    "NUM_TOXIC_SAMPLES = 200     # πόσα synthetic toxic samples\n",
    "MAX_LENGTH = 512\n",
    "BATCH_SIZE = 4\n",
    "GRAD_ACCUM = 4\n",
    "LR = 5e-5\n",
    "EPOCHS = 3\n",
    "SEEDS = [42, 43, 44]\n",
    "\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "print('✅ Configuration set.')\n",
], "step5_config")

# ── Cell 5: Data Loading ──
md([
    "## 4. Φόρτωση & Προετοιμασία Δεδομένων\n",
    "\n",
    "Φορτώνουμε τα synthetic toxic queries και τα safe samples, τα ενώνουμε και τα ανακατεύουμε.\n",
], "step5_data_header")

code([
    "import pandas as pd\n",
    "import numpy as np\n",
    "from datasets import Dataset, concatenate_datasets\n",
    "\n",
    "def load_tsv(path):\n",
    "    \"\"\"Load a TSV file with columns: text, label.\"\"\"\n",
    "    df = pd.read_csv(path, sep='\\t', header=None, names=['text', 'label'])\n",
    "    df['label'] = df['label'].astype(int)\n",
    "    print(f'  Loaded {len(df)} rows from {os.path.basename(path)} '\n",
    "          f'(label=0: {(df.label==0).sum()}, label=1: {(df.label==1).sum()})')\n",
    "    return Dataset.from_dict(df.to_dict(orient='list'))\n",
    "\n",
    "def load_and_sample(path, label_value, n, seed):\n",
    "    \"\"\"Load TSV and sample n rows with the given label.\"\"\"\n",
    "    df = pd.read_csv(path, sep='\\t', header=None, names=['text', 'label'])\n",
    "    df['label'] = df['label'].astype(int)\n",
    "    subset = df[df['label'] == label_value]\n",
    "    if len(subset) < n:\n",
    "        print(f'  ⚠️ Only {len(subset)} samples with label={label_value}, using all.')\n",
    "        sampled = subset\n",
    "    else:\n",
    "        sampled = subset.sample(n=n, random_state=seed)\n",
    "    return Dataset.from_dict(sampled.to_dict(orient='list'))\n",
    "\n",
    "# Preview the synthetic data\n",
    "print('📊 Synthetic toxic queries:')\n",
    "toxic_ds = load_tsv(SYNTHETIC_TOXIC_TSV)\n",
    "print(f'\\n📊 First 3 examples:')\n",
    "for i in range(min(3, len(toxic_ds))):\n",
    "    txt = toxic_ds[i]['text'][:120]\n",
    "    print(f'  [{i}] (label={toxic_ds[i][\"label\"]}) {txt}...')\n",
], "step5_data_load")

# ── Cell 6: Model + Tokenizer ──
md([
    "## 5. Φόρτωση Μοντέλου & Tokenizer\n",
    "\n",
    "Φορτώνουμε το Llama-3.1-8B-Instruct ως **sequence classifier** (2 labels) με 4-bit quantization για T4 compatibility.\n",
], "step5_model_header")

code([
    "import torch\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModelForSequenceClassification,\n",
    "    BitsAndBytesConfig,\n",
    ")\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "# Tokenizer\n",
    "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)\n",
    "if tokenizer.pad_token is None:\n",
    "    tokenizer.add_special_tokens({'pad_token': '[PAD]'})\n",
    "tokenizer.truncation_side = 'right'\n",
    "tokenizer.model_max_length = MAX_LENGTH\n",
    "\n",
    "# 4-bit quantization for T4 GPU\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
    "    bnb_4bit_quant_type='nf4',\n",
    ")\n",
    "\n",
    "# Model as classifier\n",
    "model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    BASE_MODEL,\n",
    "    num_labels=2,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map='auto',\n",
    "    torch_dtype=torch.bfloat16,\n",
    ")\n",
    "model.resize_token_embeddings(len(tokenizer))\n",
    "model.config.pad_token_id = tokenizer.pad_token_id\n",
    "model.gradient_checkpointing_enable()\n",
    "\n",
    "# LoRA adapters\n",
    "peft_config = LoraConfig(\n",
    "    task_type=TaskType.SEQ_CLS,\n",
    "    inference_mode=False,\n",
    "    r=8,\n",
    "    lora_alpha=16,\n",
    "    lora_dropout=0.1,\n",
    "    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',\n",
    "                    'gate_proj', 'up_proj', 'down_proj'],\n",
    "    modules_to_save=['score'],\n",
    ")\n",
    "\n",
    "model = get_peft_model(model, peft_config)\n",
    "model.print_trainable_parameters()\n",
    "print('✅ Model loaded with LoRA adapters.')\n",
], "step5_model_load")

# ── Cell 7: Training Function ──
md([
    "## 6. Training Loop\n",
    "\n",
    "Ορίζουμε τη συνάρτηση εκπαίδευσης που τρέχει για κάθε seed. Χρησιμοποιεί τον HuggingFace `Trainer` με AUPRC ως evaluation metric.\n",
], "step5_train_header")

code([
    "import random\n",
    "from transformers import Trainer, TrainingArguments\n",
    "from sklearn.metrics import average_precision_score\n",
    "from scipy.special import softmax\n",
    "\n",
    "def set_seed(seed):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "def preprocess(example):\n",
    "    \"\"\"Wrap text in Llama chat template, then tokenize.\"\"\"\n",
    "    prompt = tokenizer.apply_chat_template(\n",
    "        [{'role': 'user', 'content': example['text']}],\n",
    "        tokenize=False,\n",
    "        add_generation_prompt=False,\n",
    "    )\n",
    "    enc = tokenizer(prompt, truncation=True, padding=False, max_length=MAX_LENGTH)\n",
    "    enc['labels'] = int(example['label'])\n",
    "    return enc\n",
    "\n",
    "def compute_metrics(eval_pred):\n",
    "    logits, labels = eval_pred\n",
    "    logits = np.array(logits)\n",
    "    labels = np.array(labels)\n",
    "    try:\n",
    "        auprc = float(average_precision_score(labels, softmax(logits, axis=1)[:, 1]))\n",
    "    except Exception as e:\n",
    "        auprc = float('nan')\n",
    "        print(f'[WARN] AUPRC failed: {e}')\n",
    "    return {'auprc': auprc}\n",
    "\n",
    "def run_training(seed):\n",
    "    \"\"\"Run one full training + evaluation cycle for a given seed.\"\"\"\n",
    "    print(f'\\n{\"=\"*60}')\n",
    "    print(f'🔹 TRAINING WITH SEED {seed}')\n",
    "    print(f'{\"=\"*60}')\n",
    "    set_seed(seed)\n",
    "\n",
    "    # Build training set: safe + toxic\n",
    "    safe_ds = load_and_sample(SAFE_DATA_TSV, label_value=0, n=NUM_SAFE_SAMPLES, seed=seed)\n",
    "    toxic_full = load_tsv(SYNTHETIC_TOXIC_TSV)\n",
    "    if len(toxic_full) > NUM_TOXIC_SAMPLES:\n",
    "        toxic_ds = toxic_full.shuffle(seed).select(range(NUM_TOXIC_SAMPLES))\n",
    "    else:\n",
    "        toxic_ds = toxic_full\n",
    "    train_ds = concatenate_datasets([safe_ds, toxic_ds]).shuffle(seed)\n",
    "\n",
    "    # Load val/test\n",
    "    valid_ds = load_tsv(VALID_DATA_TSV)\n",
    "    test_ds = load_tsv(TEST_DATA_TSV)\n",
    "\n",
    "    # Tokenize\n",
    "    train_tok = train_ds.map(preprocess, remove_columns=['text', 'label'], num_proc=2, desc='tok-train')\n",
    "    valid_tok = valid_ds.map(preprocess, remove_columns=['text', 'label'], num_proc=2, desc='tok-valid')\n",
    "    test_tok = test_ds.map(preprocess, remove_columns=['text', 'label'], num_proc=2, desc='tok-test')\n",
    "\n",
    "    seed_output = os.path.join(OUTPUT_DIR, f'seed{seed}')\n",
    "\n",
    "    training_args = TrainingArguments(\n",
    "        output_dir=seed_output,\n",
    "        per_device_train_batch_size=BATCH_SIZE,\n",
    "        per_device_eval_batch_size=2,\n",
    "        gradient_accumulation_steps=GRAD_ACCUM,\n",
    "        num_train_epochs=EPOCHS,\n",
    "        learning_rate=LR,\n",
    "        save_strategy='no',\n",
    "        evaluation_strategy='steps',\n",
    "        eval_steps=50,\n",
    "        logging_steps=10,\n",
    "        bf16=True,\n",
    "        report_to='none',\n",
    "        save_total_limit=1,\n",
    "    )\n",
    "\n",
    "    trainer = Trainer(\n",
    "        model=model,\n",
    "        args=training_args,\n",
    "        train_dataset=train_tok,\n",
    "        eval_dataset=valid_tok,\n",
    "        tokenizer=tokenizer,\n",
    "        compute_metrics=compute_metrics,\n",
    "    )\n",
    "\n",
    "    trainer.train()\n",
    "\n",
    "    # Test evaluation\n",
    "    print(f'\\n📊 Evaluating on test set (seed={seed})...')\n",
    "    metrics = trainer.evaluate(eval_dataset=test_tok)\n",
    "    print(f'✅ Test AUPRC: {metrics.get(\"eval_auprc\", \"N/A\")}')\n",
    "\n",
    "    # Save logits\n",
    "    logits_dir = os.path.join(seed_output, 'logits')\n",
    "    os.makedirs(logits_dir, exist_ok=True)\n",
    "    preds = trainer.predict(test_tok)\n",
    "    np.savetxt(\n",
    "        os.path.join(logits_dir, 'test_logits.tsv'),\n",
    "        np.column_stack((preds.predictions, preds.label_ids)),\n",
    "        delimiter='\\t', fmt='%.6f',\n",
    "        header='logit0\\tlogit1\\tlabel', comments='',\n",
    "    )\n",
    "    print(f'💾 Saved logits to {logits_dir}')\n",
    "    return metrics\n",
    "\n",
    "print('✅ Training function defined.')\n",
], "step5_train_fn")

# ── Cell 8: Run All Seeds ──
md([
    "## 7. Εκτέλεση (3 Seeds)\n",
    "\n",
    "Τρέχουμε το training για κάθε seed και συλλέγουμε τα αποτελέσματα.\n",
], "step5_run_header")

code([
    "all_results = {}\n",
    "\n",
    "for seed in SEEDS:\n",
    "    metrics = run_training(seed)\n",
    "    all_results[seed] = metrics\n",
    "\n",
    "print(f'\\n{\"=\"*60}')\n",
    "print('📊 ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ')\n",
    "print(f'{\"=\"*60}')\n",
    "auprc_values = []\n",
    "for seed, m in all_results.items():\n",
    "    auprc = m.get('eval_auprc', float('nan'))\n",
    "    auprc_values.append(auprc)\n",
    "    print(f'  Seed {seed}: AUPRC = {auprc:.4f}')\n",
    "\n",
    "mean_auprc = np.mean(auprc_values)\n",
    "std_auprc = np.std(auprc_values)\n",
    "print(f'\\n  Mean AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}')\n",
    "print(f'\\n✅ ΤΕΛΟΣ Step 5!')\n",
], "step5_run_all")

# ── Cell 9: Summary ──
md([
    "---\n",
    "\n",
    "## Σύνοψη Pipeline\n",
    "\n",
    "```\n",
    "Steps 1-3: SAE Feature Analysis → 318 missing toxic features\n",
    "Step 4a:   Candidate Generation → 1,590 raw toxic queries\n",
    "Step 4b:   SAE Scoring          → Contrastive pairs (good/bad)\n",
    "Step 4c:   Refined Generation   → 636 refined toxic queries\n",
    "Step 5:    Finetuning           → LoRA classifier (AUPRC eval)  ◄── DONE\n",
    "```\n",
], "step5_summary")

# ── Build notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = "/home/michalis/Documents/ece_ntua/8th/speech/semester_ex/FAC-Synthesis/our_work/synthesis/code/step5/step5_finetuning.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"✅ Notebook saved to: {out_path}")
