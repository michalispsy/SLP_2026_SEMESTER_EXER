import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from sklearn.metrics import average_precision_score
from scipy.special import softmax
from datasets import concatenate_datasets
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Args:
    base_model_dir: str = "MODEL_PATH"
    negative_data_path: str = ""
    synthetic_data_path: str = ""
    valid_data_path: str = ""
    test_data_path: str = ""
    output_dir: str = ""
    seed: int = 42
    num_toxic_samples: int = 200
    num_safe_samples: int = 1000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: float = 3.0
    max_length: int = 512
    bf16: bool = True


parser = HfArgumentParser((Args,))
args = parser.parse_args_into_dataclasses()[0]

print(args.seed)

base_path_prefix = ""
args.base_model_dir = os.path.join(base_path_prefix, "checkpoint")
print("args.base_model_dir", args.base_model_dir)

dataset_name = os.path.basename(args.negative_data_path).replace(".tsv", "")
args.output_dir = os.path.join(args.output_dir, dataset_name, f"seed{args.seed}")
os.makedirs(args.output_dir, exist_ok=True)

# ── Safe logging setup (step4c-style: flush after every write) ──
log_file = os.path.join(args.output_dir, "training_log.jsonl")
progress_file = os.path.join(args.output_dir, "progress.txt")

def safe_log(entry):
    """Append a JSON entry to the log file and flush immediately."""
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()

safe_log({"event": "start", "seed": args.seed, "config": {
    "lr": args.learning_rate, "epochs": args.num_train_epochs,
    "batch_size": args.per_device_train_batch_size,
    "grad_accum": args.gradient_accumulation_steps,
    "max_length": args.max_length,
    "num_toxic_samples": args.num_toxic_samples,
    "num_safe_samples": args.num_safe_samples,
}})


class SafeLoggingCallback(TrainerCallback):
    """Callback that logs metrics to disk with immediate flush (step4c-style)."""

    def on_log(self, _args, state, control, logs=None, **kwargs):
        if logs:
            safe_log({"event": "log", "step": state.global_step, "metrics": logs})

    def on_evaluate(self, _args, state, control, metrics=None, **kwargs):
        if metrics:
            safe_log({"event": "eval", "step": state.global_step, "metrics": metrics})
            print(f"[CHECKPOINT] Step {state.global_step} — AUPRC: {metrics.get('eval_auprc', 'N/A')}")

    def on_save(self, _args, state, control, **kwargs):
        safe_log({"event": "checkpoint_saved", "step": state.global_step})
        # Update progress file (step4c-style)
        with open(progress_file, "w", encoding="utf-8") as pf:
            pf.write(f"last_checkpoint_step={state.global_step}\n")
            pf.write(f"seed={args.seed}\n")
            pf.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "right"
tokenizer.model_max_length = args.max_length


def load_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].astype(int)
    return Dataset.from_dict(df.to_dict(orient="list"))


def load_and_sample_safe_dataset(real_data_path, num_pos_samples=1000, seed=42):
    df = pd.read_csv(real_data_path, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].astype(int)
    pos_samples = df[df["label"] == 0].sample(n=num_pos_samples, random_state=seed)
    return Dataset.from_dict(pos_samples.to_dict(orient="list"))


safe_subset = load_and_sample_safe_dataset(
    real_data_path="",
    num_pos_samples=args.num_safe_samples,
    seed=args.seed,
)

synthetic_toxic_ds = load_tsv(args.synthetic_data_path)

train_ds = concatenate_datasets(
    [safe_subset, synthetic_toxic_ds.shuffle(args.seed).select(range(min(args.num_toxic_samples, len(synthetic_toxic_ds))))]
)
train_ds = train_ds.shuffle(args.seed)
valid_ds = load_tsv(args.valid_data_path)
test_ds = load_tsv(args.test_data_path)


def preprocess(example):
    prompt = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if hasattr(tokenizer, "apply_chat_template")
        else example["text"]
    )
    enc = tokenizer(prompt, truncation=True, padding=False, max_length=args.max_length)
    enc["labels"] = int(example["label"])
    return enc


train_ds = train_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-train")
train_ds = train_ds.shuffle(seed=args.seed)
valid_ds = valid_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-valid")
test_ds = test_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-test")

model = AutoModelForSequenceClassification.from_pretrained(
    args.base_model_dir,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    num_labels=2,
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["score"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)

    save_dir = os.path.join(args.output_dir, "logits")
    os.makedirs(save_dir, exist_ok=True)

    logits_path = os.path.join(save_dir, "eval_logits.tsv")
    print(logits_path)
    np.savetxt(
        logits_path,
        np.column_stack((logits, labels)),
        delimiter="\t",
        fmt="%.6f",
        header="logit0\tlogit1\tlabel",
        comments="",
    )
    print(f"[INFO] Saved logits to: {logits_path}")

    try:
        auprc = float(average_precision_score(labels, softmax(logits, axis=1)[:, 1]))
    except Exception as e:
        auprc = float("nan")
        print(f"[WARN] auprc failed: {e}")

    return {"auprc": auprc}


training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_strategy="steps",
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    bf16=args.bf16,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="auprc",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[SafeLoggingCallback()],
)

# ── Resume from checkpoint if available (step4c-style) ──
last_checkpoint = None
if os.path.isdir(args.output_dir):
    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(args.output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1])
        safe_log({"event": "resume", "checkpoint": last_checkpoint})
        print(f"[RESUME] Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

print("Evaluating on test set…")
metrics = trainer.evaluate(eval_dataset=test_ds)
print(metrics)
safe_log({"event": "final_test", "metrics": metrics})

# Mark completion in progress file
with open(progress_file, "w", encoding="utf-8") as pf:
    pf.write(f"status=completed\n")
    pf.write(f"seed={args.seed}\n")
    pf.write(f"test_auprc={metrics.get('eval_auprc', 'N/A')}\n")
    pf.flush()

print(f"[DONE] Logs saved to: {log_file}")
print(f"[DONE] Progress saved to: {progress_file}")
