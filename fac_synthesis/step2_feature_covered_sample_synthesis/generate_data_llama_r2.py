import os
import re
import json
import time
import argparse
import random
from typing import Dict, Any, List, Tuple
from llama_wrapper import llama3_generate
import numpy as np
from tqdm import tqdm

from gen_utils import (
    _extract_json_block,
    _parse_transcript,
    _check_alternating_roles,
    _validate_multi_turn_pair,
    _validate_single_turn_pair,
    parse_instruction_input_pairs,
    _clean_ins_inp,
    _prepend_input_to_human,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--min_exchanges", type=int, default=1)
    parser.add_argument("--max_exchanges", type=int, default=3)
    parser.add_argument("--max_retry_per_question", type=int, default=2)
    parser.add_argument("--num_synthetic_samples", type=int, default=2,
                        help="Number of queries generated per feature (num_return_sequences).")
    args = parser.parse_args()

    features = []
    num_per_feature = 1
    with open(args.features, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            features.append(data)
    if not features:
        raise ValueError("No valid features found.")

    print(f"Loaded {len(features)} features.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Establish output paths and load progress logic
    tsv_out = args.out + ".queries.tsv"
    prompt_log_file = args.out + ".prompts.jsonl"
    progress_file = args.out + ".progress.txt"

    completed_fids = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r") as pf:
            for line in pf:
                if line.strip():
                    completed_fids.add(line.strip())
        print(f"[RESUME] Found {len(completed_fids)} already completed features. Resuming from where we left off.")

    error_fids = []
    for feat in features:
        for key in [
            "Feature Summary", "Good example", "Bad example",
            "Good Span Activated", "Bad Span Activated",
            "Good Activation score", "Bad Activation score"
        ]:
            if key in feat:
                val = feat[key]
                if not isinstance(val, str):
                    feat[key] = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)

    # Open files in append/write mode depending on resumption status
    with open(tsv_out, "a" if completed_fids else "w", encoding="utf-8") as tf, \
         open(prompt_log_file, "a" if completed_fids else "w", encoding="utf-8") as plf, \
         open(progress_file, "a" if completed_fids else "w", encoding="utf-8") as pf:

        for feat in tqdm(features, desc="Generating queries"):
            fid = str(feat.get("feature_id", ""))
            if fid in completed_fids:
                continue

            summary = feat.get("Feature Summary", "").strip()
            good_ex = feat.get("Good example", "").strip()
            bad_ex = feat.get("Bad example", "").strip()
            good_span = feat.get("Good Span Activated", "").strip()
            good_score = feat.get("Good Activation score", "")
            bad_span = feat.get("Bad Span Activated", "").strip()
            bad_score = feat.get("Bad Activation score", "")
            
            context_text = (
                f"Feature Summary: {summary}\n\n"
                f"Good Example:\n{good_ex}\n[Good Span Activated]: {good_span} (Good Score: {good_score})\n\n"
                f"Bad Example:\n{bad_ex}\n[Bad Span Activated]: {bad_span} (Bad Score: {bad_score})"
            )
            
            user_msg = f"{context_text.strip()}"
            
            print(f"\n{'='*60}\n🔹 PROMPT FOR FEATURE {fid}:\n{user_msg}\n{'='*60}")
            
            success = False
            for i in range(num_per_feature):
                response = llama3_generate(user_msg, temperature=args.temperature, num_return_sequences=args.num_synthetic_samples, feature_content=user_msg)

                if not response:
                    print(f"[WARN] Empty response for feature {fid}")
                    continue
                
                # We expect 2 output sequences from num_return_sequences=2
                for slot_idx, text in enumerate(response):
                    if not text.strip():
                        continue
                    text = text.strip()

                    # Fallback trailing label if missing
                    if not re.search(r'\t[01]\s*$', text):
                        text = text.rstrip("\n ") + "\t0"

                    try:
                        label = 1

                        # Relaxed regex to match Query segments
                        segs = re.findall(
                            r'(?:^|\n)\s*(Query[- ]\d+\s*:\s*.*?)(?=(?:\n\s*Query[- ]\d+\s*:)|(?:\t\s*[01]\s*$)|$)',
                            text,
                            flags=re.S
                        )

                        if not segs:
                            print(text)
                            print(f"[WARN] No valid Query segments for feature {fid}, slot {slot_idx}")
                            continue

                        tmp = []
                        for s in segs:
                            m_idx = re.match(r'\s*Query[- ](\d+)\s*:', s)
                            idx = int(m_idx.group(1)) if m_idx else 999999
                            s_clean = re.sub(r'^\s*Query[- ]\d+\s*:\s*', '', s, flags=re.I).strip()
                            s_clean = s_clean.replace("\r", "").replace("\n", " ").replace("\t", " ")
                            tmp.append((idx, s_clean))
                        tmp.sort(key=lambda x: x[0])

                        qtext = " ".join(s for _, s in tmp)
                        qtext = re.sub(r'\s+', ' ', qtext).strip()

                        # Write to files immediately
                        qline = qtext
                        tf.write(f"{qline}\t{label}\n")
                        tf.flush()

                        plf.write(json.dumps({
                            "FeatureID": fid,
                            "slot": slot_idx,
                            "query": qtext,
                            "label": label,
                            "context_used": context_text[:4000]
                        }, ensure_ascii=False) + "\n")
                        plf.flush()

                        print(f"🔸 LLM RESPONSE (Slot {slot_idx}):\n{qtext}\n{'-'*60}")
                        success = True

                    except Exception as e:
                        print(f"[skip {fid}] {e}")

            if not success:
                print(f"[CRITICAL] Failed to generate valid queries for feature {fid}. Using fallback placeholder.")
                error_fids.append(int(fid))
                # Fallback to keep sequence index alignment
                qtext = f"This is a placeholder toxic query to maintain feature sequence alignment for missing feature {fid}."
                qline = qtext
                tf.write(f"{qline}\t1\n")
                tf.flush()
                plf.write(json.dumps({
                    "FeatureID": fid,
                    "slot": 0,
                    "query": qtext,
                    "label": 1,
                    "context_used": context_text[:4000]
                }, ensure_ascii=False) + "\n")
                plf.flush()

            # Record progress
            pf.write(f"{fid}\n")
            pf.flush()
            
            time.sleep(args.sleep)

    print(f"Features with no valid segments: {error_fids}")
    print(f"[INFO] Saved queries to {tsv_out}")

if __name__ == "__main__":
    main()
