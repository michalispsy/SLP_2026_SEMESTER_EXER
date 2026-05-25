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
    parser.add_argument("--max_retry_per_question", type=int, default=10)
    parser.add_argument("--num_synthetic_samples", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.2, help="Sampling ratio (e.g., 0.2)")
    parser.add_argument("--max-features", type=int, default=None, help="Max number of features to run (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    features = []
    num_per_feature = 1
    with open(args.features, "r", encoding="utf-8", errors="ignore") as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                if "FeatureID" in line and "Summary" in line:
                    header_skipped = True
                    continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fid = parts[0].strip()
            summary = parts[1].strip().strip('"').strip("'").replace('""', '"')
            words = parts[2].strip() if len(parts) > 2 else ""
            if fid.isdigit():
                features.append({"FeatureID": fid, "Summary": summary, "Words": words})
    if not features:
        raise ValueError("No valid features found.")

    print(f"Loaded {len(features)} features.")

    if args.max_features is not None:
        features = features[:args.max_features]
        print(f"[INFO] Limited to {len(features)} features for testing.")

    if args.ratio < 1.0:
        sample_size = int(len(features) * args.ratio)
        features = random.sample(features, sample_size)
        print(f"[INFO] Subsampled to {len(features)} features (ratio={args.ratio}, seed={args.seed})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

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

    all_tasks = []
    error_fids = []
    
    with open(tsv_out, "a" if completed_fids else "w", encoding="utf-8") as tf, \
         open(prompt_log_file, "a" if completed_fids else "w", encoding="utf-8") as plf, \
         open(progress_file, "a" if completed_fids else "w", encoding="utf-8") as pf:

        for feat in tqdm(features, desc="Generating queries"):
            fid = str(feat["FeatureID"])
            if fid in completed_fids:
                continue

            summary = feat.get("Summary", "").strip()
            spans = feat.get("Words", "").strip()
            spans_clean = re.sub(r"\s+", " ", spans)
            if spans_clean:
                context_text = f"Feature Summary: {summary}\nExample Spans:\n{spans_clean}"
            else:
                context_text = f"Feature Summary: {summary}"
            feature_content = context_text.strip()
            user_msg = feature_content
            
            print(f"\n{'='*60}\n🔹 PROMPT FOR FEATURE {fid}:\n{user_msg}\n{'='*60}")
            
            feature_q_records = []
            
            # Guarantee exactly args.num_synthetic_samples generated queries per feature
            for slot in range(args.num_synthetic_samples):
                success = False
                qtext = ""
                
                # Retry loop: exactly 5 retries as requested
                for attempt in range(1, 6):
                    try:
                        response = llama3_generate(
                            user_msg,
                            temperature=args.temperature,
                            num_return_sequences=1,
                            feature_content=feature_content
                        )
                        
                        if not response or not response[0].strip():
                            print(f"[WARN] Empty response for FeatureID {fid}, slot {slot}, attempt {attempt}")
                            continue
                            
                        text = response[0].strip()

                        label = 1
                        
                        # Relaxed regex to catch "Query-1", "Query 1", etc.
                        segs = re.findall(
                            r'(?:^|\n)\s*(Query[- ]\d+\s*:\s*.*?)(?=(?:\n\s*Query[- ]\d+\s*:)|(?:\t\s*[01]\s*$)|$)',
                            text,
                            flags=re.S
                        )
                        
                        if not segs:
                            print(f"[WARN] No valid Query segments for FeatureID {fid}, slot {slot}, attempt {attempt}")
                            continue
                            
                        tmp = []
                        for s in segs:
                            m_idx = re.match(r'\s*Query[- ](\d+)\s*:', s)
                            idx = int(m_idx.group(1)) if m_idx else 999999
                            tmp.append((idx, s.strip()))
                        tmp.sort(key=lambda x: x[0])
                        
                        qtext = "\n".join(s for _, s in tmp)
                        
                        if qtext.strip():
                            print(f"🔸 LLM RESPONSE (Slot {slot}):\n{qtext}\n{'-'*60}")
                            plf.write(json.dumps({
                                "FeatureID": fid,
                                "slot": slot,
                                "prompt": user_msg,
                                "response": qtext
                            }, ensure_ascii=False) + "\n")
                            plf.flush()
                            success = True
                            break # Successfully generated and parsed, break out of retry loop

                    except Exception as e:
                        print(f"[ERROR] Exception on FeatureID {fid}, slot {slot}, attempt {attempt}: {e}")
                        
                if not success:
                    print(f"[CRITICAL] All 5 attempts failed for FeatureID {fid}, slot {slot}. Using fallback placeholder.")
                    if int(fid) not in error_fids:
                        error_fids.append(int(fid))
                    # Fallback placeholder to maintain sequence index alignment
                    qtext = f"Query-1: This is a placeholder toxic query to maintain feature sequence alignment for missing feature {fid}."
                    print(f"🔸 FALLBACK RESPONSE (Slot {slot}):\n{qtext}\n{'-'*60}")
                    
                label = 1
                all_tasks.append((qtext, str(label)))
                feature_q_records.append({
                    "FeatureID": fid,
                    "query": qtext,
                    "label": label,
                    "context_used": context_text[:4000]
                })

            for rec in feature_q_records:
                qline = rec["query"].replace("\r", "").replace("\t", " ").replace("\n", "\\n")
                tf.write(f"{qline}\t{rec['label']}\n")
            tf.flush()
            
            pf.write(f"{fid}\n")
            pf.flush()
            
            time.sleep(args.sleep)

    print(f"Features with no valid segments: {error_fids}")
    print(f"[INFO] Saved queries to {tsv_out} and prompts to {prompt_log_file}")

    if not all_tasks and not completed_fids:
        raise RuntimeError("No queries generated.")


if __name__ == "__main__":
    main()

