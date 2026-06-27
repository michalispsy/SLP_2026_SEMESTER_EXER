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
    forced_fids = []   # features whose span had to be force-injected (no natural hit)
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
            
            # ── Span enforcement setup ───────────────────────────────────
            span_norm = good_span.lower().strip() if good_span else ""
            try:
                mandatory = float(good_score or 0) > 4   # span required when Good Score > 4
            except (TypeError, ValueError):
                mandatory = False

            def _parse(raw):
                """Extract a single cleaned query string from a raw generation, or None."""
                raw = raw.strip()
                if not re.search(r'\t[01]\s*$', raw):
                    raw = raw.rstrip("\n ") + "\t0"
                segs = re.findall(
                    r'(?:^|\n)\s*(Query[- ]\d+\s*:\s*.*?)(?=(?:\n\s*Query[- ]\d+\s*:)|(?:\t\s*[01]\s*$)|$)',
                    raw, flags=re.S)
                if not segs:
                    return None
                tmp = []
                for s in segs:
                    m_idx = re.match(r'\s*Query[- ](\d+)\s*:', s)
                    idx = int(m_idx.group(1)) if m_idx else 999999
                    s_clean = re.sub(r'^\s*Query[- ]\d+\s*:\s*', '', s, flags=re.I).strip()
                    s_clean = s_clean.replace("\r", "").replace("\n", " ").replace("\t", " ")
                    tmp.append((idx, s_clean))
                tmp.sort(key=lambda x: x[0])
                q = re.sub(r'\s+', ' ', " ".join(s for _, s in tmp)).strip()
                return q or None

            # ── Best-of-N generation across temperatures; prefer span hits ─
            accepted, fallback_q = [], None
            MAX_TRIES = 8
            temps = [args.temperature, 0.7, 0.9, 1.1, 0.6, 1.0, 0.8, 1.2]
            for attempt in range(MAX_TRIES):
                if len(accepted) >= 2:
                    break
                response = llama3_generate(user_msg, temperature=temps[attempt % len(temps)],
                                           num_return_sequences=2, feature_content=user_msg)
                if not response:
                    continue
                for text in response:
                    if not text.strip():
                        continue
                    try:
                        q = _parse(text)
                    except Exception as e:
                        print(f"[skip {fid}] {e}")
                        continue
                    if not q:
                        continue
                    if fallback_q is None:
                        fallback_q = q
                    if (not span_norm) or (span_norm in q.lower()):
                        accepted.append(q)
                        if len(accepted) >= 2:
                            break

            # ── Decide what to write ─────────────────────────────────────
            forced = False
            if accepted:
                to_write = accepted[:2]                       # natural span-containing query(ies)
            elif mandatory and good_span:
                # No natural hit: force-inject the activating span so the feature still fires in 4d
                base = fallback_q or ""
                fq = (f"{base.rstrip('.!? ')}. {good_span}".strip() if base else good_span)
                to_write = [re.sub(r'\s+', ' ', fq).strip()]
                forced = True
                forced_fids.append(int(fid))
            elif fallback_q:
                to_write = [fallback_q]                        # span not mandatory: keep best paraphrase
            else:
                to_write = []

            if to_write:
                for slot_idx, qtext in enumerate(to_write):
                    tf.write(f"{qtext}\t1\n")
                    tf.flush()
                    plf.write(json.dumps({
                        "FeatureID": fid,
                        "slot": slot_idx,
                        "query": qtext,
                        "label": 1,
                        "span_hit": bool(span_norm and span_norm in qtext.lower()),
                        "forced": forced,
                        "context_used": context_text[:4000]
                    }, ensure_ascii=False) + "\n")
                    plf.flush()
                print(f"🔸 FEATURE {fid}: wrote {len(to_write)} query(ies) "
                      f"(span_hit={bool(accepted)}, forced={forced})")
            else:
                print(f"[CRITICAL] No query produced for feature {fid}. Using placeholder.")
                error_fids.append(int(fid))
                qtext = f"This is a placeholder toxic query to maintain feature sequence alignment for missing feature {fid}."
                tf.write(f"{qtext}\t1\n")
                tf.flush()
                plf.write(json.dumps({
                    "FeatureID": fid, "slot": 0, "query": qtext, "label": 1,
                    "context_used": context_text[:4000]
                }, ensure_ascii=False) + "\n")
                plf.flush()

            # Record progress
            pf.write(f"{fid}\n")
            pf.flush()
            
            time.sleep(args.sleep)

    print(f"Features with no valid segments (placeholder): {error_fids}")
    print(f"Features that needed span force-injection: {len(forced_fids)} {forced_fids}")
    print(f"[INFO] Saved queries to {tsv_out}")

if __name__ == "__main__":
    main()
