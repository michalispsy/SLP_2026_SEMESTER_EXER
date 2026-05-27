#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

DATASETS=(
    # "xxx.tsv"
)

NUM_TOXIC_SAMPLES=200
NUM_SAFE_SAMPLES=1000

for dataset_path in "${DATASETS[@]}"
do
    echo "Running with dataset: $dataset_path"

    for seed in 42 43 44
    do
        echo "Running with seed: $seed"
        python finetune_with_synthetic_lora.py \
            --seed $seed \
            --negative_data_path "$dataset_path" \
            --num_toxic_samples $NUM_TOXIC_SAMPLES \
            --num_safe_samples $NUM_SAFE_SAMPLES
    done
done
