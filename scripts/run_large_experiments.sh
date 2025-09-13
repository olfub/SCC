#!/bin/bash

export PYTHONPATH=/home/workspace/SCC:/home/workspace/SCC/causal_flows
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Define nodes and edges combinations
declare -a NODE_EDGE_COMBINATIONS=("5-5" "10-12" "15-20" "20-30" "50-100" "100-250")

for combination in "${NODE_EDGE_COMBINATIONS[@]}"; do
    IFS="-" read -r nodes edges <<< "$combination"

    # Generate dataset
    for seed in {0..4}; do
        GPU=$((0 + seed % 5))
        CUDA_VISIBLE_DEVICES=$GPU python datasets/create_large_binary_dataset.py --seed $seed --nodes $nodes --edges $edges --samples_per_int 10000 --path datasets --folder large &
    done
    wait

    # Train cf-SPN
    for seed in {0..4}; do
        GPU=$((0 + seed % 5))
        CUDA_VISIBLE_DEVICES=$GPU python ciSPN/exp_run.py --epochs 100 --dataset LARGE --dataset-params $nodes-$edges --seed $seed --known-intervention &
    done
    wait

    # Train NF own
    for seed in {0..4}; do
        GPU=$((0 + seed % 5))
        CUDA_VISIBLE_DEVICES=$GPU python ciSPN/create_nf_config.py --nodes $nodes --edges $edges --seed $seed &
    done
    wait
    for seed in {0..4}; do
        GPU=$((0 + seed % 5))
        CUDA_VISIBLE_DEVICES=$GPU python ciSPN/exp_run_nf.py \
            --config_file causal_flows/causal_nf/configs/causal_nf_synthetic_${nodes}_${edges}_${seed}.yaml \
            --wandb_mode disabled \
            --project CAUSAL_NF \
            --config_default_file causal_flows/causal_nf/configs/default_config.yaml &
    done
    wait

    # Evaluate cf-SPN
    for seed in {0..4}; do
        GPU=$((0 + seed % 5))
        CUDA_VISIBLE_DEVICES=$GPU python ciSPN/exp_eval.py \
            --dataset LARGE \
            --dataset-params $nodes-$edges \
            --seed $seed \
            --known-intervention \
            --statistics &
    done
    wait
done