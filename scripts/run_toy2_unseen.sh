#!/bin/bash

export PYTHONPATH=/home/workspace/SCC
export CUDA_VISIBLE_DEVICES=13
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in {0..4}; do
    # create training dataset
    python datasets/create_toy2_dataset.py --seed "$seed"
    # create evaluation dataset
    python datasets/create_toy2_dataset_eval_unseen.py

    remove_worlds_list=(0 8 16 24 32 40 48 56 63)
    for remove_world in "${remove_worlds_list[@]}"; do
        python ciSPN/E4_tabular_unknown_train.py \
            --epochs 50 \
            --model ciSPN \
            --loss NLLLoss \
            --dataset TOY2 \
            --seed "$seed" \
            --known-intervention \
            --remove-worlds "$remove_world"

        python ciSPN/E4_tabular_unknown_eval.py \
            --model ciSPN \
            --loss NLLLoss \
            --dataset TOY2 \
            --seed "$seed" \
            --known-intervention \
            --remove-worlds "$remove_world"
    done
done