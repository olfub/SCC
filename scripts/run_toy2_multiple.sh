#!/bin/bash

export PYTHONPATH=/home/workspace/SCC
export CUDA_VISIBLE_DEVICES=14
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for combination in "1 2" "1 3"; do
    for seed in {0..4}; do
        # create training dataset
        python datasets/create_toy2_dataset_train_multiple.py --nr_int_list $combination --seed "$seed"
        # create evaluation dataset
        python datasets/create_toy2_dataset_eval_multiple.py

        python ciSPN/E5_tabular_multiple_train.py \
            --epochs 50 \
            --model ciSPN \
            --loss NLLLoss \
            --dataset TOY2 \
            --seed "$seed" \
            --known-intervention \
            --nr_int_list $combination

        for nr_intervs in 1 2 3 4 5 6; do
            python ciSPN/E5_tabular_multiple_eval.py \
                --model ciSPN \
                --loss NLLLoss \
                --dataset TOY2 \
                --seed "$seed" \
                --known-intervention \
                --nr-intervs $nr_intervs \
                --nr_int_list $combination
        done
    done
done
