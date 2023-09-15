#!/bin/bash

export PYTHONPATH=/home/workspace/SCC
export CUDA_VISIBLE_DEVICES=13
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# generate dataset
python datasets/create_toy2_dataset.py
python ciSPN/E1_tabular_train.py --epochs 50 --model ciSPN --loss NLLLoss --dataset TOY2 --seed 0 --known-intervention

# 1000 samples
python ciSPN/E1_tabular_eval2.py --model ciSPN --loss NLLLoss --dataset TOY2 --seed 0 --known-intervention --samples 1000 --skip_samples 0 --save
mkdir -p "./figures/toy2/1000_samples"
cp -r "./experiments/E1_2/visualizations/E1_TOY2_ciSPN_knownI_NLLLoss/"* "./figures/toy2/1000_samples"
