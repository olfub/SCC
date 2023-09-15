#!/bin/bash

export PYTHONPATH=/home/workspace/SCC
export CUDA_VISIBLE_DEVICES=13
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# generate dataset
python datasets/create_toy1_dataset_intervention.py
python ciSPN/E1_tabular_train.py --epochs 50 --model ciSPN --loss NLLLoss --dataset TOY1I --seed 0 --known-intervention

# single sample
python ciSPN/E1_tabular_eval2.py --model ciSPN --loss NLLLoss --dataset TOY1I --seed 0 --known-intervention --samples 1 --save
mkdir -p "./figures/toy1i/single_sample"
cp -r "./experiments/E1_2/visualizations/E1_TOY1I_ciSPN_knownI_NLLLoss/"* "./figures/toy1i/single_sample"

# 1000 samples
python ciSPN/E1_tabular_eval2.py --model ciSPN --loss NLLLoss --dataset TOY1I --seed 0 --known-intervention --samples 1000 --save
mkdir -p "./figures/toy1i/1000_samples"
cp -r "./experiments/E1_2/visualizations/E1_TOY1I_ciSPN_knownI_NLLLoss/"* "./figures/toy1i/1000_samples"
