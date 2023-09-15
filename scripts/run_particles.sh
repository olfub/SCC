#!/bin/bash

export PYTHONPATH=/home/workspace/SCC
export CUDA_VISIBLE_DEVICES=13

# generate dataset
python datasets/create_particle_collision.py

# train model
python ciSPN/E3_large_ds_train.py --epochs 100 --model ciSPN --loss NLLLoss --dataset PC --seed 1

# create figures
python ciSPN/E3_large_ds_eval.py --model ciSPN --loss NLLLoss --dataset PC --seed 1 --vis --vis_args 0
python ciSPN/E3_large_ds_eval.py --model ciSPN --loss NLLLoss --dataset PC --seed 1 --vis --vis_args 1
mkdir -p "./figures/particles/"
cp -r "./experiments/E3/outputs/E3_PC_ciSPN_knownI_NLLLoss/1/"* "./figures/particles"
