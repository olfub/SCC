#!/bin/bash

export PYTHONPATH=/home/workspace/scc
export CUDA_VISIBLE_DEVICES=0

# generate dataset
python datasets/create_galaxyCollision_dataset.py --seed 1

# train model
python ciSPN/E3_large_ds_train.py --model ciSPN --loss NLLLoss --dataset GC --seed 2 --nn_neurons 50 --nn_layers 0 --epochs 100 --batch_size 256 --gradient_clipping 0.5

# create figures
python ciSPN/E3_large_ds_eval.py --model ciSPN --loss NLLLoss --dataset GC --seed 2 --nn_neurons 50 --nn_layers 0 --vis
mkdir -p "./figures/galaxies/"
cp -r "./experiments/E3/outputs/E3_GC_ciSPN_knownI_NLLLoss/2/"* "./figures/galaxies"
