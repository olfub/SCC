# Structural Causal Circuits: Probabilistic Circuits Fully Climbing the Ladder of Causality

## Abstract
The complexity and vastness of our world can require large models with numerous variables. Unfortunately, coming up with a model that is both accurate and able to provide predictions in a reasonable amount of time can prove difficult. One possibility to help overcome such problems are sum-product networks (SPNs), probabilistic models with the ability to do inference in linear time. In this paper, we extend SPNs' capabilities to the field of causality and introduce the family of structural causal circuits (SCCs), a type of SPNs capable of answering causal questions. Starting from conventional SPNs, we "climb the ladder of causation" and show how SCCs can represent not only observational, but also interventional and counterfactual problems. We demonstrate successful application in different experiments.

## Repository Information

### Datasets

In addition to some tabular datasets, there is a dataset on particle collision and another dataset on galaxy collision. These are created using code from two other repositories, namely [particles_in_a_box](https://github.com/ineporozhnii/particles_in_a_box) and [GalaxyCollision](https://github.com/EnguerranVidal/GalaxyCollision/tree/main), respectively. The code of these is slightly adapted and can be found in the `external` folder.

## Running the Code

### Setup the Repository

From inside this repository, run
`docker build -f .docker/Dockerfile -t scc .`

and afterwards connect to the container via
`docker run -it -v /pathtofolder/SCC:/home/workspace/SCC --name scc --gpus all scc`

Navigate to the scc folder (`cd SCC`) and from here you can access the code.

### Run the Paper Experiments

There are several scripts which cover all experiments shown in the paper.

Watering problem (cf-SPN):
`./scripts/run_watering.sh`

Toy problem (iSPN):
`./scripts/run_toy1i.sh`

Toy problem (cf-SPN):
`./scripts/run_toy1.sh`

Noisy toy problem (cf-SPN):
`./scripts/run_toy2.sh`

Particle collision (cf-SPN):
`./scripts/run_particles.sh`

Galaxy collision (cf-SPN):
`./scripts/run_galaxy.sh`

The scripts move the results to the `figures/` folder.

### Run your own Experiments

Files to create datasets are located in the `datasets` folder. Experiments for tabular data can be run and evaluated using the `ciSPN/E1_tabular...` files. For other datasets (implemented are particle collision and galaxy collision), run `ciSPN/E3_large_ds_train.py` and `ciSPN/E3_large_ds_eval.py`.

