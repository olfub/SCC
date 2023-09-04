import argparse
from pathlib import Path

import numpy as np

from external.GalaxyCollision.data_handler import (
    continue_simulation,
    sample_data_random,
    sample_simulation,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

"""
 Galaxy Collision Dataset
 Using a galaxy collision simulation (see external/GalaxyCollision), generate data containing information about the
 position (x and y) and velocity (in x and y direction) of stars and black holes. There are many possible variations,
 like changing the number of stars, the number of black holes and how the stars and black holes are initialized.
 However, these are not provided as parameters, instead this file needs to be changed in that case.

"""

dataset_name = "GC"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True

"""
Data generation approach
Generate observational and interventional data.
Since there are no external, random influences, this makes it possible to calculate counterfactuals by, given the same
input (positions and velocities), calculate the output once for no intervention and once for an intervention.
"""


# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS START
# ----------------------------------------------------------------------------------------------------------------------

rng = np.random.default_rng(seed)

nr_samples = 100000
# current code only works for 2 black holes (because of sample_data_random)
nr_black_holes = 2
nr_vars = (
    nr_black_holes + 1
) * 4  # two black holes and one star, each with x, y, vx, vy
method = "Euler_semi_implicit"  # Euler_explicit, Euler_semi_implicit, Runge_Kutta
data_priors = False

# percentage of interventional (not observational) data points
int_number = 20000
# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS END
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE DATA POINTS START
# ----------------------------------------------------------------------------------------------------------------------

nr_obs = nr_samples
nr_int = int_number
# two extra columns: 1. which value to be intervened on (-1: no intervention), 2. intervention value
data_observations = np.zeros((nr_obs, nr_vars * 2 + 2))
data_interventions = np.zeros((nr_int, nr_vars * 2 + 2))

# data_priors: try to more often generate values which are more likely to occur in simulations
# might be simpler and better to just generate uniformly instead (data_priors False)
if data_priors:
    priors = (True, True, True, True)
else:
    priors = (False, False, False, False)

# observational data
timesteps = 100
data = sample_simulation(timesteps, method=method, seed=42)
nr_particles = int(data.shape[0] / timesteps)
int_time = timesteps // 2
state_for_intervention = np.copy(
    data[int_time * nr_particles : (int_time + 1) * nr_particles, :nr_vars]
)

data_all_obs = np.zeros((data.shape[0], data.shape[1] + 2))
data_all_obs[:, :-2] = data
data_all_obs[:, -2] = -1

# interventional data
i = 0
interventional_batches = []
observational_batches = []
while i < nr_int:
    state_with_intervention = np.copy(state_for_intervention)
    # intervene on a particle position only
    x_or_y = np.array(rng.random(state_with_intervention.shape[0]) > 0.5, dtype=int)
    # intervention value can only be +1 or -1
    intervention_value = (
        (np.array(rng.random(state_with_intervention.shape[0]) > 0.5, dtype=float) * 2)
        - 1
    )[:, np.newaxis]
    # the following line sets the intervention value to the new desired value
    # I think in this setup, it is better and easier to instead just have the delta written there (+1 or -1)
    intervention_delta = np.copy(intervention_value)
    intervention_value += np.take_along_axis(
        state_with_intervention, x_or_y[:, None], axis=1
    )
    data_int_i = np.zeros((nr_particles, nr_vars * 2 + 2))
    data_int_i[:, :nr_vars] = state_with_intervention
    data_int_i[:, -2] = x_or_y
    data_int_i[:, -1] = intervention_delta[:, 0]
    np.put_along_axis(
        state_with_intervention, x_or_y[:, None], intervention_value, axis=1
    )

    data_with_ints = continue_simulation(
        int_time, method=method, seed=42, initial_state=state_with_intervention
    )
    data_int_i[:, nr_vars : nr_vars * 2] = data_with_ints[:nr_particles, nr_vars:]
    interventional_batches.append(data_int_i)

    data_obs_i = np.zeros((nr_particles * (int_time - 1), nr_vars * 2 + 2))
    data_obs_i[:, : nr_vars * 2] = data_with_ints[nr_particles:]
    data_obs_i[:, -2] = -1
    observational_batches.append(data_obs_i)

    i += nr_particles
    print(f"\r{100*i/nr_int:.2f}%", end="")

fourth_of_data = nr_obs // 4
data_interventions = np.concatenate(interventional_batches)[:nr_int]
choose_half_from = data_all_obs[: int_time * nr_particles]
choose_fourth_from = data_all_obs[int_time * nr_particles :]
choose_last_fourth_from = np.concatenate(observational_batches)
# just shuffle all independently and choose the first rows from it until enough are chosen
rng.shuffle(choose_half_from)
rng.shuffle(choose_fourth_from)
rng.shuffle(choose_last_fourth_from)

# more of the ones which are learned worse
first_obs = fourth_of_data
second_obs = int(fourth_of_data * 2.5)
data_observations[:first_obs] = choose_half_from[:first_obs]
data_observations[first_obs:second_obs] = choose_fourth_from[: second_obs - first_obs]
data_observations[second_obs : 4 * fourth_of_data] = choose_last_fourth_from[
    : second_obs - first_obs
]

# if necessary, fill with points before the intervention
rest_points = nr_obs - (fourth_of_data * 4)
if rest_points > 0:
    data_observations[4 * fourth_of_data :] = choose_half_from[
        2 * fourth_of_data : 2 * fourth_of_data + rest_points
    ]
# one last shuffle
rng.shuffle(data_observations)

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE DATA POINTS END
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA START
# ----------------------------------------------------------------------------------------------------------------------

save_dir.mkdir(exist_ok=True, parents=True)

train_percentage = 0.8

# combine and shuffle observational and interventional data
all_data = np.concatenate((data_observations, data_interventions))
rng.shuffle(all_data)

for dset in ["train", "test"]:
    if dset == "train":
        data_save = all_data[: int(all_data.shape[0] * train_percentage)]
    else:
        data_save = all_data[int(all_data.shape[0] * train_percentage) :]

    save_location = save_dir / (
        dataset_name + "_N{}_{}.npy".format(len(data_save), dset)
    )
    np.save(str(save_location), data_save)
    print("Saved Data @ {}".format(save_location))
print("Remember to set the correct method in environment.py")

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA END
# ----------------------------------------------------------------------------------------------------------------------
