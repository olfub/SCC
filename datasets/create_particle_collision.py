from pathlib import Path

import numpy as np

from external.particles_in_a_box.data_handler import sample_data

"""
 Particle Collision Dataset
 Using a particle simulation (see external/particles_in_a_box), generate data containing information about the
 particles position (x and y) and velocity (in x and y direction). There are many possible variations, like changing
 the number of particles, the size of the box (the environment) and the strength of other forces. However, these are
 not provided as parameters, instead this file needs to be changed in that case.

"""

dataset_name = "PC"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True
attr_to_index = {"x": 0, "y": 1, "vx": 2, "vy": 3}

"""
Data generation approach
1. Generate data as before (an original, observational sample of the number of frames and then for each desired
intervention another sample for the same number of frames)
2. Save all interventional datapoints (as in one time step to the other, including the intervention) to one variable
3. Add all other datapoints to another variables (purely observational datapoints), this will contain many more
datapoints since in each "sample", only 1 of the number of frames transitions is interventional (rest is observational)
4. Remove non-interventional data points, to have a better ratio of interventional and non-interventional data
Also: add one value indicating the type of intervention (-1 for No-intervention, otherwise the index for the intervened
variable, the interventional value can then be read from the target; this can be done before using the data later)
"""


# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS START
# ----------------------------------------------------------------------------------------------------------------------

box_size = (6, 9)
box_size_x = box_size[0]
box_size_y = box_size[1]
position_limits_x = np.array(
    [-box_size_x / 2 + box_size_x / 10, box_size_x / 2 - box_size_x / 10]
)
position_limits_y = np.array(
    [-box_size_y / 2 + box_size_y / 10, box_size_y / 2 - box_size_y / 10]
)

nr_samples = 200  # per intervention (results in many more samples with a large number of frames and interventions)
nr_frames = 50  # all data is generated using this number of frames, but each consecutive frame pair is one data point
nr_particles = 3
nr_vars = nr_particles * 4  # 4: x, y, vx, vy

# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS END
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# INTERVENTION SETTINGS START
# ----------------------------------------------------------------------------------------------------------------------

int_time = list(range(50))
int_part = list(range(3))
int_attr = ["x", "y", "vx", "vy"]


def int_value_func(which_attr):
    if which_attr == "x":
        limits_min = position_limits_x[0]
        limits_max = position_limits_x[1]
    elif which_attr == "y":
        limits_min = position_limits_y[0]
        limits_max = position_limits_y[1]
    elif which_attr == "vx":
        limits_min = -5
        limits_max = 5
    elif which_attr == "vy":
        limits_min = -5
        limits_max = 5
    else:
        raise RuntimeError(f"Invalid argument value {which_attr}")
    return np.random.uniform(limits_min, limits_max)


nr_interventions = len(int_time) * len(int_part) * len(int_attr)

# ----------------------------------------------------------------------------------------------------------------------
# INTERVENTION SETTINGS END
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# GENERATE DATA POINTS START
# ----------------------------------------------------------------------------------------------------------------------

# ratio of observational to interventional data point (larger = more observational data)
# can be increased by quite a lot relatively easily (as much data is discarded otherwise), but it might be better to
# not have too many non-interventional datapoints compared to interventional ones
obs_to_int = 1
obs_samples_per_samples = int(obs_to_int * nr_interventions)
data_observations = np.zeros((nr_samples * obs_samples_per_samples, nr_vars * 2 + 2))
data_interventions = np.zeros((nr_samples * nr_interventions, nr_vars * 2 + 2))

c = 0  # this is just easier than calculating the index everytime (but that would also be possible)
for i in range(nr_samples):
    # without an intervention (counterfactual is what actually happened), fully included in that data
    original = sample_data(i, nr_frames)
    for j in range(original.shape[0] - 1):
        data_observations[i * obs_samples_per_samples + j, :nr_vars] = original[j]
        data_observations[
            i * obs_samples_per_samples + j, nr_vars : 2 * nr_vars
        ] = original[j + 1]
        # "-1" indicates no intervention, will be processed according to that by the dataset
        data_observations[i * obs_samples_per_samples + j, -2] = -1
        # can leave  data_observations[i*obs_samples_per_samples+j, -1] at 0, there is no intervention value

    # same sample but with an intervention resulting in a counterfactual
    not_interventional = []
    for time in int_time:
        for part in int_part:
            for attr in int_attr:
                int_value = int_value_func(attr)
                cf = sample_data(
                    i, nr_frames, intervention_info=(time, part, attr, int_value)
                )
                # the data point containing the intervention
                data_interventions[c] = np.concatenate(
                    (
                        cf[time],
                        cf[time + 1],
                        np.array([part * 4 + attr_to_index[attr]]),
                        np.array([int_value]),
                    )
                )
                # all other, non-interventional data points (only a part of it is kept later)
                not_interventional.append(cf[:time])
                not_interventional.append(cf[time + 2 :])
                c += 1

    # how many more observational data points to add (the original observation from above is always included)
    obs_to_sample = obs_samples_per_samples - (original.shape[0] - 1)

    # all other observational data to choose from
    obs_data = np.concatenate(not_interventional)

    # sample random data points (here indices, but representing data points)
    samples_ids = np.random.choice(obs_data.shape[0] - 1, obs_to_sample, replace=False)

    # use correct indices
    index_start = i * obs_samples_per_samples + original.shape[0] - 1
    index_end = (i + 1) * obs_samples_per_samples

    # add sampled observational data points to data
    data_observations[index_start:index_end, :nr_vars] = obs_data[samples_ids, :]
    data_observations[index_start:index_end, nr_vars : 2 * nr_vars] = obs_data[
        samples_ids + 1, :
    ]
    # "-1" indicates no intervention, will be processed according to that by the dataset
    data_observations[index_start:index_end, -2] = -1
    # can leave  data_observations[index_start:index_end, -1] at 0, there is no intervention value

    print(f"\r{100*i/nr_samples:.2f}%", end="")

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
np.random.shuffle(all_data)

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

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA END
# ----------------------------------------------------------------------------------------------------------------------
