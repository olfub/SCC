import numpy as np

def save_TOY2_eval(data, dataset_name, variable_names, intervention_vars, save_dir):
    # save as array: intervention, original world, counterfactual world 
    save_dir.mkdir(exist_ok=True, parents=True)
    datasets = []
    for interv_and_data in data:
        interv_desc, d = interv_and_data
        interv_vector = np.zeros(len(intervention_vars) + 1)
        if interv_desc != "None":
            interv_var = interv_desc.split("(")[1]
            interv_var = interv_var.split(")")[0]
            interv_val = interv_desc.split("=")[1]
            interv_vector[intervention_vars.index(interv_var)] = 1
            interv_vector[-1] = int(interv_val)
        data_orig_worlds = np.array([d[key] for key in variable_names if "cf" not in key and key != "A" and key != "B"])
        data_cf_worlds = np.array([d[key] for key in variable_names if "cf" in key and key != "A-cf" and key != "B-cf"])
        assert data_orig_worlds.shape[-1] == 1
        assert data_cf_worlds.shape[-1] == 1
        data_orig_worlds = data_orig_worlds[:, :, 0].T
        data_cf_worlds = data_cf_worlds[:, :, 0].T
        _, unique_indices = np.unique(data_orig_worlds, axis=0, return_index=True)
        data_orig_worlds = data_orig_worlds[unique_indices]
        data_cf_worlds = data_cf_worlds[unique_indices]
        concatenated_data = np.concatenate((data_orig_worlds, data_cf_worlds), axis=1)
        concatenated_data = np.hstack(
            (np.tile(interv_vector, (concatenated_data.shape[0], 1)), concatenated_data)
        )
        datasets.append(concatenated_data)
    data = np.vstack(datasets)

    intervention_array = data[:, :interv_vector.shape[0]]
    orig_world_data = data[:, interv_vector.shape[0]:interv_vector.shape[0]+data_orig_worlds.shape[1]]
    cf_world_data = data[:, interv_vector.shape[0]+data_orig_worlds.shape[1]:]
        
    if save_dir is not None:
        print("Saving data as np arrays")
        np.save(save_dir / f"{dataset_name}_intervention.npy", intervention_array)
        np.save(save_dir / f"{dataset_name}_original_world.npy", orig_world_data)
        np.save(save_dir / f"{dataset_name}_counterfactual_world.npy", cf_world_data)


def save_TOY2_multiple(data, dataset_name, variable_names, intervention_vars, save_dir, test=False):
    # save as array: intervention, original world, counterfactual world 
    save_dir.mkdir(exist_ok=True, parents=True)
    datasets = []
    for interv_and_data in data:
        interv_desc, d = interv_and_data
        data_orig_worlds = np.array([d[key] for key in variable_names if "cf" not in key and key != "A" and key != "B"])
        data_cf_worlds = np.array([d[key] for key in variable_names if "cf" in key and key != "A-cf" and key != "B-cf"])
        assert data_orig_worlds.shape[-1] == 1
        assert data_cf_worlds.shape[-1] == 1
        data_orig_worlds = data_orig_worlds[:, :, 0].T
        data_cf_worlds = data_cf_worlds[:, :, 0].T
        interv_vector = np.zeros((data_orig_worlds.shape[0], len(intervention_vars) * 2))
        if interv_desc != ["None"]:
            for desc in interv_desc:
                interv_var = desc.split("(")[1]
                interv_var = interv_var.split(")")[0]
                interv_index = intervention_vars.index(interv_var)
                interv_vector[:, interv_index] = 1
                interv_vector[:, len(intervention_vars)+interv_index] = data_cf_worlds[:, interv_index]
        if test:
            _, unique_indices = np.unique(data_orig_worlds, axis=0, return_index=True)
            interv_vector = interv_vector[unique_indices]
            data_orig_worlds = data_orig_worlds[unique_indices]
            data_cf_worlds = data_cf_worlds[unique_indices]
        concatenated_data = np.concatenate((interv_vector, data_orig_worlds, data_cf_worlds), axis=1)
        datasets.append(concatenated_data)
    data = np.vstack(datasets)

    intervention_array = data[:, :interv_vector.shape[1]]
    orig_world_data = data[:, interv_vector.shape[1]:interv_vector.shape[1]+data_orig_worlds.shape[1]]
    cf_world_data = data[:, interv_vector.shape[1]+data_orig_worlds.shape[1]:]
        
    if save_dir is not None:
        print("Saving data as np arrays")
        np.save(save_dir / f"{dataset_name}_intervention.npy", intervention_array)
        np.save(save_dir / f"{dataset_name}_original_world.npy", orig_world_data)
        np.save(save_dir / f"{dataset_name}_counterfactual_world.npy", cf_world_data)