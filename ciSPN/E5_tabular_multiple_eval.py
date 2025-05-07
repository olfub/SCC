import argparse

import torch
import numpy as np
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.nn_wrapper import NNWrapper
from models.spn_create import load_model_params, load_spn

from ciSPN.E1_helpers import get_experiment_name
from ciSPN.evals.classifyStats import ClassifyStats
from ciSPN.models.model_creation import create_nn_model
from datasets.batchProvider import BatchProvider
from datasets.tabularDataset import TabularDataset
from pathlib import Path

print_progress = True


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)  # 606, 1011, 3004, 5555, 12096
parser.add_argument("--model", choices=["mlp", "ciSPN"], default="mlp")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument(
    "--dataset",
    choices=[
        "TOY2",
    ],
    default="TOY2",
)
parser.add_argument("--known-intervention", action="store_true", default=False)
parser.add_argument("--nr-intervs", type=int, default=2)
parser.add_argument(
    "--nr_int_list",
    type=int,
    nargs="+",
    default=[1, 2],
    help="List of numbers of interventions the model was trained with (e.g., 1 2 3).",
)
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.known_intervention = cli_args.known_intervention
conf.model_name = cli_args.model
conf.batch_size = 1000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed
conf.nr_intervs = cli_args.nr_intervs
conf.nr_int_list = cli_args.nr_int_list

conf.explicit_load_part = conf.model_name

seed = conf.seed
make_deterministic(seed)

nr_int_str = "nr_int_" + "_".join([str(i) for i in conf.nr_int_list])

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E5" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E5" / "eval_logs"

experiment_name = get_experiment_name(
    conf.dataset,
    conf.model_name,
    conf.known_intervention,
    seed,
    conf.loss_name,
    conf.loss2_name,
    conf.loss2_factor,
    E=5,
    param=nr_int_str,
)
load_dir = runtime_base_dir / experiment_name

# redirect logs
log_path = log_base_dir / (experiment_name +  "_for_interv_numb_" + str(conf.nr_intervs) + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)

print("Arguments:", cli_args)

X_vars, Y_vars, providers = get_data_description(conf.dataset)
dataset_paths = get_dataset_paths(conf.dataset, "test")
dataset = TabularDataset(
    dataset_paths,
    X_vars,
    Y_vars,
    conf.known_intervention,
    seed,
    part_transformers=providers,
)

# load special evaluation data
test_setting_name = f"{conf.dataset}_eval_multiple"
eval_dir = Path(f"./datasets/{test_setting_name}")
intervention_array = torch.from_numpy(
    np.load(eval_dir / f"{test_setting_name}_intervention.npy")
)
orig_world_data = torch.from_numpy(
    np.load(eval_dir / f"{test_setting_name}_original_world.npy")
)
cf_world_data = torch.from_numpy(
    np.load(eval_dir / f"{test_setting_name}_counterfactual_world.npy")
)
# Get indices where the number of 1s in the first 6 rows of intervention_array equals nr_intervs
if conf.nr_intervs == 1:
    # no or only one intervention (training data)
    indices = torch.nonzero(intervention_array[:, :6].sum(dim=1) <= 1).squeeze()
else:
    indices = torch.nonzero(intervention_array[:, :6].sum(dim=1) == conf.nr_intervs).squeeze()
new_X = torch.cat((intervention_array, orig_world_data), dim=1)[indices]
new_Y = cf_world_data[indices]
dataset.X = new_X.to(dataset.X.device, dtype=torch.float32)
dataset.Y = new_Y.to(dataset.Y.device, dtype=torch.float32)

provider = BatchProvider(dataset, conf.batch_size, provide_incomplete_batch=True)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]

print(f"Loading {conf.explicit_load_part}")
if conf.explicit_load_part == "ciSPN":
    spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir)
    eval_wrapper = spn
elif conf.explicit_load_part == "mlp":
    nn = create_nn_model(num_condition_vars, num_target_vars)
    load_model_params(nn, load_dir=load_dir)
    eval_wrapper = NNWrapper(nn)
else:
    raise ValueError(f"invalid load part {conf.explicit_load_part}")
eval_wrapper.eval()

with torch.no_grad():
    # test performance on test set, unseen data
    stat = ClassifyStats()

    # zero out target vars, to avoid evaluation errors, if marginalization is not working
    demo_target_batch, demo_condition_batch = provider.get_sample_batch()
    if torch.cuda.is_available():
        placeholder_target_batch = torch.zeros_like(demo_target_batch).cuda()
        marginalized = torch.ones_like(demo_target_batch).cuda()
    else:
        placeholder_target_batch = torch.zeros_like(demo_target_batch)
        marginalized = torch.ones_like(demo_target_batch)

    i = 0
    while provider.has_data():
        condition_batch, target_batch = provider.get_next_batch()

        if condition_batch.shape[0] != placeholder_target_batch.shape[0]:
            # can only be true in the last iteration
            nr_elem = condition_batch.shape[0]
            placeholder_target_batch = placeholder_target_batch[:nr_elem]
            marginalized = marginalized[:nr_elem]
        reconstruction = eval_wrapper.predict(
            condition_batch, placeholder_target_batch, marginalized
        )

        # print samples only if only one seed is considered
        print_samples = i == 0
        stat.eval(target_batch, reconstruction, print_samples)

        if print_progress:
            print(f"Processed batch {i}.", end="\r")
        i += 1

eval_str = stat.get_eval_result_str()

print(eval_str + "\n")

logger.close()
