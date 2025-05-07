import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.nn_wrapper import NNWrapper
from models.spn_create import save_model_params, save_spn
from trainers.dynamicTrainer import DynamicTrainer
from trainers.losses import AdditiveLoss

from ciSPN.E1_helpers import create_loss, get_experiment_name, get_loss_path
from ciSPN.E2_helpers import create_cnn_model
from ciSPN.models.model_creation import create_nn_model, create_spn_model
from datasets.batchProvider import BatchProvider
from datasets.tabularDataset import TabularDataset

print("start")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)  # 606, 1011, 3004, 5555, 12096
parser.add_argument("--model", choices=["mlp", "ciSPN"], default="mlp")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--lr", type=float, default=1e-3)  # default is 1e-3
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument("--epochs", type=int, default=50)  # or 50
parser.add_argument(
    "--loss_load_seed", type=int, default=None
)  # is set to seed if none
parser.add_argument(
    "--dataset",
    choices=[
        "TOY2",
    ],
    default="TOY2",
)
parser.add_argument("--known-intervention", action="store_true", default=False)
parser.add_argument(
    "--nr_int_list",
    type=int,
    nargs="+",
    default=[1, 2],
    help="List of numbers of interventions to consider (e.g., 1 2 3).",
)
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.known_intervention = cli_args.known_intervention
conf.model_name = cli_args.model
conf.num_epochs = cli_args.epochs  # Causal Health: prev was 130 - but 80 is enough ...
conf.num_epochs_load = (
    conf.num_epochs
)  # set to 80 when e.g. using a causal loss trained with 80 epochs
conf.loss_load_seed = (
    cli_args.seed if cli_args.loss_load_seed is None else cli_args.loss_load_seed
)
conf.optimizer_name = "adam"
conf.lr = float(cli_args.lr)
conf.batch_size = 1000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed
conf.nr_int_list = cli_args.nr_int_list

make_deterministic(conf.seed)

nr_int_str = "nr_int_" + "_".join([str(i) for i in conf.nr_int_list])

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E5" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E5" / "logs"

experiment_name = get_experiment_name(
    conf.dataset,
    conf.model_name,
    conf.known_intervention,
    conf.seed,
    conf.loss_name,
    conf.loss2_name,
    conf.loss2_factor,
    E=5,
    param=nr_int_str
)
save_dir = runtime_base_dir / experiment_name
save_dir.mkdir(exist_ok=True, parents=True)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)


# setup dataset
X_vars, Y_vars, providers = get_data_description(conf.dataset)
dataset_paths = get_dataset_paths(conf.dataset, "train")
dataset = TabularDataset(
    dataset_paths,
    X_vars,
    Y_vars,
    conf.known_intervention,
    conf.seed,
    part_transformers=providers,
)
# load train data
train_data_name = f"{conf.dataset}_train_multiple_"
train_data_name += nr_int_str
eval_dir = Path(f"./datasets/{train_data_name}")
intervention_array = torch.from_numpy(
    np.load(eval_dir / f"{train_data_name}_intervention.npy")
)
orig_world_data = torch.from_numpy(
    np.load(eval_dir / f"{train_data_name}_original_world.npy")
)
cf_world_data = torch.from_numpy(
    np.load(eval_dir / f"{train_data_name}_counterfactual_world.npy")
)
new_X = torch.cat((intervention_array, orig_world_data), dim=1)
new_Y = cf_world_data
dataset.X = new_X.to(dataset.X.device, dtype=torch.float32)
dataset.Y = new_Y.to(dataset.Y.device, dtype=torch.float32)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]

provider = BatchProvider(dataset, conf.batch_size)

if conf.model_name == "ciSPN":
    # build spn graph
    rg, params, spn = create_spn_model(num_target_vars, num_condition_vars, conf.seed)
    model = spn
elif conf.model_name == "mlp":
    nn = create_nn_model(num_condition_vars, num_target_vars)
    model = NNWrapper(nn)

model.print_structure_info()


loss, loss_ispn = create_loss(
    conf.loss_name,
    conf,
    num_condition_vars,
    load_dir=runtime_base_dir
    / get_loss_path(conf.dataset, conf.known_intervention, conf.loss_load_seed),
)

if conf.loss2_name is not None:
    (
        loss2,
        loss2_ispn,
    ) = create_loss(
        conf.loss2_name,
        conf,
        num_condition_vars,
        load_dir=runtime_base_dir
        / get_loss_path(conf.dataset, conf.known_intervention, conf.loss_load_seed),
    )
    final_loss = AdditiveLoss(loss, loss2, float(conf.loss2_factor))
else:
    final_loss = loss


trainer = DynamicTrainer(
    model,
    conf,
    final_loss,
    train_loss=False,
    pre_epoch_callback=None,
    optimizer=conf.optimizer_name,
    lr=conf.lr,
)

t0 = time.time()
loss_curve = trainer.run_training(provider)
training_time = time.time() - t0
print(f"TIME {training_time:.2f}")


if conf.model_name == "ciSPN":
    save_spn(save_dir, spn, params, rg, file_name="spn.model")
elif conf.model_name == "mlp":
    save_model_params(save_dir, nn, file_name="nn.model")

# save loss curve
with open(save_dir / "loss.pkl", "wb") as f:
    pickle.dump(loss_curve, f)

with open(save_dir / "runtime.txt", "wb") as f:
    pickle.dump(training_time, f)

print(f'Final parameters saved to "{save_dir}"')
logger.close()
