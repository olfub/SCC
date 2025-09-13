import argparse
import pickle
import time

from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.spn_create import save_spn
from trainers.dynamicTrainer import DynamicTrainer

from ciSPN.E1_helpers import create_loss, get_experiment_name
from ciSPN.models.model_creation import create_spn_model
from datasets.batchProvider import BatchProvider
from datasets.tabularDataset import TabularDataset
from datasets.largeDataset import load_data

print("start")

parser = argparse.ArgumentParser()
parser.add_argument("--series", type=str, default="E6")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)  # default is 1e-3
parser.add_argument("--epochs", type=int, default=100)  # or 50
parser.add_argument(
    "--dataset",
    choices=[
        "WATERING",
        "TOY1",
        "TOY2",
        "TOY1I",
        "LARGE"
    ],
    default="LARGE",
)
parser.add_argument("--dataset-params", type=str, default="5-5")
parser.add_argument("--known-intervention", action="store_true", default=True)
cli_args = parser.parse_args()

conf = Config()
conf.series = cli_args.series
conf.dataset = cli_args.dataset
conf.dataset_params = cli_args.dataset_params
conf.known_intervention = cli_args.known_intervention
conf.num_epochs = cli_args.epochs
conf.num_epochs_load = (
    conf.num_epochs
)
conf.optimizer_name = "adam"
conf.lr = float(cli_args.lr)
conf.batch_size = 1000
conf.seed = cli_args.seed


make_deterministic(conf.seed)

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / conf.series / "runtimes"
log_base_dir = environment["experiments"]["base"] / conf.series / "logs"

name = conf.dataset if conf.dataset_params == "" else f"{conf.dataset}_{conf.dataset_params}"
experiment_name = get_experiment_name(
    name,
    "ciSPN",
    conf.known_intervention,
    conf.seed,
    "NLLLoss",
    E=6
)
save_dir = runtime_base_dir / experiment_name
save_dir.mkdir(exist_ok=True, parents=True)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)

print("Arguments:", cli_args)

# setup dataset
if conf.dataset == "LARGE":
    nodes, edges = map(int, conf.dataset_params.split("-"))
    dataset, graph = load_data("datasets", "large", nodes, edges, "train", conf.seed)
else:
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
provider = BatchProvider(dataset, conf.batch_size)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]

# build spn graph
rg, params, spn = create_spn_model(num_target_vars, num_condition_vars, conf.seed)
model = spn

model.print_structure_info()


loss, _ = create_loss("NLLLoss", None)

trainer = DynamicTrainer(
    model,
    conf,
    loss,
    train_loss=False,
    pre_epoch_callback=None,
    optimizer=conf.optimizer_name,
    lr=conf.lr,
)

t0 = time.time()
loss_curve = trainer.run_training(provider)
training_time = time.time() - t0
print(f"TIME {training_time:.2f}")


save_spn(save_dir, spn, params, rg, file_name="spn.model")

# save loss curve
with open(save_dir / "loss.pkl", "wb") as f:
    pickle.dump(loss_curve, f)

with open(save_dir / "runtime.txt", "wb") as f:
    pickle.dump(training_time, f)

print(f'Final parameters saved to "{save_dir}"')
logger.close()
