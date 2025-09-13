import argparse
import time

import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import numpy as np
import seaborn as sns
import torch
from causal_nf.config import cfg
from causal_nf.preparators.synthetic_preparator import SyntheticPreparator
from environment import environment
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.spn_create import load_spn
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination

from ciSPN.E1_helpers import get_experiment_name
from ciSPN.evals.marginalStats import MarginalStats
from datasets.batchProvider import BatchProvider
from datasets.largeDataset import load_data, load_model

sns.set_theme()
np.set_printoptions(suppress=True)

print_progress = True


parser = argparse.ArgumentParser()
# parser.add_argument("--model-name", type=str, default="cf-SPN")
parser.add_argument("--series", type=str, default="E6")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--dataset",
    choices=[
        "LARGE"
    ],
    default="LARGE",
)
parser.add_argument("--dataset-params", type=str, default="")
parser.add_argument("--known-intervention", action="store_true", default=False)
parser.add_argument("--statistics", action="store_true", default=False)
cli_args = parser.parse_args()

conf = Config()
conf.series = cli_args.series
conf.dataset = cli_args.dataset
conf.dataset_params = cli_args.dataset_params
conf.known_intervention = cli_args.known_intervention
conf.batch_size = 1000
conf.seed = cli_args.seed

conf.statistics = cli_args.statistics

name = conf.dataset if conf.dataset_params == "" else f"{conf.dataset}_{conf.dataset_params}"

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / conf.series / "runtimes"
log_base_dir = environment["experiments"]["base"] / conf.series / "eval_logs"

seed = int(cli_args.seed)
if conf.dataset == "LARGE":
    nodes, edges = map(int, conf.dataset_params.split("-"))
    dataset, graph = load_data("datasets", "large", nodes, edges, "test", conf.seed, include_exo=True)
    train_data = load_data("datasets", "large", nodes, edges, "train", conf.seed)[0]
    bm = load_model("datasets", "large", nodes, edges, conf.seed)
    X_vars = ["intervention"] + graph[0]
    Y_vars = [name + "_cf" for name in graph[0]]

    # usually, generate random configs
    # but in this case, use test data and provide exogenous variables
    use_test_instead_of_random = nodes > 20
else:
    raise NotImplementedError("Only LARGE dataset is supported in this script.")


# dictionary with intervention -> dictionary (variable -> dictionary (0 or 1 -> counts))
ground_truth = {}

make_deterministic(seed)

experiment_name = get_experiment_name(
    name,
    "ciSPN",
    conf.known_intervention,
    seed,
    "NLLLoss",
    E=6
)
load_dir = runtime_base_dir / experiment_name

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)

print("Arguments:", cli_args)


provider = BatchProvider(dataset, conf.batch_size, provide_incomplete_batch=True)
data_x = dataset.X.cpu().numpy()
data_y = dataset.Y.cpu().numpy()

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]

print(f"Loading SPN")
spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir)
eval_wrapper = spn
eval_wrapper.eval()
train_configs = train_data.X.int()
train_configs = torch.unique(train_configs, dim=0)

# load the normalizing flows model (currently the data loading process is only supported for "LARGE")
if conf.dataset == "LARGE":
    nodes, edges = map(int, conf.dataset_params.split("-"))
    nf_path = f"experiments/E6/NF/NF_synthetic_{nodes}_{edges}_{seed}"
    config = f"causal_flows/causal_nf/configs/causal_nf_synthetic_{nodes}_{edges}_{seed}.yaml"
args_list, args = causal_nf_config.parse_args(["--config_file", "causal_flows/causal_nf/configs/causal_nf_synthetic.yaml", "--wandb_mode", "disabled", "--project", "CAUSAL_NF", "--config_default_file", "causal_flows/causal_nf/configs/default_config.yaml", "--load_model", nf_path])
assert args.load_model is not None, "Please provide a model to load with --load_model"
config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)
causal_nf_config.assert_cfg_and_config(cfg, config)
causal_nf_train.set_reproducibility(cfg)
preparator = SyntheticPreparator.loader(cfg.dataset)
preparator.prepare_data()
loaders = preparator.get_dataloaders(
    batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
)
nf_model = causal_nf_train.load_model(cfg=cfg, preparator=preparator, ckpt_file=args.load_model + "/last.ckpt")

with torch.no_grad():

    if conf.statistics:
        # zero out target vars, to avoid evaluation errors, if marginalization is not working
        demo_target_batch, demo_condition_batch = provider.get_sample_batch()
        if torch.cuda.is_available():
            placeholder_target_batch = torch.zeros_like(demo_target_batch).cuda()
            marginalized = torch.ones_like(demo_target_batch).cuda()
        else:
            placeholder_target_batch = torch.zeros_like(demo_target_batch)
            marginalized = torch.ones_like(demo_target_batch)
        marginalized = marginalized[:1]

        # now, do another evaluation that tracks marginal probabilities
        start_time = time.time()
        bn_time = 0
        spn_time = 0
        nf_time = 0
        with torch.no_grad():
            stat_spn_indis = MarginalStats(len(Y_vars))
            stat_flow_indis = MarginalStats(len(Y_vars))
            stat_spn_oodis = MarginalStats(len(Y_vars))
            stat_flow_oodis = MarginalStats(len(Y_vars))
            demo_target_batch, demo_condition_batch = provider.get_sample_batch()

            # no intervention
            intervention = torch.zeros((1, len(Y_vars) + 1))
            # how many configs to evaluate (orig world + intervention)
            nr_configs = 100
            max_configs = 2 ** len(Y_vars) * len(Y_vars)  # original worlds times variable to intervene on
            # only accept interventions that change something
            if use_test_instead_of_random:
                # get binary configs from test data and fix exogenous variables later
                intervention_indices = torch.full((dataset.X.shape[0],), -1).to(dataset.X.device)
                int_mask = (dataset.X[:, :nodes] > 0).any(dim=1)
                intervention_indices[int_mask] = dataset.X[:, :nodes].argmax(dim=1)[int_mask]
                dataset_configs = torch.cat((
                    dataset.X[:, -nodes:],
                    intervention_indices.unsqueeze(1),
                    dataset.X[:, nodes].unsqueeze(1)
                ), dim=1)

                # make sure no intervention is the same as the original value
                # this also does something for no intervention cases (-1), but those will be ignored later anyway
                selected_column = dataset_configs[:, -2].long()
                selected_values = dataset_configs[torch.arange(dataset_configs.size(0)), selected_column]
                dataset_configs[selected_values == dataset_configs[:, -1], -1] = 1- dataset_configs[selected_values == dataset_configs[:, -1], -1]
                
                current_nr = 0
                binary_configs = torch.full((nr_configs, len(Y_vars) + 2), -1, dtype=torch.int64)
                dataset_configs = dataset_configs.to(binary_configs.device)
                binary_configs_exo = torch.full((nr_configs, nodes), -1, dtype=torch.int64)
                max_count = 0
                while current_nr < nr_configs:
                    rand_idx = torch.randint(0, dataset_configs.shape[0], (1,)).item()
                    current_config = dataset_configs[rand_idx]
                    if current_config[-2] == -1:
                        continue
                    if not any(torch.equal(current_config, bc) for bc in binary_configs):
                        binary_configs[current_nr] = current_config
                        binary_configs_exo[current_nr] = dataset.exo[rand_idx]
                        current_nr += 1
                    max_count += 1
                    if max_count > 10000:
                        # 10000 is a bit arbitrary, but it ensures no undetected infinite loops
                        raise RuntimeError("Could not find enough configs from test data that are not interventions.")
                assert len(binary_configs) == len(torch.unique(binary_configs, dim=0))
            else:
                if max_configs <= nr_configs:
                    binary_configs = torch.tensor(
                        [
                            [*map(int, f"{i:0{len(Y_vars)}b}"), var, val]
                            for i in range(2 ** len(Y_vars))
                            for var in range(len(Y_vars))
                            for val in [0, 1]
                        ]
                    )
                    # make sure no intervention is the same as the original value
                    selected_column = binary_configs[:, -2].long()
                    selected_values = binary_configs[torch.arange(binary_configs.size(0)), selected_column]
                    binary_configs[selected_values == binary_configs[:, -1], -1] = 1- binary_configs[selected_values == binary_configs[:, -1], -1]
                    binary_configs = torch.unique(binary_configs, dim=0)
                else:
                    # generate twice as many
                    binary_configs = torch.randint(0, 2, (nr_configs * 2, len(Y_vars) + 2))
                    # second to last column is interventon index (needs more possible values)
                    binary_configs[:, -2] = torch.randint(0, len(Y_vars), (binary_configs.shape[0],))
                    # last column is intervention value (0 or 1) (can stay as is)

                    # make sure no intervention is the same as the original value
                    selected_column = binary_configs[:, -2].long()
                    selected_values = binary_configs[torch.arange(binary_configs.size(0)), selected_column]
                    binary_configs[selected_values == binary_configs[:, -1], -1] = 1- binary_configs[selected_values == binary_configs[:, -1], -1]

                    binary_configs = torch.unique(binary_configs, dim=0)
                    binary_configs = binary_configs[:nr_configs]  # take only the first nr_worlds
                    while binary_configs.shape[0] < nr_configs:
                        missing_nr = nr_configs - binary_configs.shape[0]
                        additional_configs = torch.randint(0, 2, (missing_nr * 2, len(Y_vars) + 2))
                        additional_configs[:, -2] = torch.randint(0, len(Y_vars), (additional_configs.shape[0],))
                        binary_configs = torch.cat((binary_configs, additional_configs), dim=0)

                        # make sure no intervention is the same as the original value
                        selected_column = binary_configs[:, -2].long()
                        selected_values = binary_configs[torch.arange(binary_configs.size(0)), selected_column]
                        binary_configs[selected_values == binary_configs[:, -1], -1] = 1- binary_configs[selected_values == binary_configs[:, -1], -1]

                        binary_configs = torch.unique(binary_configs, dim=0)
                        binary_configs = binary_configs[:nr_configs]  # take only the first nr_worlds
            invalid_evals = 0
            for i in range(binary_configs.shape[0]):
                orig_world = binary_configs[i:i+1, :-2]
                node = binary_configs[i, -2].item()
                intervention_value = binary_configs[i, -1].item()
                assert node != -1
                assert orig_world[0, node] != intervention_value  # make sure intervention changes something
                intervention = torch.zeros((1, len(Y_vars) + 1))
                intervention[0, node] = 1
                intervention[0, -1] = intervention_value
                # save probs
                bn_probs_positive = np.zeros((1, len(Y_vars)))
                spn_probs_zero = np.zeros((1, len(Y_vars)))
                spn_probs_one = np.zeros((1, len(Y_vars)))
                nf_cf_res = np.zeros((1, len(Y_vars)))
                # bm evidence
                bn_evidence = {var: binary_configs[i, j] for j, var in enumerate(X_vars[1:])}
                if use_test_instead_of_random:
                    # set exogenous variables according to the selected test data point
                    bn_evidence.update({f"{var}_exog": binary_configs_exo[i, j] for j, var in enumerate(X_vars[1:])})
                # spn evidence
                condition_sample = torch.cat(
                    (intervention, orig_world), dim=1
                ).to(dataset.X.device)
                indis = any(torch.equal(condition_sample.int()[0], train_config) for train_config in train_configs)
                # nf input
                nf_orig_world = orig_world.to(dataset.X.device).float()
                nf_int_index = node
                nf_int_value = intervention_value
                # iterate over all variables and predict their marginal probability
                bn_time_temp = 0
                spn_time_temp = 0
                nf_time_temp = 0
                for j in range(len(Y_vars)):
                    # bn prediction
                    # create the mutilated bayesian model
                    bm_do = bm.do([f"{Y_vars[node]}"])
                    bm_do.remove_cpds(bm_do.get_cpds(f"{Y_vars[node]}"))
                    new_cpd = TabularCPD(
                        variable=f"{Y_vars[node]}",
                        variable_card=2,
                        values=[[1.0] if intervention_value == 0 else [0.0], [0.0] if intervention_value == 0 else [1.0]],
                    )
                    bm_do.add_cpds(new_cpd)
                    infer = VariableElimination(bm_do)
                    start_bn = time.time()
                    marginal_prob_bn = infer.query(variables=[f"{Y_vars[j]}"], evidence=bn_evidence, show_progress=False)
                    bn_probs_positive[0, j] = marginal_prob_bn.values[1]
                    bn_time_temp += time.time() - start_bn
                    # spn prediction
                    y_zero = torch.zeros_like(dataset.Y[:1])
                    y_one = torch.ones_like(dataset.Y[:1])
                    marginalized[:] = 1
                    marginalized[:, j] = 0  # marginalize the i-th variable
                    start_spn = time.time()
                    spn_probs_zero[0, j] = torch.exp(eval_wrapper.forward(condition_sample, y_zero, marginalized))
                    spn_probs_one[0, j] = torch.exp(eval_wrapper.forward(condition_sample, y_one, marginalized))
                    spn_time_temp += time.time() - start_spn
                # nf prediction
                start_nf = time.time()
                nf_cf_res[0:1] = nf_model.model.compute_counterfactual(nf_orig_world, nf_int_index, nf_int_value, scaler=nf_model.preparator.scaler_transform).cpu().numpy()
                nf_time_temp += time.time() - start_nf
                
                if np.isnan(bn_probs_positive).any():
                    invalid_evals += 1
                    continue

                bn_time += bn_time_temp
                spn_time += spn_time_temp
                nf_time += nf_time_temp

                spn_probs = spn_probs_one / (spn_probs_zero + spn_probs_one)
                if indis:
                    stat_spn_indis.eval(bn_probs_positive, spn_probs)
                    stat_flow_indis.eval(bn_probs_positive, nf_cf_res)
                else:
                    stat_spn_oodis.eval(bn_probs_positive, spn_probs)
                    stat_flow_oodis.eval(bn_probs_positive, nf_cf_res)
                print(f"Processed {i+1} out of {binary_configs.shape[0]} samples for marginal statistics.", end="\r")

        end_time = time.time()

        if invalid_evals > 0:
            # rescale time to account for invalid evaluations
            bn_time *= binary_configs.shape[0] / (binary_configs.shape[0] - invalid_evals)
            spn_time *= binary_configs.shape[0] / (binary_configs.shape[0] - invalid_evals)
            nf_time *= binary_configs.shape[0] / (binary_configs.shape[0] - invalid_evals)

        print("\nSPN Statistics (in training data):")
        print(stat_spn_indis.get_eval_result_str())
        print("\nFlow Statistics (in training data):")
        print(stat_flow_indis.get_eval_result_str())
        print("\nSPN Statistics (not in training data):")
        print(stat_spn_oodis.get_eval_result_str())
        print("\nFlow Statistics (not in training data):")
        print(stat_flow_oodis.get_eval_result_str())
        print("\n")
        print(f"Time for BN inference: {bn_time:.2f} seconds ({bn_time / len(Y_vars):.2f} seconds per variable)")
        print(f"Time for SPN inference: {spn_time:.2f} seconds ({spn_time / len(Y_vars):.2f} seconds per variable)")
        print(f"Time for NF inference: {nf_time:.2f} seconds ({nf_time:.2f} seconds per variable)")
        print(f"Total time: {end_time - start_time:.2f} seconds ({(end_time - start_time) / len(Y_vars):.2f} seconds per variable)")

        stat_spn_indis.save_stats(
            log_base_dir / (experiment_name + "_spn_indis.csv")
        )
        stat_flow_indis.save_stats(
            log_base_dir / (experiment_name + "_flow_indis.csv")
        )
        stat_spn_oodis.save_stats(
            log_base_dir / (experiment_name + "_spn_oodis.csv")
        )
        stat_flow_oodis.save_stats(
            log_base_dir / (experiment_name + "_flow_oodis.csv")
        )
        # Save all times to a text file
        time_log_path = log_base_dir / (experiment_name + "_times.txt")

        with open(time_log_path, "w") as time_log_file:
            time_log_file.write(f"Time for BN inference: {bn_time:.2f} seconds ({bn_time / len(Y_vars):.2f} seconds per variable)\n")
            time_log_file.write(f"Time for SPN inference: {spn_time:.2f} seconds ({spn_time / len(Y_vars):.2f} seconds per variable)\n")
            time_log_file.write(f"Time for NF inference: {nf_time:.2f} seconds ({nf_time:.2f} seconds per variable)\n")
            time_log_file.write(f"Total time: {end_time - start_time:.2f} seconds ({(end_time - start_time) / len(Y_vars):.2f} seconds per variable)\n")
        print(f"Saved time logs @ {time_log_path}")

        print("\n")
        if use_test_instead_of_random:
            print("This evaluation was using the test data to provide exogenous variables, resulting in not entirely accurate counterfactual probabilities.")
        else:
            print("This evaluation was using random configurations, resulting in accurate counterfactual probabilities.")

        print(f"Invalid evaluations (nan in BN result): {invalid_evals} out of {binary_configs.shape[0]} ({invalid_evals / binary_configs.shape[0] * 100:.2f}%)")

    logger.close()