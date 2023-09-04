import math
import pickle

import matplotlib.pyplot as plt
import numpy as np


class InterventionSCM:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.equations = {}

        self.intervention = None
        self.intervention_range = None

    def do(self, intervention):
        """
        perform a uniform intervention on a single node
        """
        if intervention is None or "None" in intervention:
            return

        elif "UBin" in intervention:
            low = 0
            high = 1
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda size, *args: self.rng.binomial(
                1, 0.5, size=(size, 1)
            )
            print(
                "Performed Uniform Intervention do({}=U({},{}))".format(
                    intervention, low, high
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "Uniform" in intervention:
            low = 0
            high = 100
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda size, *args: self.rng.uniform(
                low, high, size=(size, 1)
            )
            print(
                "Performed Uniform Intervention do({}=U({},{}))".format(
                    intervention, low, high
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "N" in intervention:
            low = 0
            high = 100
            mu = int(intervention.split("N(")[1].split(",")[0])
            sigma = int(intervention.split("N(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda size, *args: self.rng.normal(
                mu, np.sqrt(sigma), size=(size, 1)
            )
            print(
                "Performed Normal Intervention do({}=N({},{}))".format(
                    intervention, mu, sigma
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "SBeta" in intervention:
            low = 0
            high = 100
            p = float(intervention.split("SBeta(")[1].split(",")[0])
            q = float(intervention.split("SBeta(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = (
                lambda size, *args: self.rng.beta(p, q, size=(size, 1)) * (high - low)
                + low
            )
            print(
                "Performed Non-Standard Beta Intervention do({}=SBeta({},{}))".format(
                    intervention, p, q
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "Gamma" in intervention:
            low = 0
            high = 100
            p = float(intervention.split("Gamma(")[1].split(",")[0])
            q = float(intervention.split("Gamma(")[1].split(",")[1].split(")")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda size, *args: self.rng.gamma(
                p, q, size=(size, 1)
            )
            print(
                "Performed Gamma Intervention do({}=Gamma({},{}))".format(
                    intervention, p, q
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "Val" in intervention:
            # FIXME
            low = 0
            high = 100
            scalar = int(intervention.split("=")[1])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda size, *args: np.repeat(
                scalar, size
            ).reshape(size, 1)
            print(
                "Performed perfect Intervention do({}={})".format(intervention, scalar)
            )
            self.intervention = intervention
            self.intervention_range = (scalar, scalar)
        else:
            raise ValueError(f"Unknown intervention type ({intervention})")


def create_dataset_train_test(
    scm,
    intervention_desc,
    num_samples,
    dataset_name,
    test_split=0.2,
    save_dir=None,
    save_plot_and_info=True,
    variable_names=None,
    variable_abrvs=None,
    exclude_vars=None,
):
    num_samples_train = int(num_samples * (1 - test_split))
    num_samples_test = int(num_samples * test_split)
    create_dataset(
        scm,
        intervention_desc,
        num_samples_train,
        dataset_name,
        "_train",
        save_dir,
        save_plot_and_info,
        variable_names,
        variable_abrvs,
        exclude_vars,
    )
    create_dataset(
        scm,
        intervention_desc,
        num_samples_test,
        dataset_name,
        "_test",
        save_dir,
        save_plot_and_info,
        variable_names,
        variable_abrvs,
        exclude_vars,
    )


def create_dataset(
    scm,
    intervention_desc,
    num_samples,
    dataset_name,
    data_name,
    save_dir=None,
    save_plot_and_info=True,
    variable_names=None,
    variable_abrvs=None,
    exclude_vars=None,
):
    if exclude_vars is None:
        exclude_vars = []
    assert not save_plot_and_info or (
        save_dir is not None
        and variable_names is not None
        and variable_abrvs is not None
    )

    save_dir.mkdir(exist_ok=True, parents=True)

    if save_plot_and_info:
        plot_and_info_dir = save_dir / "info"
        plot_and_info_dir.mkdir(exist_ok=True)

    # create a dataset
    scm.do(intervention_desc)
    data = scm.create_data_sample(num_samples, domains=True)

    if save_plot_and_info:
        with open(
            plot_and_info_dir / f"info_{intervention_desc}{data_name}.txt", "w+"
        ) as fi:
            for e, d in zip(variable_abrvs, variable_names):
                print(
                    "Min {:.2f}\t Max {:.2f}\t Mean {:.2f}\t Median {:.2f}\t STD {:.2f}\t\t - {}".format(
                        np.min(data[e]),
                        np.max(data[e]),
                        np.mean(data[e]),
                        np.median(data[e]),
                        np.std(data[e]),
                        d,
                    ),
                    file=fi,
                )
            n = 25

            print(
                f"(Continuous) First {n} samples from a total of {num_samples} samples:\n"
                "\n".join(
                    [
                        f"\t{var_name} = {data[var_abrv][:n]}"
                        for var_name, var_abrv in zip(variable_names, variable_abrvs)
                    ]
                ),
                file=fi,
            )

            rows = 2
            cols = math.ceil(len(variable_names) / rows)
            fig, axs = plt.subplots(cols, rows, figsize=(12, 10))
            colors = [
                "black",
                "blue",
                "orange",
                "green",
                "purple",
                "red",
                "yellow",
                "lime",
                "cyan",
                "brown",
            ]
            for ind_d, (e, d) in enumerate(zip(variable_abrvs, variable_names)):
                axs.flatten()[ind_d].hist(data[e], bins=2, color=colors[ind_d % 10])
                axs.flatten()[ind_d].set_title(f"{d}")
                axs.flatten()[ind_d].set_xlim(
                    -0.1, 1.1
                )  # data only contains binary data
            plt.suptitle(
                "Intervention: {} {}\nHistograms for {} Samples (x: Value, y: Frequency)".format(
                    intervention_desc, scm.intervention_range, num_samples
                )
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.91)
            plt.savefig(
                plot_and_info_dir
                / (
                    dataset_name
                    + "_{}_N{}{}.png".format(intervention_desc, num_samples, data_name)
                )
            )

    if save_dir is not None:
        save_location = save_dir / (
            dataset_name
            + "_{}_N{}{}.pkl".format(intervention_desc, num_samples, data_name)
        )
        for i in exclude_vars:
            del data[i]
        print("Saving data with keys:", data.keys())
        with open(save_location, "wb") as f:
            pickle.dump(data, f)
            print("Saved Data @ {}".format(save_location))
