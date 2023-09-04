from pathlib import Path

import numpy as np

np.set_printoptions(
    edgeitems=30,
    linewidth=100000,
    suppress=True,
    formatter={"float_kind": "{:.2f}".format},
)
import os
import pickle

import matplotlib.pyplot as plt

"""
Structural Causal Model

    A = N_A, N_A is Uniform distributed, A in N
    F = 1_X(A) OR N_F, N_F is Bernoulli distributed, F in {0,1}
    H = alpha * F + beta * A + gamma * N_H, N_H is Bernoulli distributed and alpha + beta + gamma = 1, H in (0, 1]
    M = delta * H + (1-delta) * N_M, N_M is Bernoulli distributed, M in (0, 1]
    
    Diagnose = (1/(1 + (A - 5)^2) * N(5, 5)) +
               (1/(1 + (A - 20)^2) * N(30, 10)) * 0.5 * H +
               (1/(1 + (A - 40)^2) * N(40, 20)) * 2.2 * F +
               (1/(1 + (A - 60)^2) * N(60, 10)) * M * H

    Age -> Food Habit
    Age -> Health
    Food Habit -> Health
    Health -> Mobility
    
    Age, Food Habit, Health, Mobility -> Diagnose
    
"""


class SCM_HealthClassification:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed=seed)

        age = lambda size: self.rng.uniform(low=0, high=100, size=size)[0]
        food_habit = lambda age: 0.5 * age + self.rng.normal(loc=10, scale=np.sqrt(10))
        health = (
            lambda age, food_habit: 0.008 * (100 - age**2)
            + 0.5 * food_habit
            + self.rng.normal(loc=40, scale=np.sqrt(30))
        )
        mobility = lambda health: 0.5 * health + self.rng.normal(
            loc=20, scale=np.sqrt(10)
        )

        d1 = lambda age, food_habit, health, mobility: (
            np.where(
                age <= 45.667,
                (0.00108 * age**3 - 0.08862 * age**2 + 1.337 * age + 25),
                4.09837,
            )
        ) + self.rng.normal(loc=5, scale=np.sqrt(10))
        d2 = (
            lambda age, food_habit, health, mobility: 0.525 * mobility
            + 0.0175 * food_habit
            + self.rng.normal(loc=0, scale=np.sqrt(5))
        )
        d3 = (
            lambda age, food_habit, health, mobility: 0.00013857 * age**3
            - 0.0135 * age**2
            + 0.2025 * age
            + 0.2025 * health
            + 17.1714
            + self.rng.normal(loc=0, scale=np.sqrt(age * 0.2))
        )
        diagnose = lambda d1, d2, d3: np.argmax([d1, d2, d3]) + 1

        self.equations = {
            "A": age,
            "F": food_habit,
            "H": health,
            "M": mobility,
            "D1": d1,
            "D2": d2,
            "D3": d3,
            "D": diagnose,
        }
        self.intervention = None
        self.intervention_range = None

    def create_data_sample(self, sample_size):
        As = np.array([self.equations["A"](1) for _ in range(sample_size)])
        Fs = np.array([self.equations["F"](a) for a in As])
        Hs = np.array([self.equations["H"](a, Fs[ind]) for ind, a in enumerate(As)])
        Ms = np.array([self.equations["M"](h) for h in Hs])

        # compute sub variables
        D1 = np.array(
            [self.equations["D1"](a, Fs[i], Hs[i], Ms[i]) for i, a in enumerate(As)]
        )
        D2 = np.array(
            [self.equations["D2"](a, Fs[i], Hs[i], Ms[i]) for i, a in enumerate(As)]
        )
        D3 = np.array(
            [self.equations["D3"](a, Fs[i], Hs[i], Ms[i]) for i, a in enumerate(As)]
        )
        # compute argmax(+1)
        Ds = np.array(
            [self.equations["D"](D1[i], D2[i], D3[i]) for i, _ in enumerate(D1)]
        )
        # set only argmax to true
        D1 = (Ds == 1).astype(int)
        D2 = (Ds == 2).astype(int)
        D3 = (Ds == 3).astype(int)

        data = {
            "A": As,
            "F": Fs,
            "H": Hs,
            "M": Ms,
            "D": Ds,
            "D1": D1,
            "D2": D2,
            "D3": D3,
        }

        return data

    def do(self, intervention):
        """
        perform a uniform intervention on a single node
        """
        if intervention is None or "None" in intervention:
            return

        if "U" in intervention:
            low = 0
            high = 100
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.uniform(low, high)
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
            self.equations[intervention] = lambda *args: self.rng.normal(
                mu, np.sqrt(sigma)
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
                lambda *args: self.rng.beta(p, q) * (high - low) + low
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
            self.equations[intervention] = lambda *args: self.rng.gamma(p, q)
            print(
                "Performed Gamma Intervention do({}=Gamma({},{}))".format(
                    intervention, p, q
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif "[" in intervention:
            low = 0
            high = 100
            a = int(intervention.split("[")[1].split(",")[0])
            b = int(intervention.split("[")[1].split(",")[1].split("]")[0])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: self.rng.choice([a, b])
            print(
                "Performed Choice Intervention do({}=[{},{}])".format(
                    intervention, a, b
                )
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        elif intervention is not None:
            low = 0
            high = 100
            scalar = int(intervention.split("=")[1])
            intervention = intervention.split("do(")[1].split(")")[0]
            self.equations[intervention] = lambda *args: scalar
            print(
                "Performed perfect Intervention do({}={})".format(intervention, scalar)
            )
            self.intervention = intervention
            self.intervention_range = (low, high)
        else:
            raise ValueError(f"Unknown intervention type ({intervention})")


interventions = [
    (None, "None"),
    ("H", "do(H)=U(H)"),
    ("M", "do(M)=U(M)"),
    ("A", "do(A)=U(A)"),
    ("F", "do(F)=U(F)"),
]

seed = 123
np.random.seed(seed)
num_samples = 100000
test_split = 0.2

dir_save = Path(f"./datasets/causalHealthClassification")
dir_save.mkdir(exist_ok=True)
save = True
save_plot_and_info = True
plot_and_info_dir = dir_save / "info"
if save_plot_and_info:
    plot_and_info_dir.mkdir(exist_ok=True)


num_samples_train = int(num_samples * (1 - test_split))
num_samples_test = int(num_samples * test_split)

for j, (N, data_name) in enumerate(
    [(num_samples_train, "train"), (num_samples_test, "test")]
):
    print(f"[{data_name}]")
    for i, interv in enumerate(interventions):
        interv, interv_desc = interv

        # create a dataset
        scm = SCM_HealthClassification(seed + 100 * j + i)
        scm.do(interv_desc)
        data = scm.create_data_sample(N)

        if save_plot_and_info:
            with open(
                plot_and_info_dir / f"info_{interv_desc}_{data_name}.txt", "w+"
            ) as fi:
                for ind_d, d in enumerate(
                    ["Age", "Food Habits", "Health", "Mobility", "Diagnose"]
                ):
                    e = d[0]
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
                    "(Continuous) First {} samples from a total of {} samples:\n"
                    "\tAge         = {}\n"
                    "\tFood Habits = {}\n"
                    "\tHealth      = {}\n"
                    "\tMobility    = {}\n"
                    "\tDiagnose    = {}\n"
                    "\tDiagnose1   = {}\n"
                    "\tDiagnose2   = {}\n"
                    "\tDiagnose3   = {}\n"
                    "\n\n***********************************\n\n".format(
                        n,
                        N,
                        data["A"][:n],
                        data["F"][:n],
                        data["H"][:n],
                        data["M"][:n],
                        data["D"][:n],
                        data["D1"][:n],
                        data["D2"][:n],
                        data["D3"][:n],
                    ),
                    file=fi,
                )

                # plot the median health per age group
                plt.figure(figsize=(12, 7))
                for v in [
                    "Food Habits",
                    "Health",
                    "Mobility",
                    "Diagnose",
                    "1 Diagnose",
                    "2 Diagnose",
                    "3 Diagnose",
                ]:
                    median_var_per_age = []
                    mean_var_per_age = []
                    std_var_per_age = []
                    # age_intervals = [(0, 10), (10, 30), (30, 55), (55, 75), (75, 100)]
                    age_intervals = [(n * 5, ((n + 1) * 5)) for n in range(19)]
                    dd = v[0] if v[0] not in ["1", "2", "3"] else "D" + v[0]
                    for a in age_intervals:
                        indices = np.where(
                            np.logical_and(data["A"] > a[0], data["A"] < a[1])
                        )[0]
                        corresponding_var_data = [data[dd][i] for i in indices]
                        median_var = np.median(corresponding_var_data)
                        mean_var = np.mean(corresponding_var_data)
                        std_var = np.std(corresponding_var_data)
                        median_var_per_age.append(median_var)
                        mean_var_per_age.append(mean_var)
                        std_var_per_age.append(std_var)

                    factor = 10 if v[0] == "D" else 1

                    e = dd
                    p = plt.plot(
                        range(len(age_intervals)),
                        np.array(mean_var_per_age) * factor,
                        label="{} |All Data {:.1f}*scaled{:.1f}, {:.1f}, {:.1f}|".format(
                            v,
                            np.mean(data[e]),
                            factor,
                            np.min(data[e]),
                            np.max(data[e]),
                        ),
                    )
                    plt.errorbar(
                        range(len(age_intervals)),
                        np.array(mean_var_per_age) * factor,
                        yerr=std_var_per_age,
                        color=p[0].get_color(),
                    )
                    plt.title(
                        "Intervention: {} {}\nContinuous Data Mean Values per Age intervals x<a<y (Sampled {} Persons via SCM)\nVariable Name |All Data Mean, Min, Max|".format(
                            interv_desc, scm.intervention_range, N
                        )
                    )
                    plt.xlabel("Age $A$")
                    plt.ylabel("Mean for Variable in Interval")
                    plt.xticks(
                        range(len(age_intervals)), [str(x) for x in age_intervals]
                    )
                plt.ylim(-10, 70)
                plt.legend(bbox_to_anchor=[0.5, -0.11], loc="center", ncol=3)
                plt.tight_layout()
                axes = plt.gca()
                # plt.show()
                plt.savefig(plot_and_info_dir / f"age_avg_{interv_desc}.jpg", dpi=300)

                fig, axs = plt.subplots(3, 2, figsize=(12, 10))
                colors = ["black", "blue", "orange", "green", "purple"]
                for ind_d, d in enumerate(
                    ["Age", "Food Habits", "Health", "Mobility", "Diagnose"]
                ):
                    axs.flatten()[ind_d].set_title("{}".format(d))
                    if d[0] == "D":
                        # limit range of diagnoses
                        axs.flatten()[ind_d].hist(
                            data[d[0]], bins=5, color=colors[ind_d]
                        )
                        axs.flatten()[ind_d].set_xlim(-0.1, 2.1)
                    else:
                        axs.flatten()[ind_d].hist(
                            data[d[0]], bins=50, color=colors[ind_d]
                        )
                        # axs.flatten()[ind_d].set_title('{}'.format(d))
                        axs.flatten()[ind_d].set_xlim(-20, 100)
                plt.suptitle(
                    "Intervention: {} {}\nHistograms for {} Samples (x: Value, y: Frequency)".format(
                        interv_desc, scm.intervention_range, N
                    )
                )
                plt.savefig(
                    plot_and_info_dir / f"stats_{data_name}_{interv_desc}.jpg", dpi=300
                )

        if save:
            save_location = os.path.join(
                dir_save,
                f"causalHealthClassification_{interv_desc}_N{N}_{data_name}.pkl",
            )
            # excludeData = ['1', '2', '3'] # exclude intermediate diagnoses class data
            excludeData = ["D"]  # exclude diagnoses class data
            for i in excludeData:
                del data[i]
            print("Saving data with keys:", data.keys())
            with open(save_location, "wb") as f:
                pickle.dump(data, f)
                print("Saved Data @ {}".format(save_location))
