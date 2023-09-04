from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Asia Dataset

            'A': asia,
            'S': smoker,
            'T': tub,
            'L': lung,
            "B": bronc,
            "E": either,
            "X": xray,
            "D": dysp
    
"""

dataset_name = "ASIA"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True


class SCM_ASIA(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)
        yes = 0
        no = 1

        asia = lambda size: self.rng.binomial(1, 0.99, size=(size, 1))
        smoker = lambda size: self.rng.binomial(1, 0.5, size=(size, 1))
        tub = lambda size, asia: np.where(
            asia == yes,
            self.rng.binomial(1, 0.95, size=(size, 1)),
            self.rng.binomial(1, 0.99, size=(size, 1)),
        )
        lung = lambda size, smoke: np.where(
            smoke == yes,
            self.rng.binomial(1, 0.9, size=(size, 1)),
            self.rng.binomial(1, 0.99, size=(size, 1)),
        )
        bronc = lambda size, smoke: np.where(
            smoke == yes,
            self.rng.binomial(1, 0.4, size=(size, 1)),
            self.rng.binomial(1, 0.7, size=(size, 1)),
        )
        either = lambda size, lung, tub: np.minimum(lung, tub)
        xray = lambda size, either: np.where(
            either == yes,
            self.rng.binomial(1, 0.02, size=(size, 1)),
            self.rng.binomial(1, 0.95, size=(size, 1)),
        )
        dysp = lambda size, bronc, either: np.where(
            bronc == yes,
            np.where(
                either == yes,
                self.rng.binomial(1, 0.1, size=(size, 1)),
                self.rng.binomial(1, 0.2, size=(size, 1)),
            ),
            np.where(
                either == yes,
                self.rng.binomial(1, 0.3, size=(size, 1)),
                self.rng.binomial(1, 0.9, size=(size, 1)),
            ),
        )

        self.equations = {
            "A": asia,
            "S": smoker,
            "T": tub,
            "L": lung,
            "B": bronc,
            "E": either,
            "X": xray,
            "D": dysp,
        }

    def create_data_sample(self, sample_size, domains=True):
        As = self.equations["A"](sample_size)
        Ss = self.equations["S"](sample_size)
        Ts = self.equations["T"](sample_size, As)
        Ls = self.equations["L"](sample_size, Ss)
        Bs = self.equations["B"](sample_size, Ss)
        Es = self.equations["E"](sample_size, Ls, Ts)
        Xs = self.equations["X"](sample_size, Es)
        Ds = self.equations["D"](sample_size, Bs, Es)

        data = {"A": As, "S": Ss, "T": Ts, "L": Ls, "B": Bs, "E": Es, "X": Xs, "D": Ds}
        return data


"""
parameters
"""

variable_names = [
    "Asia",
    "Smoker",
    "Tuberculosis",
    "Lung cancer",
    "Bronchitis",
    "Either",
    "X-Ray",
    "Dyspnoea",
]
variable_abrvs = ["A", "S", "T", "L", "B", "E", "X", "D"]
intervention_vars = ["A", "T", "L", "B", "E"]  # S, X and D are used as target vars
exclude_vars = []  # exclude intermediate variables from the final dataset

interventions = [
    (None, "None"),
    *[(iv, f"do({iv})=UBin({iv})") for iv in intervention_vars],
]

seed = 123
np.random.seed(seed)
N = 100000
test_split = 0.2

for i, interv in enumerate(interventions):
    _, interv_desc = interv
    scm = SCM_ASIA(seed + i)
    create_dataset_train_test(
        scm,
        interv_desc,
        N,
        dataset_name,
        test_split=test_split,
        save_dir=save_dir,
        save_plot_and_info=save_plot_and_info,
        variable_names=variable_names,
        variable_abrvs=variable_abrvs,
        exclude_vars=exclude_vars,
    )
