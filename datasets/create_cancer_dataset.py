from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Cancer Dataset

            'P': Pollution
            'S': Smoker
            'C': Cancer
            'X': X-ray
            'D': Dyspnoea 
    
"""

dataset_name = "CANCER"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True

if save:
    save_dir.mkdir(exist_ok=True, parents=True)


class SCM_Cancer(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)

        low = yes = positive = 0
        high = no = negative = 1

        pollution = lambda size: self.rng.binomial(1, 0.1, size=(size, 1))
        smoker = lambda size: self.rng.binomial(1, 0.7, size=(size, 1))
        cancer = lambda size, pollution, smoker: np.where(
            pollution == low,
            np.where(
                smoker == yes,
                self.rng.binomial(1, 0.97, size=(size, 1)),
                self.rng.binomial(1, 0.999, size=(size, 1)),
            ),
            np.where(
                smoker == yes,
                self.rng.binomial(1, 0.95, size=(size, 1)),
                self.rng.binomial(1, 0.98, size=(size, 1)),
            ),
        )
        xray = lambda size, cancer: np.where(
            cancer == yes,
            self.rng.binomial(1, 0.1, size=(size, 1)),
            self.rng.binomial(1, 0.8, size=(size, 1)),
        )
        dyspnoea = lambda size, cancer: np.where(
            cancer == yes,
            self.rng.binomial(1, 0.35, size=(size, 1)),
            self.rng.binomial(1, 0.7, size=(size, 1)),
        )

        self.equations = {
            "P": pollution,
            "S": smoker,
            "C": cancer,
            "X": xray,
            "D": dyspnoea,
        }

    def create_data_sample(self, sample_size, domains=True):
        Ps = self.equations["P"](sample_size)
        Ss = self.equations["S"](sample_size)
        Cs = self.equations["C"](sample_size, Ps, Ss)
        Xs = self.equations["X"](sample_size, Cs)
        Ds = self.equations["D"](sample_size, Cs)

        data = {"P": Ps, "S": Ss, "C": Cs, "X": Xs, "D": Ds}
        return data


"""
parameters
"""


variable_names = ["Pollution", "Smoker", "Cancer", "X-Ray", "Dyspnoea"]
variable_abrvs = ["P", "S", "C", "X", "D"]
intervention_vars = ["S", "C"]  # P, X and D are used as target vars
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
    scm = SCM_Cancer(seed + i)
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
