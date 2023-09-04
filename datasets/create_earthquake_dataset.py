from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Cancer Dataset

            'B': Burglary
            'E': Earthquake
            'A': Alarm
            'J': JohnCalls
            'M': MaryCalls 
    
"""

dataset_name = "EARTHQUAKE"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True

if save:
    save_dir.mkdir(exist_ok=True, parents=True)


class SCM_Earthquake(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)
        yes = 0
        no = 1

        burglary = lambda size: self.rng.binomial(1, 0.99, size=(size, 1))
        earthquake = lambda size: self.rng.binomial(1, 0.98, size=(size, 1))
        alarm = lambda size, burglary, earthquake: np.where(
            burglary == yes,
            np.where(
                earthquake == yes,
                self.rng.binomial(1, 0.05, size=(size, 1)),
                self.rng.binomial(1, 0.06, size=(size, 1)),
            ),
            np.where(
                earthquake == yes,
                self.rng.binomial(1, 0.71, size=(size, 1)),
                self.rng.binomial(1, 0.999, size=(size, 1)),
            ),
        )
        johnCalls = lambda size, alarm: np.where(
            alarm == yes,
            self.rng.binomial(1, 0.1, size=(size, 1)),
            self.rng.binomial(1, 0.95, size=(size, 1)),
        )
        maryCalls = lambda size, alarm: np.where(
            alarm == yes,
            self.rng.binomial(1, 0.3, size=(size, 1)),
            self.rng.binomial(1, 0.99, size=(size, 1)),
        )

        self.equations = {
            "B": burglary,
            "E": earthquake,
            "A": alarm,
            "J": johnCalls,
            "M": maryCalls,
        }

    def create_data_sample(self, sample_size, domains=True):
        Bs = self.equations["B"](sample_size)
        Es = self.equations["E"](sample_size)
        As = self.equations["A"](sample_size, Bs, Es)
        Js = self.equations["J"](sample_size, As)
        Ms = self.equations["M"](sample_size, As)

        data = {"B": Bs, "E": Es, "A": As, "J": Js, "M": Ms}
        return data


"""
parameters
"""

variable_names = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]
variable_abrvs = ["B", "E", "A", "J", "M"]
intervention_vars = ["B", "E", "A"]  # J and M are used as target vars
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
    scm = SCM_Earthquake(seed + i)
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
