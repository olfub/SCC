from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Watering Dataset

            "U": person remembering the plant,
            "M": message sent,
            "A": person A waters the plant,
            "B": person B waters the plant,
            "H": health of the plant,

 Any variable with an added '-cf' indicates the value for the counterfactual query.
    
"""

dataset_name = "WATERING"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True


class SCM_WATERING(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)

        # sample original world
        plant_owner = lambda size: self.rng.binomial(1, 0.5, size=(size, 1))
        message = lambda size, plant_owner: plant_owner
        watering_person = lambda size, message: message
        health = lambda size, person_a, person_b: np.logical_or(
            person_a, person_b
        ).astype(person_a.dtype)

        # counterfactual world before intervention (identical to original world)
        u_cf = lambda size, plant_owner: plant_owner

        self.equations = {
            "U": plant_owner,
            "M": message,
            "A": watering_person,
            "B": watering_person,
            "H": health,
            "U-cf": u_cf,
            "M-cf": message,
            "A-cf": watering_person,
            "B-cf": watering_person,
            "H-cf": health,
        }

    def create_data_sample(self, sample_size, domains=True):
        Us = self.equations["U"](sample_size)
        Ms = self.equations["M"](sample_size, Us)
        As = self.equations["A"](sample_size, Ms)
        Bs = self.equations["B"](sample_size, Ms)
        Hs = self.equations["H"](sample_size, As, Bs)

        U_cfs = self.equations["U-cf"](sample_size, Us)
        M_cfs = self.equations["M-cf"](sample_size, U_cfs)
        A_cfs = self.equations["A-cf"](sample_size, M_cfs)
        B_cfs = self.equations["B-cf"](sample_size, M_cfs)
        H_cfs = self.equations["H-cf"](sample_size, A_cfs, B_cfs)

        data = {
            "U": Us,
            "M": Ms,
            "A": As,
            "B": Bs,
            "H": Hs,
            "U-cf": U_cfs,
            "M-cf": M_cfs,
            "A-cf": A_cfs,
            "B-cf": B_cfs,
            "H-cf": H_cfs,
        }
        return data


"""
parameters
"""

variable_names = ["Person U", "Message", "Person A", "Person B", "Health"]
variable_names += [
    "Person U CF",
    "Message CF",
    "Person A CF",
    "Person B CF",
    "Health CF",
]
variable_abrvs = ["U", "M", "A", "B", "H"]
variable_abrvs += ["U-cf", "M-cf", "A-cf", "B-cf", "H-cf"]
intervention_vars = [
    "M-cf",
    "A-cf",
    "B-cf",
    "H-cf",
]  # excluding U, since that is unobserved
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
    scm = SCM_WATERING(seed + i)
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
