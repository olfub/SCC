from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Deterministic Dataset taken from https://plato.stanford.edu/entries/counterfactuals/ Figure 2 (ident: TOY1I)

            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
            "H": H

 Any variable with an added '-cf' indicates the value for the counterfactual query.
 
"""

dataset_name = "TOY1I"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True


class SCM_TOY1I(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)

        a = lambda size: self.rng.binomial(1, 0.7, size=(size, 1))
        b = lambda size: self.rng.binomial(1, 0.4, size=(size, 1))
        c = lambda size, a, b: np.logical_and(a, b).astype(a.dtype)
        d = lambda size, c: np.logical_not(c).astype(c.dtype)
        e = lambda size, c: c
        f = lambda size, d: np.logical_not(d).astype(d.dtype)
        g = lambda size, e: e
        h = lambda size, f, g: np.logical_or(f, g).astype(f.dtype)

        self.equations = {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "E": e,
            "F": f,
            "G": g,
            "H": h,
        }

    def create_data_sample(self, sample_size, domains=True):
        As = self.equations["A"](sample_size)
        Bs = self.equations["B"](sample_size)
        Cs = self.equations["C"](sample_size, As, Bs)
        Ds = self.equations["D"](sample_size, Cs)
        Es = self.equations["E"](sample_size, Cs)
        Fs = self.equations["F"](sample_size, Ds)
        Gs = self.equations["G"](sample_size, Es)
        Hs = self.equations["H"](sample_size, Fs, Gs)

        data = {"A": As, "B": Bs, "C": Cs, "D": Ds, "E": Es, "F": Fs, "G": Gs, "H": Hs}
        return data


"""
parameters
"""

variable_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
variable_abrvs = ["A", "B", "C", "D", "E", "F", "G", "H"]
intervention_vars = ["C", "D", "E", "F", "G", "H"]  # exclude unobserved
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
    scm = SCM_TOY1I(seed + i)
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
