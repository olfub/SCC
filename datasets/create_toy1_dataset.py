from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM, create_dataset_train_test

"""
 Deterministic Dataset taken from https://plato.stanford.edu/entries/counterfactuals/ Figure 2 (ident: TOY1)

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

dataset_name = "TOY1"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True


class SCM_TOY1(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)

        # sample original world
        a = lambda size: self.rng.binomial(1, 0.7, size=(size, 1))
        b = lambda size: self.rng.binomial(1, 0.4, size=(size, 1))
        c = lambda size, a, b: np.logical_and(a, b).astype(a.dtype)
        d = lambda size, c: np.logical_not(c).astype(c.dtype)
        e = lambda size, c: c
        f = lambda size, d: np.logical_not(d).astype(d.dtype)
        g = lambda size, e: e
        h = lambda size, f, g: np.logical_or(f, g).astype(f.dtype)

        # counterfactual world before intervention (identical to original world)
        var_cf = lambda size, var: var

        self.equations = {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "E": e,
            "F": f,
            "G": g,
            "H": h,
            "A-cf": var_cf,
            "B-cf": var_cf,
            "C-cf": c,
            "D-cf": d,
            "E-cf": e,
            "F-cf": f,
            "G-cf": g,
            "H-cf": h,
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

        A_cfs = self.equations["A-cf"](sample_size, As)
        B_cfs = self.equations["B-cf"](sample_size, Bs)
        C_cfs = self.equations["C-cf"](sample_size, A_cfs, B_cfs)
        D_cfs = self.equations["D-cf"](sample_size, C_cfs)
        E_cfs = self.equations["E-cf"](sample_size, C_cfs)
        F_cfs = self.equations["F-cf"](sample_size, D_cfs)
        G_cfs = self.equations["G-cf"](sample_size, E_cfs)
        H_cfs = self.equations["H-cf"](sample_size, F_cfs, G_cfs)

        data = {"A": As, "B": Bs, "C": Cs, "D": Ds, "E": Es, "F": Fs, "G": Gs, "H": Hs}
        data.update(
            {
                "A-cf": A_cfs,
                "B-cf": B_cfs,
                "C-cf": C_cfs,
                "D-cf": D_cfs,
                "E-cf": E_cfs,
                "F-cf": F_cfs,
                "G-cf": G_cfs,
                "H-cf": H_cfs,
            }
        )
        return data


"""
parameters
"""

variable_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
variable_names += ["A CF", "B CF", "C CF", "D CF", "E CF", "F CF", "G CF", "H CF"]
variable_abrvs = ["A", "B", "C", "D", "E", "F", "G", "H"]
variable_abrvs += ["A-cf", "B-cf", "C-cf", "D-cf", "E-cf", "F-cf", "G-cf", "H-cf"]
intervention_vars = [
    "C-cf",
    "D-cf",
    "E-cf",
    "F-cf",
    "G-cf",
    "H-cf",
]  # exclude unobserved
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
    scm = SCM_TOY1(seed + i)
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
