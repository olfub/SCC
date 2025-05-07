from pathlib import Path

import numpy as np
from interventionSCM import InterventionSCM
from eval_datasets_helper import save_TOY2_eval

"""
 Inspired by TOY1, but with added randomness on each variable.
 On all observable variables, there is a 10% chance for the "false" (considering TOY1) result which can then also change
 the following variables.

            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
            "H": H

 Any variable with an added '-cf' indicates the value for the counterfactual query.
 
This dataset in particular is based on TOY2 but contains all "inputs" (original world + intervention) exactly once.
"""

dataset_name = "TOY2_eval_unseen"  # used as filename prefix
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True


class SCM_TOY2_EVAL_UNSEEN(InterventionSCM):
    def __init__(self, seed):
        super().__init__(seed)

        # sample original world
        a = lambda size: self.rng.binomial(1, 0.7, size=(size, 1))
        b = lambda size: self.rng.binomial(1, 0.4, size=(size, 1))
        c = lambda size, a, b, negate: np.logical_xor(
            negate, np.logical_and(a, b)
        ).astype(a.dtype)
        d = lambda size, c, negate: np.logical_xor(negate, np.logical_not(c)).astype(
            c.dtype
        )
        e = lambda size, c, negate: np.logical_xor(negate, c).astype(c.dtype)
        f = lambda size, d, negate: np.logical_xor(negate, np.logical_not(d)).astype(
            d.dtype
        )
        g = lambda size, e, negate: np.logical_xor(negate, e).astype(e.dtype)
        h = lambda size, f, g, negate: np.logical_xor(
            negate, np.logical_or(f, g)
        ).astype(f.dtype)

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

    def create_data_sample(self):
        # keep the random factors of C to H constant by determining the random elements first
        # (this is not necessary for A and B, since the var_takes care of that, this is possible as there are no
        # interventions on A or B)
        random_factors = 8  # there are 8 variables
        sample_size = 2 ** random_factors  # therefore, 2**8 possible variable settings

        random_values = np.zeros((sample_size, 8), dtype=int)
        for i in range(sample_size):
            binary_string = format(i, f'0{random_factors}b')
            random_values[i] = [int(bit) for bit in binary_string]

        As = random_values[:,0:1]
        Bs = random_values[:,1:2]
        negate_c = random_values[:,2:3]
        negate_d = random_values[:,3:4]
        negate_e = random_values[:,4:5]
        negate_f = random_values[:,5:6]
        negate_g = random_values[:,6:7]
        negate_h = random_values[:,7:8]
        Cs = self.equations["C"](sample_size, As, Bs, negate_c)
        Ds = self.equations["D"](sample_size, Cs, negate_d)
        Es = self.equations["E"](sample_size, Cs, negate_e)
        Fs = self.equations["F"](sample_size, Ds, negate_f)
        Gs = self.equations["G"](sample_size, Es, negate_g)
        Hs = self.equations["H"](sample_size, Fs, Gs, negate_h)

        A_cfs = self.equations["A-cf"](sample_size, As)
        B_cfs = self.equations["B-cf"](sample_size, Bs)
        C_cfs = self.equations["C-cf"](sample_size, A_cfs, B_cfs, negate_c)
        D_cfs = self.equations["D-cf"](sample_size, C_cfs, negate_d)
        E_cfs = self.equations["E-cf"](sample_size, C_cfs, negate_e)
        F_cfs = self.equations["F-cf"](sample_size, D_cfs, negate_f)
        G_cfs = self.equations["G-cf"](sample_size, E_cfs, negate_g)
        H_cfs = self.equations["H-cf"](sample_size, F_cfs, G_cfs, negate_h)

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
    *[(iv, f"Val_do({iv})=0") for iv in intervention_vars],
    *[(iv, f"Val_do({iv})=1") for iv in intervention_vars],
]

seed = 123
np.random.seed(seed)
datas = []
for i, interv in enumerate(interventions):
    _, interv_desc = interv
    scm = SCM_TOY2_EVAL_UNSEEN(seed + i)
    scm.do(interv_desc)
    data = scm.create_data_sample()
    datas.append((interv_desc, data))
save_TOY2_eval(datas, dataset_name, variable_abrvs, intervention_vars, save_dir)
