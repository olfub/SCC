from pathlib import Path
from itertools import combinations

import numpy as np
from interventionSCM import InterventionSCM
from eval_datasets_helper import save_TOY2_multiple
import argparse

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
 This dataset in particular is based on TOY2 but support multiple interventions.
 
"""
parser = argparse.ArgumentParser(description="Create TOY2 dataset with multiple interventions.")
parser.add_argument("--seed", type=int, default=123, help="Random seed for dataset generation.")
parser.add_argument(
    "--nr_int_list",
    type=int,
    nargs="+",
    default=[1, 2],
    help="List of numbers of interventions to consider (e.g., 1 2 3).",
)
args = parser.parse_args()
seed=args.seed
nr_int_list = args.nr_int_list


dataset_name = "TOY2_train_multiple"  # used as filename prefix
dataset_name += "_nr_int_" + "_".join([str(i) for i in nr_int_list])
save_dir = Path(f"./datasets/{dataset_name}/")  # base folder
save = True
save_plot_and_info = True

class SCM_TOY2_TRAIN_MULTIPLE(InterventionSCM):
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

    def create_data_sample(self, sample_size):
        # keep the random factors of C to H constant by determining the random elements first
        # (this is not necessary for A and B, since the var_takes care of that, this is possible as there are no
        # interventions on A or B)
        negate_c = self.rng.binomial(1, 0.1, size=(sample_size, 1))
        negate_d = self.rng.binomial(1, 0.1, size=(sample_size, 1))
        negate_e = self.rng.binomial(1, 0.1, size=(sample_size, 1))
        negate_f = self.rng.binomial(1, 0.1, size=(sample_size, 1))
        negate_g = self.rng.binomial(1, 0.1, size=(sample_size, 1))
        negate_h = self.rng.binomial(1, 0.1, size=(sample_size, 1))

        As = self.equations["A"](sample_size)
        Bs = self.equations["B"](sample_size)
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
    *[(iv, f"do({iv})=UBin({iv})") for iv in intervention_vars],
]

np.random.seed(seed)
N = 10000
datas = []

interv_desc = "None"
scm = SCM_TOY2_TRAIN_MULTIPLE(seed)
scm.do(interv_desc)
data = scm.create_data_sample(N)
datas.append(([interv_desc], data))
seed += 1


# for different number of interventions
for nr_ints in nr_int_list:
    # for all combinations of nr_ints intervention_vars
    for interv_comb in combinations(intervention_vars, nr_ints):
        interv_comb_desc = [
            f"do({interv})=UBin({interv})" for interv in interv_comb
        ]
        # get data
        scm = SCM_TOY2_TRAIN_MULTIPLE(seed)
        for interv in interv_comb_desc:
            scm.do(interv)
        data = scm.create_data_sample(N)
        datas.append((interv_comb_desc, data))
        seed += 1

save_TOY2_multiple(datas, dataset_name, variable_abrvs, intervention_vars, save_dir, test=False)