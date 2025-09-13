import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset

from causal_nf.distributions import SCM

import numpy as np

import os

import torch


# %%
class SyntheticDataset(Dataset):
    def __init__(self, root_dir: str, split: str, seed: int = None, num_nodes: int = None, num_edges: int = None, data_seed: int = None):

        self.root_dir = root_dir

        self.seed = seed
        self.split = split

        self.column_names = None

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.data_seed = data_seed

        self.x = None
        self.y = None

        self._add_noise = False

    def set_add_noise(self, value):
        if value == True:
            raise ValueError("Adding noise is not supported.")
        self._add_noise = value

    def _create_data(self):
        """
        This method sets the value for self.X and self.U
        Returns: None

        """

        config_name = f"{self.num_nodes}_{self.num_edges}_{self.data_seed}"

        if self.split == "train":
            data_x = np.load(
            os.path.join(self.root_dir, "large", f"{config_name}_train_data_orig.npy")
        )
        elif self.split in ["valid", "val"]:
            data_x = np.load(
            os.path.join(self.root_dir, "large", f"{config_name}_val_data_orig.npy")
            )
        elif self.split == "test":
            data_x = np.load(
            os.path.join(self.root_dir, "large", f"{config_name}_test_data_orig.npy")
        )
        else:
            raise NotImplementedError(f"Split {self.split} not implemented")
        
        cg_file_path = os.path.join(self.root_dir, "large", f"{config_name}.cg")
        with open(cg_file_path, "r") as f:
            lines = f.readlines()
        variable_names = []
        for line in lines:
            line = line.strip()
            if line == "<NODES>":
                continue
            if line == "":
                break
            variable_names.append(line)

        self.column_names = variable_names

        x = data_x
        x = torch.from_numpy(data_x).float()
        y = x[:, -1]  # TODO can I just do that or what should I choose for y?

        return x, y

    def prepare_data(self) -> None:
        print(f"\nPreparing data...")
        x, y = self._create_data()

        self.x = x
        self.y = y

    def data(self, one_hot=False, scaler=None, x=None):
        raise NotImplementedError("This method is currently unused and should be tested before usage.")

        if x is not None:
            x_tmp = x.clone()
        else:
            x_tmp = self.x.clone()

        x_output = []
        x_sex = x_tmp[:, [0]]
        x_last = None
        if one_hot:
            x_1, x_2 = x_tmp[:, :4], x_tmp[:, 4:]
            assert len(x_2[:, 0].unique()) == 3
            x_s_1 = torch.nn.functional.one_hot(x_2[:, 0].long(), num_classes=3)
            assert len(x_2[:, 1].unique()) == 5
            x_s_2 = torch.nn.functional.one_hot(x_2[:, 1].long(), num_classes=5)
            assert len(x_2[:, 2].unique()) == 4
            x_s_3 = torch.nn.functional.one_hot(x_2[:, 2].long(), num_classes=4)
            x_last = torch.cat((x_s_1, x_s_2, x_s_3), dim=1).float()
        else:
            x_last = x_tmp[:, -3:]

        if scaler:
            x_norm = scaler.transform(x_tmp)
            x_middle = x_norm[:, 1:4]
        else:
            x_middle = x_tmp[:, 1:4]

        x = torch.cat((x_sex, x_middle, x_last), dim=1)
        return x, self.y

    def __getitem__(self, index):

        x = self.x[index].clone()
        return x, self.y[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        my_str = f"Synthetic Dataset\n"
        my_str += f"\tcolumns: {self.column_names}\n"

        return my_str
