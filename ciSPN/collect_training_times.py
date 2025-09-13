import pickle
import numpy as np

configs = ["5-5", "10-12", "15-20", "20-30", "50-100", "100-250"]
seeds = range(5)

data = {}

for config in configs:
    for seed in seeds:
        file_path = f"/home/workspace/SCC/experiments/E6/runtimes/E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}/runtime.txt"
        try:
            with open(file_path, "rb") as f:
                data[(config, seed)] = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{c" + "|c" * len(configs) + "}")
print("Seed", end="")
for config in configs:
    print(f" & {config}", end="")
print(" \\\\ \\hline")

print("Time (s)", end="")
for config in configs:
    config_times = [data[(config, seed)] for seed in seeds if (config, seed) in data]
    if config_times:
        mean = np.mean(config_times)
        std = np.std(config_times)
        print(f" & {mean:.2f} \\scriptsize{{\\(\\pm {std:.2f}\\)}}", end="")
    else:
        print(" & N/A", end="")
print(" \\\\")

print("\\end{tabular}")
print("\\caption{Training times for different configurations.}")
print("\\label{tab:training_times}")
print("\\end{table}")