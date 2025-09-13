import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory for the experiment logs
base_dir = "/home/workspace/SCC/experiments/E6/eval_logs"

# Configurations for nodes-edges and seeds
configurations = ["5-5", "10-12", "15-20", "20-30", "50-100", "100-250"]
seeds = range(5)  # Seeds 0 to 4
def set_seeds(config):
    if config == "100-250":
        seeds = range(5)
    else:
        seeds = range(5)
    return seeds

# Dictionary to store the content of the files
data = {}

# Iterate over configurations and seeds
for config in configurations:
    seeds = set_seeds(config)
    for seed in seeds:
        # Construct the file path
        file_path = os.path.join(base_dir, f"E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}_times.txt")
        
        # Read the file if it exists
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = file.read()
                data[(config, seed)] = content
        else:
            print(f"File not found: {file_path}")

# Prepare data for plotting
normalized_times = {"BN": [], "SPN": [], "NF": []}
std_devs = {"BN": [], "SPN": [], "NF": []}
x_labels = []

for config in configurations:
    x_labels.append(config)
    bn_times = []
    spn_times = []
    nf_times = []
    seeds = set_seeds(config)
    for seed in seeds:
        if (config, seed) in data:
            content = data[(config, seed)]
            lines = content.splitlines()
            number = int(config.split('-')[0])  # Extract the first part of the configuration
            bn_times.append(float(lines[0].split(":")[1].split()[0]) / number)  # BN inference time
            spn_times.append(float(lines[1].split(":")[1].split()[0]) / number)  # SPN inference time
            nf_times.append(float(lines[2].split(":")[1].split()[0]))  # NF inference time
    # Calculate mean and std for each method
    normalized_times["BN"].append(np.mean(bn_times))
    normalized_times["SPN"].append(np.mean(spn_times))
    normalized_times["NF"].append(np.mean(nf_times))
    std_devs["BN"].append(np.std(bn_times))
    std_devs["SPN"].append(np.std(spn_times))
    std_devs["NF"].append(np.std(nf_times))

# Plot the results with error bars
x = range(len(x_labels))
plt.figure(figsize=(10, 6))
plt.errorbar(x, normalized_times["BN"], yerr=std_devs["BN"], label="BN Inference", marker="o", capsize=5)
plt.errorbar(x, normalized_times["SPN"], yerr=std_devs["SPN"], label="SPN Inference", marker="o", capsize=5)
plt.errorbar(x, normalized_times["NF"], yerr=std_devs["NF"], label="NF Inference", marker="o", capsize=5)
plt.xticks(x, x_labels)
plt.xlabel("Nodes-Edges Configuration")
plt.ylabel("Normalized Time (seconds)")
plt.title("Normalized Inference Time by Configuration (with Std Dev)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Ensure the "plots" directory exists
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Save the plot to the "plots" directory as a PDF
plot_path = os.path.join(plots_dir, "normalized_inference_time_with_std.pdf")
plt.savefig(plot_path, format="pdf")

print(f"Plot saved to: {plot_path}")

# Save the data as a LaTeX tabular
latex_table_path = os.path.join(plots_dir, "normalized_inference_time_with_std.tex")
with open(latex_table_path, "w") as latex_file:
    # Write the header of the LaTeX tabular
    latex_file.write("\\begin{tabular}{c" + "|c" * len(configurations) + "}\n")
    configurations_slash = [config.replace("-", "/") for config in configurations]
    latex_file.write("Method & " + " & ".join(configurations_slash) + " \\\\\n")
    latex_file.write("\\hline\n")
    
    # Write the data for each method
    for method in ["SPN", "BN", "NF"]:
    # for method in ["BN", "SPN"]:
        row = [f"{normalized_times[method][i]:.2f} $\\scriptstyle" + "{\\pm}$" + f"\\scriptsize{{{std_devs[method][i]:.2f}}}" for i in range(len(configurations))]
        method_name = "CBN" if method == "BN" else "cf-SPN" if method == "SPN" else "CNF"
        latex_file.write(f"{method_name} & " + " & ".join(row) + " \\\\\n")
        if method == "BN":
            latex_file.write("\\hline\n")
    
    # Write the footer of the LaTeX tabular
    latex_file.write("\\end{tabular}\n")

print(f"LaTeX table saved to: {latex_table_path}")


all_data = {}

# Iterate over configurations and seeds to load the .npz files
for config in configurations:
    seeds = set_seeds(config)
    for seed in seeds:
        # Construct the file paths for the .npz files
        nf_indis_file_path = os.path.join(base_dir, f"E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}_flow_indis.csv.npz")
        nf_oodis_file_path = os.path.join(base_dir, f"E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}_flow_oodis.csv.npz")
        spn_indis_file_path = os.path.join(base_dir, f"E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}_spn_indis.csv.npz")
        spn_oodis_file_path = os.path.join(base_dir, f"E6_LARGE_{config}_ciSPN_knownI_NLLLoss/{seed}_spn_oodis.csv.npz")
        
        # Load the nf indis .npz file if it exists
        if os.path.exists(nf_indis_file_path):
            all_data[(config, seed, "nf_indis")] = np.load(nf_indis_file_path)
        else:
            print(f"File not found: {nf_indis_file_path}")
        
        # Load the nf oodis .npz file if it exists
        if os.path.exists(nf_oodis_file_path):
            all_data[(config, seed, "nf_oodis")] = np.load(nf_oodis_file_path)
        else:
            print(f"File not found: {nf_oodis_file_path}")
        
        # Load the spn indis .npz file if it exists
        if os.path.exists(spn_indis_file_path):
            all_data[(config, seed, "spn_indis")] = np.load(spn_indis_file_path)
        else:
            print(f"File not found: {spn_indis_file_path}")
        
        # Load the spn oodis .npz file if it exists
        if os.path.exists(spn_oodis_file_path):
            all_data[(config, seed, "spn_oodis")] = np.load(spn_oodis_file_path)
        else:
            print(f"File not found: {spn_oodis_file_path}")

# Prepare data for plotting errors
metrics = ["accuracy", "correct_elementwise", "error_l1", "error_l2"]
plot_titles = {
    "accuracy": "Accuracy by Configuration",
    "correct_elementwise": "Elementwise Correctness by Configuration",
    "error_l1": "L1 Error by Configuration",
    "error_l2": "L2 Error by Configuration",
}
y_labels = {
    "accuracy": "Accuracy",
    "correct_elementwise": "Elementwise Correctness",
    "error_l1": "L1 Error",
    "error_l2": "L2 Error",
}

for dis in ["indis", "oodis"]:
    for metric in metrics:
        spn_values = []
        nf_values = []
        spn_stds = []
        nf_stds = []
        x_labels = []

        for config in configurations:
            seeds = set_seeds(config)
            x_labels.append(config)
            spn_metric_values = []
            nf_metric_values = []

            for seed in seeds:
                # Extract SPN and NF data for the current metric
                if dis == "indis":
                    spn_key = (config, seed, "spn_indis")
                    nf_key = (config, seed, "nf_indis")
                elif dis == "oodis":
                    spn_key = (config, seed, "spn_oodis")
                    nf_key = (config, seed, "nf_oodis")
                else:  # combined
                    raise ValueError("Invalid distribution type. Use 'indis', 'oodis', or 'combined'.")
                value_spn = all_data[spn_key][metric]
                value_nf = all_data[nf_key][metric]
                count_spn = all_data[spn_key]["num_evals"]
                count_nf = all_data[nf_key]["num_evals"]
                if count_spn == 0:
                    value_spn = np.nan
                else:
                    if metric in ["error_l1", "error_l2"]:
                        value_spn /= (count_spn * int(config.split('-')[0]))
                if spn_key in all_data:
                    spn_metric_values.append(value_spn)
                if count_nf == 0:
                    value_nf = np.nan
                else:
                    if metric in ["error_l1", "error_l2"]:
                        value_nf /= (count_nf * int(config.split('-')[0]))
                if nf_key in all_data:
                    nf_metric_values.append(value_nf)

            # Calculate mean and std for SPN and NF
            spn_metric_values = [value for value in spn_metric_values if not np.isnan(value)]
            nf_metric_values = [value for value in nf_metric_values if not np.isnan(value)]
            
            spn_values.append(np.mean(spn_metric_values) if spn_metric_values else np.nan)
            spn_stds.append(np.std(spn_metric_values) if spn_metric_values else np.nan)
            nf_values.append(np.mean(nf_metric_values) if nf_metric_values else np.nan)
            nf_stds.append(np.std(nf_metric_values) if nf_metric_values else np.nan)

        # Plot the results for the current metric
        x = range(len(x_labels))
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, spn_values, yerr=spn_stds, label="SPN", color="blue", marker="o", capsize=5)
        plt.errorbar(x, nf_values, yerr=nf_stds, label="NF", color="orange", marker="o", capsize=5)
        plt.xticks(x, x_labels)
        plt.xlabel("Nodes-Edges Configuration")
        plt.ylabel(y_labels[metric])
        plt.title(plot_titles[metric])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to the "plots" directory as a PDF
        plot_path = os.path.join(plots_dir, f"{metric}_{dis}_by_configuration.pdf")
        plt.savefig(plot_path, format="pdf")
        print(f"Plot saved to: {plot_path}")

# Prepare LaTeX tables for all four metrics: accuracy, elementwise accuracy, L1 error, and L2 error
metrics = ["accuracy", "correct_elementwise", "error_l1", "error_l2"]
metric_labels = {
    "accuracy": "Accuracy",
    "correct_elementwise": "Elementwise Accuracy",
    "error_l1": "L1 Error",
    "error_l2": "L2 Error",
}

for metric in metrics:
    latex_table_path = os.path.join(plots_dir, f"{metric}_by_configuration.tex")
    with open(latex_table_path, "w") as latex_file:
        latex_file.write("\\begin{tabular}{c|c|" + "|".join(["c"] * len(configurations)) + "}\n")
        
        # Write the header
        header = ["", "Method"] + [config.replace("-", "/") for config in configurations]
        latex_file.write(" & ".join(header) + " \\\\\n")
        latex_file.write("\\hline\n")
        
        # Write the data for ID and OOD with multirows
        for dis_index, (dis, dis_label) in enumerate(zip(["indis", "oodis"], ["ID", "OOD"])):
            latex_file.write(f"\\multirow{{2}}{{*}}{{{dis_label}}} ")
            for method in ["SPN", "NF"]:
                row = ["", "cf-SPN" if method == "SPN" else "CNF"]
                for config in configurations:
                    seeds = set_seeds(config)
                    values = []
                    for seed in seeds:
                        key = (config, method.lower() + "_" + dis)
                        value = all_data[(config, seed, method.lower() + "_" + dis)][metric]
                        counts = all_data[(config, seed, method.lower() + "_" + dis)]["num_evals"]
                        if counts == 0:
                            value = np.nan
                        else:
                            if metric in ["error_l1", "error_l2"]:
                                    value /= counts * int(config.split('-')[0])
                                    value *= 100
                        values.append(value)
                    contains_nans = any(np.isnan(values))
                    if all(np.isnan(values)):
                        # only nans
                        row.append("N/A")
                    elif any(np.isnan(values)):
                        # some nans
                        print("Some seeds contain NaNs for", method, config, dis)
                        mean = np.nanmean(values)
                        std = np.nanstd(values)
                        row.append(f"{mean:.2f} $\\scriptstyle{{\\pm}}$ \\scriptsize{{{std:.2f}}}")
                    else:
                        # no nans
                        mean = np.mean(values)
                        std = np.std(values)
                        row.append(f"{mean:.2f} $\\scriptstyle{{\\pm}}$ \\scriptsize{{{std:.2f}}}")
                latex_file.write(" & ".join(row).lstrip("&") + " \\\\\n")
            
            # Add a horizontal line between ID and OOD
            if dis_index == 0:
                latex_file.write("\\hline\n")
        
        latex_file.write("\\end{tabular}\n")

    print(f"LaTeX table for {metric_labels[metric]} saved to: {latex_table_path}")
