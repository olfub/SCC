import argparse

# Path to the original YAML file
yaml_file_path = "causal_flows/causal_nf/configs/causal_nf_synthetic_template.yaml"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Modify dataset attributes in the YAML file.")
parser.add_argument("--nodes", type=int, help="Number of nodes", required=True)
parser.add_argument("--edges", type=int, help="Number of edges", required=True)
parser.add_argument("--seed", type=int, help="Dataset seed", required=True)
args = parser.parse_args()

# Generate the output file path dynamically
output_yaml_file_path = f"causal_flows/causal_nf/configs/causal_nf_synthetic_{args.nodes}_{args.edges}_{args.seed}.yaml"

try:
    # Read the original YAML file
    with open(yaml_file_path, 'r') as file:
        lines = file.readlines()
        print("YAML file loaded successfully.")

    # Modify lines 13-15 manually
    lines[12] = f"  nr_nodes: {args.nodes}\n"
    lines[13] = f"  nr_edges: {args.edges}\n"
    lines[14] = f"  data_seed: {args.seed}\n"

    # Write the updated content to a new file
    with open(output_yaml_file_path, 'w') as file:
        file.writelines(lines)
        print(f"Updated YAML file saved as {output_yaml_file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {yaml_file_path}")
except Exception as e:
    print(f"Error: {e}")
