"""
Create a weights & biases sweep file based on the .yaml config file specified as an argument to the script.
Then sets the sweep ID as a environment variable so that it can be accessed easily.
"""

# STD
import yaml
import sys
import subprocess

# EXT
import wandb

# CONST
PROJECT_NAME = "nlp-uncertainty-zoo"


if __name__ == "__main__":

    # Get path to sweep .yaml
    config_yaml = sys.argv[1]

    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(config_dict, project=PROJECT_NAME)
    subprocess.Popen(["wandb", "agent", sweep_id])
