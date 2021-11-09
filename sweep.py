"""
Create a weights & biases sweep file based on the .yaml config file specified as an argument to the script.
Then sets the sweep ID as a environment variable so that it can be accessed easily.
"""

# STD
import os
import yaml
import sys
import subprocess

# EXT
import wandb

# PROJECT
from secret import WANDB_API_KEY

# CONST
USER_NAME = "kaleidophon"
PROJECT_NAME = "nlp-uncertainty-zoo"


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(PROJECT_NAME)

    # Get path to sweep .yaml
    config_yaml = sys.argv[1]
    num_runs = sys.argv[2]

    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(config_dict, project=PROJECT_NAME)
    subprocess.Popen(
        [
            "wandb",
            "agent",
            f"{USER_NAME}/{PROJECT_NAME}/{sweep_id}",
            "--count",
            num_runs,
        ]
    )
