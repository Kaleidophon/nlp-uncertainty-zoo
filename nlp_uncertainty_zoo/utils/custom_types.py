"""
Define custom types for this project.
"""

# STD
from typing import List, Union, Dict, Tuple

# EXT
import torch
import wandb

# TYPES
Device = Union[torch.device, str]
HiddenDict = Dict[int, torch.FloatTensor]
HiddenStates = Tuple[torch.FloatTensor, torch.FloatTensor]
WandBRun = wandb.wandb_sdk.wandb_run.Run
