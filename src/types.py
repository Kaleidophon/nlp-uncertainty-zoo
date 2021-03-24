"""
Define custom types for this project.
"""

# STD
from typing import List, Union, Dict, Tuple

# EXT
import torch

BatchedSequences = List[torch.LongTensor]
Device = Union[torch.device, str]
HiddenDict = Dict[int, torch.FloatTensor]
HiddenStates = Tuple[torch.FloatTensor, torch.FloatTensor]
