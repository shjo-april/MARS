import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from tools.ai import torch_utils

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets, certain_masks=None):
        ce = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        if certain_masks is not None:
            ce *= certain_masks
        
        return ce[targets != self.ignore_index].mean()
