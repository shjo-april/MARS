# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch
from .model import Wide_ResNet

def build_wide_resnet(model_name, last_stride=1, pretrained=True, freeze=True):
    model = Wide_ResNet(last_stride, freeze)
    
    if pretrained:
        state_dict = torch.load(f'./weights/{model_name}.pth')
        model.load_state_dict(state_dict) # , strict=False
    
    return model