import torch
import numpy as np

from torch import nn
from .abc_modules import Guide_For_Model

from .backbones import resnet
from .layers import ASPP_For_DeepLabv3, Decoder

from tools.ai import torch_utils

class Backbone(nn.Module, Guide_For_Model):
    def __init__(
            self, 
            backbone,
            norm_fn='bn', 
            activation_fn='relu',
            pretrained=True,
            last_stride=2,
            output_stride=16
        ):
        super().__init__()

        if norm_fn == 'bn':
            self.norm_fn = nn.BatchNorm2d
        
        if activation_fn == 'relu':
            self.activation_fn = lambda:nn.ReLU(inplace=True)
        
        if 'resnet' in backbone:
            self.model = resnet.build_resnet(
                backbone, self.norm_fn, self.activation_fn, 
                last_stride, pretrained, output_stride
            )

            self.in_channels = [64, 256, 512, 1024, 2048]

class DeepLabv3_Plus(Backbone):
    def __init__(self, backbone, num_classes=21, output_stride=8, feature_size=256, low_channels=48):
        super().__init__(backbone, norm_fn='bn', pretrained=True, last_stride=1, output_stride=output_stride)

        self.aspp = ASPP_For_DeepLabv3(
            self.in_channels[-1], 
            feature_size, 
            output_stride
        )
        self.decoder = Decoder(
            self.in_channels[1], 
            feature_size, 
            num_classes,
            low_channels=low_channels
        )

        self.num_classes = num_classes

    def forward(self, inputs, same=True, interpolation='bilinear'):
        output_dict = {}

        _, C2, _, _, C5 = self.model(inputs)

        x = self.aspp(C5)
        x = self.decoder(x, C2)

        output_dict['seg_logits'] = x
        if same:
            output_dict['seg_logits'] = torch_utils.resize(x, inputs.size()[2:], mode=interpolation)
        
        return output_dict

    def forward_with_scales(
            self, 
            image, image_size,
            scales=[1.0], hflip=False,
            interpolation='bilinear',
            for_evaluation=False,
        ):
        seg_h = 0
        seg_w = 0
        
        pred_masks = []

        images = image.unsqueeze(0)

        for scale in scales:
            # rescale
            scaled_images = torch_utils.resize(images, image_size, scale=scale)
            if hflip: scaled_images = torch.cat([scaled_images, scaled_images.flip(-1)], dim=0)
            
            # inference
            with torch.no_grad():
                output_dict = self.forward(scaled_images, interpolation=interpolation)
            
            if seg_h == 0:
                _, _, seg_h, seg_w = output_dict['seg_logits'].size()
            
            pred_mask = torch.softmax(output_dict['seg_logits'], dim=1)
            pred_mask = torch_utils.resize(pred_mask, (seg_h, seg_w), mode=interpolation)
            
            pred_masks.append(pred_mask[0])
            if hflip: pred_masks.append(pred_mask[1].flip(-1))

        output_dict = {}
        
        pred_mask = torch.mean(torch.stack(pred_masks), dim=0, keepdim=True)
        output_dict['pred_mask'] = pred_mask

        if for_evaluation:
            pred_mask = torch_utils.get_numpy(output_dict['pred_mask'])
            class_mask = np.max(pred_mask.reshape(self.num_classes, seg_h*seg_w), axis=1) >= 0.05
            
            output_dict['class_keys'] = np.nonzero(class_mask)[0]
            output_dict['class_masks'] = pred_mask[0, class_mask, :, :]

        return output_dict
