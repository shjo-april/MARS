import sys
import torch

from core import refinements

from tools.ai import torch_utils
from tools.general import cv_utils

class Refinement:
    def __init__(self, device, ema, background=255, scales=[1.0], hflip=False, bias_class=254):
        self.ema = ema
        self.ema.to(device)

        self.scales = scales
        self.hflip = hflip

        self.device = device
        self.background = background

        self.inference = refinements.get()
        self.inference.to(device)
        
        self.bias_class = bias_class
        self.colors = cv_utils.get_colors_from_length()[..., ::-1]

    def resize(self, image, image_size):
        OH, OW = image_size

        _, CH, CW = image.size()

        if CW < CH:
            scale = OH / CH
        else:
            scale = OW / CW

        return torch_utils.resize(image, (int(round(CH * scale)), int(round(CW * scale))))

    def normalize(self, x, eps=sys.float_info.epsilon):
        C, H, W = x.size()
        max_x = x.view(C, (H * W)).max(dim=1)[0].view(C, 1, 1)
        return x / (max_x + eps)

    def __call__(self, image, pseudo_mask, label, crop_bbox):
        oh, ow = image.size()[1:]
        xmin, ymin, xmax, ymax = crop_bbox
        
        cropped_image = image[:, ymin:ymax, xmin:xmax]
        
        with torch.no_grad():
            _image = self.resize(cropped_image, (oh, ow))

            params = {
                "image": _image,
                "image_size": _image.size()[1:],

                "scales": self.scales,
                "hflip": self.hflip
            }
            output_dict = self.ema.get_model().forward_with_scales(**params)

            cropped_seg_pred = output_dict['pred_mask'][0]
            cropped_seg_pred = torch_utils.resize(cropped_seg_pred, (ymax-ymin, xmax-xmin), mode='bilinear')
        
        bg = torch.ones((1), dtype=torch.float32).to(self.device)
        label = torch.cat((bg, label), dim=0)

        class_masks = cropped_seg_pred[label == 1]
        class_masks = self.inference(
            cropped_image.unsqueeze(0), 
            class_masks.unsqueeze(0)
        )[0]
        
        class_masks[1:] = self.normalize(class_masks[1:])
        class_masks[0] = 1. - torch.max(class_masks[1:], dim=0)[0]
        
        class_keys = torch.nonzero(label)[:, 0]

        pred_mask = class_keys[torch.argmax(class_masks, dim=0)]
        pred_prob = torch.max(class_masks[1:], dim=0)[0]

        fg_mask = torch.logical_and(
            pseudo_mask[ymin:ymax, xmin:xmax] == self.bias_class,
            pred_mask != 0
        )

        certain_mask = torch.ones((oh, ow)).float().to(self.device)

        if fg_mask.sum() > 0:
            pseudo_mask[ymin:ymax, xmin:xmax][fg_mask] = pred_mask[fg_mask]
            certain_mask[ymin:ymax, xmin:xmax][fg_mask] = pred_prob[fg_mask]
        
        return pseudo_mask, certain_mask

class Complementing_Module:
    def __init__(self, device, ignore_index, refine_params):
        self.device = device
        self.ignore_index = ignore_index
        
        refine_params['device'] = device
        refine_params['background'] = ignore_index

        self.refinement = Refinement(**refine_params)

    def __call__(self, images, masks, labels, crop_bboxes):
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)

        output_dict = {
            'pseudo_masks': [],
            'certain_masks': []
        }
        
        for b in range(images.size(0)):
            image = images[b]
            pseudo_mask = masks[b].clone()
            label = labels[b]
            crop_bbox = crop_bboxes[b]
            
            if label.sum().item() > 0:
                pseudo_mask, certain_mask = self.refinement(image, pseudo_mask, label, crop_bbox)
            else:
                certain_mask = torch.ones(image.size()[1:]).float().to(self.device)
                
            output_dict['pseudo_masks'].append(pseudo_mask)
            output_dict['certain_masks'].append(certain_mask)
        
        for key in output_dict.keys():
            output_dict[key] = torch.stack(output_dict[key])

        return output_dict
