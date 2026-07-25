import os
import copy
import numpy as np

from PIL import Image

from .abc_modules import Guide_For_Dataset

from tools.ai import torch_utils
from tools.general import io_utils
from tools.general import cv_utils
from tools.general import json_utils

class Dataset_For_Analysis(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform=None, name='VOC'):
        super().__init__(root_dir, domain, ['image_id', 'image', 'tag', 'mask'], f'./data/{name}.json')

        self.name = name
        self.transform = transform
    
    def __getitem__(self, index):
        data = super().__getitem__(index)

        input_dict = {'image': data['image'], 'mask': data['mask']}

        if self.transform is not None:
            output_dict = self.transform(input_dict)
        else:
            output_dict = input_dict

        image_id = data['image_id']
        image, mask = output_dict['image'], output_dict['mask']

        label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)
        
        return image_id, image, label, mask

class Dataset_For_Segmentation(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform, name='VOC'):
        super().__init__(root_dir, domain, ['image', 'mask'], f'./data/{name}.json')

        self.transform = transform
    
    def __getitem__(self, index):
        data = super().__getitem__(index)

        input_dict = {'image': data['image'], 'mask': data['mask']}
        output_dict = self.transform(input_dict)

        return output_dict['image'], output_dict['mask']

class Dataset_For_WSSS(Guide_For_Dataset):
    def __init__(self, root_dir, domain, transform, tag, name='VOC'):
        super().__init__(root_dir, domain, ['image', 'mask', 'tag'], f'./data/{name}.json')

        self.name = name
        self.transform = transform

        self.mask_dir = root_dir + f'{domain}/{tag}/' # for WSSS labels

    def __getitem__(self, index):
        data = super().__getitem__(index)

        input_dict = {'image': data['image'], 'mask': data['mask'], 'weak_image': copy.deepcopy(data['image'])}
        output_dict = self.transform(input_dict)

        if self.name in ['VOC', 'COCO']:
            label = torch_utils.one_hot_embedding([self.class_dict[tag] - 1 for tag in data['tag']], self.num_classes - 1)
            
        return output_dict['image'], output_dict['weak_image'], output_dict['mask'], label, output_dict['crop_bbox']
