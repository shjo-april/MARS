import os
import cv2
import glob

import numpy as np
from PIL import Image

from tqdm import tqdm

from tools.general import io_utils, cv_utils

parser = io_utils.Parser()

# dataset
parser.add('root_dir', '../VOC2012/', str)
parser.add('domain', 'train_aug', str)

# mars
parser.add('wsss', 'RS+EPM', str)
parser.add('ours', 'RS+EPM@STEGO@K_bg=2@K_fg=2@alpha(0.40)@Mask', str)

parser.add('bias_class', 254, int)

args = parser.get_args()

colors = cv_utils.get_colors_from_length()[..., ::-1]
colors[args.bias_class] = colors[255]

final_tag = args.ours.replace('@Mask', '@MARS')
final_dir = io_utils.create_directory(args.root_dir + f'{args.domain}/{final_tag}/')

for our_path in tqdm(glob.glob(args.root_dir + f'{args.domain}/{args.ours}/*')):
    wsss_path = our_path.replace(args.ours, args.wsss)
    image_path = our_path.replace(args.ours, 'image').replace('.png', '.jpg')

    image_name = os.path.basename(image_path)
    image_id = image_name.replace('.jpg', '')

    if os.path.isfile(final_dir + image_id + '.png'):
        continue
    
    our_mask = Image.open(our_path)
    our_mask = np.asarray(our_mask, dtype=np.uint8).copy()

    wsss_mask = Image.open(wsss_path)
    wsss_mask = np.asarray(wsss_mask, dtype=np.uint8).copy()
    
    final_mask = wsss_mask.copy()

    final_mask[np.logical_and(final_mask > 0, our_mask == 0)] = args.bias_class
    final_mask[np.logical_and(final_mask == 0, our_mask > 0)] = args.bias_class
    
    image = Image.fromarray(final_mask.astype(np.uint8)).convert('P')
    image.putpalette(colors[:256, ::-1])
    image.save(final_dir + image_id + '.png')
