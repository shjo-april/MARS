import os
import cv2
import torch
import joblib
import numpy as np

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from core import refinements, generators
from tools.ai import augment_utils, torch_utils
from tools.general import cv_utils, io_utils, pickle_utils, json_utils

if __name__ == '__main__':
    parser = io_utils.Parser()

    # cpu
    parser.add('cpus', 1, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train_aug', str)

    # debiased masks
    parser.add('tag', 'RS+EPM@STEGO@K_bg=2@K_fg=2@alpha(0.40)', str)

    args = parser.get_args()

    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    ib_dir = args.root_dir + f'{args.domain}/{args.tag}/'
    refine_dir = io_utils.create_directory(args.root_dir + f'{args.domain}/{args.tag}@Mask/')

    image_dir = args.root_dir + f'{args.domain}/image/'

    image_ids = os.listdir(image_dir)
    image_ids = [image_id.replace('.jpg', '') for image_id in image_ids]

    inference = refinements.DenseCRF()
    colors = cv_utils.get_colors_from_length()[:256]
    colors[1] = [255, 255, 255]

    def postprocess(image_id):
        if os.path.isfile(refine_dir + image_id + '.png'):
            return

        image = Image.open(image_dir + image_id + '.jpg').convert('RGB')
        image = np.asarray(image).copy()

        h, w, _ = image.shape 
        
        if os.path.isfile(ib_dir + image_id + '.pkl'):
            wsss_dict = pickle_utils.load_pickle(ib_dir + image_id + '.pkl')

            sum_mask = np.zeros((h, w), dtype=np.int32)
            refined_mask = np.zeros((h, w), dtype=np.int32)
            
            attention_maps = torch.stack(wsss_dict['maps'])
            attention_maps = torch_utils.resize(attention_maps, (h, w))
            attention_maps = attention_maps.numpy()

            for tag, attention_map in zip(wsss_dict['tags'], attention_maps):
                target_mask = inference(image, attention_map[np.newaxis, ...])
                target_mask = np.argmax(target_mask, axis=0)
                
                sum_mask += target_mask
                refined_mask = np.maximum(refined_mask, target_mask*data_dict['class_dict'][tag])

            refined_mask[refined_mask > 0] = 1
            refined_mask = refined_mask.astype(np.uint8)
        else:
            refined_mask = np.zeros((h, w), dtype=np.uint8)
        
        image = Image.fromarray(refined_mask).convert('P')
        image.putpalette(colors)
        image.save(refine_dir + image_id + '.png')

    joblib.Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch="all")(
        [joblib.delayed(postprocess)(image_id) for image_id in image_ids]
    )
