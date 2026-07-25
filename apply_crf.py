import os
import torch
import joblib

import numpy as np

from PIL import Image
from core import datasets, refinements

from tools.ai import torch_utils, augment_utils
from tools.general import io_utils, json_utils, cv_utils, pickle_utils

def main(args):
    # set directories
    domain = 'train' if args.domain == 'train_aug' else args.domain

    pred_dir = io_utils.create_directory(f'./experiments/predictions/{args.tag}/{domain}/')
    pseudo_dir = io_utils.create_directory(f'./experiments/pseudo-labels/{args.tag}/{domain}/')

    # read dataset information
    colors = cv_utils.get_colors_from_length()[:256, ::-1]

    # for datasets
    test_transform = augment_utils.Compose(
        [
            augment_utils.Normalize(),
            augment_utils.Transpose(),
        ]
    )

    test_dataset = datasets.Dataset_For_Analysis(
        args.root_dir, 
        args.domain, 
        test_transform,
        name=args.dataset
    )

    # for evaluation
    denorm_fn = augment_utils.Denormalize()
    inference_for_crf = refinements.DenseCRF(for_seg=True)

    def postprocess(i):
        image_id, image, _, _ = test_dataset.__getitem__(i)

        pseudo_path = pseudo_dir + image_id + '.png'
        if os.path.isfile(pseudo_path):
            return
        
        # preprocess
        _, h, w = image.shape

        infer_dict = pickle_utils.load_pickle(pred_dir + f'{image_id}.pkl')
        
        class_keys = infer_dict['keys']

        if len(class_keys) > 1:
            image = denorm_fn(image).copy()

            ih, iw = image.shape[:2]
            sh, sw = infer_dict['masks'].shape[:2]
            
            if ih != sh or iw != sw:
                masks = torch.from_numpy(infer_dict['masks'])
                masks = torch_utils.resize(masks, (ih, iw))
                infer_dict['masks'] = masks.numpy()
            
            pseudo_label = inference_for_crf(image, infer_dict['masks'])
            
            pseudo_label = np.argmax(pseudo_label, axis=0)
            pseudo_label = class_keys[pseudo_label]
        else:
            pseudo_label = np.zeros((h, w), dtype=np.uint8)
        
        image = Image.fromarray(pseudo_label.astype(np.uint8)).convert('P')
        image.putpalette(colors[..., ::-1])
        image.save(pseudo_dir + image_id + '.png')
    
    joblib.Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch="all")(
        [joblib.delayed(postprocess)(i) for i in list(range(len(test_dataset)))]
    )
    
if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # cpu
    parser.add('cpus', 16, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train', str)
    
    # evaluation
    parser.add('tag', 'ResNet-101@VOC@RS+EPM_with_MARS', str)
    
    main(parser.get_args())
