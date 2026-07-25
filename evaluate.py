import tqdm
import numpy as np

from PIL import Image
from core import datasets

from tools.ai import evaluators, augment_utils
from tools.general import io_utils, json_utils

def main(args):
    # set directories
    pred_dir = f'experiments/pseudo-labels/{args.tag}/{args.domain}/'
    
    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

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
    evaluator = evaluators.Evaluator_For_Semantic_Segmentation(data_dict['class_names'], ignore_index=255)
    
    for image_id, _, _, gt_mask in tqdm.tqdm(test_dataset):
        gt_mask = np.asarray(gt_mask)

        pred_mask = Image.open(pred_dir + image_id + '.png')
        pred_mask = np.asarray(pred_mask).copy()

        evaluator.add([pred_mask, gt_mask])
    
    evaluator.print(args.tag)

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train', str)
    
    # evaluation
    parser.add('tag', 'RS+EPM', str)

    main(parser.get_args())
