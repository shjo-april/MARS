import os
import copy
import tqdm
import torch
import joblib

from tqdm import tqdm
from core import networks, datasets

from tools.ai import torch_utils, augment_utils
from tools.general import io_utils, json_utils, pickle_utils

def produce_predictions(model, test_dataset, indices, device, params):
    scales = params['scales']
    pred_dir = params['pred_dir']
    visualize = params['visualize']

    model = model.to(device)

    if visualize:
        pbar = tqdm(total=len(indices), desc='Inference (the last process)')

    for i in indices:
        image_id, image, _, _ = test_dataset.__getitem__(i)

        if os.path.isfile(pred_dir + f'{image_id}.pkl'):
            continue

        # preprocess
        _, h, w = image.shape
        image = torch.from_numpy(image).to(device)
        
        # inference
        params = {
            "image": image,
            "image_size": (h, w),
            "scales": scales,
            "hflip": True,
            "for_evaluation": True
        }
        output_dict = model.forward_with_scales(**params)
        
        pickle_utils.dump_pickle(
            pred_dir + f'{image_id}.pkl', 
            {
                'keys': output_dict['class_keys'],
                'masks': output_dict['class_masks'],
            }
        )

        if visualize:
            pbar.update(1)

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # set directories
    model_dir = f'./experiments/models/{args.tag}/'

    domain = 'train' if args.domain == 'train_aug' else args.domain
    pred_dir = io_utils.create_directory(f'./experiments/predictions/{args.tag}/{domain}/')

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # create model
    if args.decoder == 'deeplabv3+':
        model = networks.DeepLabv3_Plus(args.backbone, data_dict['num_classes'])

    model.eval()
    torch_utils.load_model(model, model_dir + f'{args.strategy}.pth')

    print('[i] Backbone is {}'.format(args.backbone))
    print('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    print()

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
    scales = [1.0,0.5,1.5,2.0]

    length = len(test_dataset)
    indices = list(range(length))

    num_gpus = len(args.gpus.split(','))
    length_per_gpu = length // num_gpus

    params = []
    
    for gpu_index in range(num_gpus-1):
        param = [
            copy.deepcopy(model), 
            test_dataset,
            indices[:length_per_gpu], 
            torch.device('cuda', gpu_index),
            {
                'scales': scales,
                'pred_dir': pred_dir,
                'visualize': False,
            }
        ]
        params.append(param)
        indices = indices[length_per_gpu:]
        
        print('GPU: {}, Length: {}, Remainder: {}'.format(params[-1][3], len(params[-1][2]), len(indices)))

    param = [
        copy.deepcopy(model), 
        test_dataset,
        indices, 
        torch.device('cuda', num_gpus-1),
        {
            'scales': scales,
            'pred_dir': pred_dir,
            'visualize': True,
        }
    ]
    params.append(param)
    print('GPU: {}, Length: {}, Remainder: 0'.format(params[-1][3], len(params[-1][2])))

    joblib.Parallel(n_jobs=len(params), verbose=10, pre_dispatch="all")(
        [joblib.delayed(produce_predictions)(*param) for param in params]
    )

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # gpu
    parser.add('gpus', '0', str)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train', str)
    
    # networks
    parser.add('backbone', 'resnet101', str)
    parser.add('decoder', 'deeplabv3+', str)
    
    # evaluation
    parser.add('tag', 'ResNet-101@VOC@RS+EPM_with_MARS', str)
    parser.add('strategy', 'last', str)
    
    main(parser.get_args())
