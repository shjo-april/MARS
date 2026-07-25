import os
import cv2
import copy
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image, ImageFont, ImageDraw

from joblib import Parallel, delayed

from tools.ai import torch_utils
from tools.general import io_utils, cv_utils, pickle_utils, xml_utils, json_utils

def list_to_torch(data):
    return torch.from_numpy(np.asarray(data, np.float32))

def generate(image, fmap, debiased_dict_per_image, ib_path):
    h, w, _ = image.shape

    tags = []
    attention_maps = []
    
    for tag in debiased_dict_per_image:
        centroids = debiased_dict_per_image[tag]#.to(device)

        attention_map = F.relu(F.cosine_similarity(fmap, centroids, dim=1))[0]
        # attention_map = torch_utils.resize(attention_map, (h, w))
        
        tags.append(tag)
        attention_maps.append(attention_map)
    
    pickle_utils.dump_pickle(ib_path, {'tags': tags, 'maps':attention_maps, 'wh': (w, h)})

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # environment
    parser.add('gpus', '0', str)
    parser.add('num_workers', 1, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', './VOC2012_Single/', str)
    parser.add('domain', 'train_aug', str)

    parser.add('wsss', 'RS+EPM', str)
    parser.add('usss', 'STEGO', str)
    parser.add('ours', 'RS+EPM_STEGO_K=2_B=skip', str)

    parser.add('ratio', 0.40, float)
    parser.add('reverse', False, bool)

    args = parser.get_args()

    colors = cv_utils.get_colors_from_length()[..., ::-1]

    domain = args.domain
    if args.dataset == 'VOC':
        domain = 'train_aug'

    ib_dir = io_utils.create_directory(args.root_dir + f'{domain}/{args.ours}@global({args.ratio:.02f})/')
    intra_dir = args.root_dir + f'{domain}/{args.ours}/'
    
    ##################################################################################
    # 1. Stacking centroids
    ##################################################################################
    target_dict = {}
    bg_centroids = []

    image_ids = [n.replace('.pkl', '') for n in os.listdir(intra_dir)]

    for image_id in tqdm(image_ids):
        intra_dict = pickle_utils.load_pickle(intra_dir + image_id + '.pkl')
        
        for tag in intra_dict.keys():
            centroids = list(intra_dict[tag])

            if tag == 'background':
                # bg_centroids += [centroids[0]]
                bg_centroids += centroids
            else:
                if not tag in target_dict:
                    target_dict[tag] = {
                        'image_ids': [],
                        'centroids': [],
                    }
                
                target_dict[tag]['image_ids'].append(image_id)
                target_dict[tag]['centroids'].append(centroids)

    print('')
    for tag in target_dict:
        print(tag, len(target_dict[tag]['centroids']))
    print('background', len(bg_centroids))
    print()

    #####################################################################
    # 2. Disentangling class-target and biased centroids
    #####################################################################
    class_target_dict = {}
    bg_centroids = list_to_torch(bg_centroids) # shape: NK x D

    for tag in target_dict.keys():
        class_target_dict[tag] = {
            'distances': [],
            'centroids': [],
        }

        image_ids = np.asarray(target_dict[tag]['image_ids']) # shape: N
        
        centroids = []
        for cs in target_dict[tag]['centroids']:
            centroids += cs
        
        centroids = list_to_torch(centroids) # shape: NK x D
        
        for i, centroid in tqdm(list(enumerate(centroids)), tag):
            fgbg = (1 - F.cosine_similarity(centroid.unsqueeze(0), bg_centroids, dim=1))/2 # NK x K
            fgbg = torch.mean(fgbg, dim=0) # K

            # print(centroid.shape)
            # input()
            
            class_target_dict[tag]['distances'] += [fgbg.item()]
            class_target_dict[tag]['centroids'] += [centroid.numpy()]

    ##################################################################################
    # 3. Debiased Label
    ##################################################################################
    debiased_dict = {}

    for tag in class_target_dict.keys():
        distances = np.asarray(class_target_dict[tag]['distances'])
        distances = np.argsort(distances)[::-1]
        indices = distances[:int(len(distances)*args.ratio)]

        # print(indices.shape, indices.min(), indices.max(), len(distances))

        centroids = np.asarray(class_target_dict[tag]['centroids'])[indices]

        # print(centroids.shape, np.shape(class_target_dict[tag]['centroids']))
        # input()
        
        debiased_dict[tag] = torch.from_numpy(centroids).unsqueeze(-1).unsqueeze(-1)
        debiased_dict[tag] = torch.mean(debiased_dict[tag], dim=0, keepdim=True)

        print(tag, len(class_target_dict[tag]['centroids']), len(centroids), debiased_dict[tag].shape)

    ##################################################################################
    def id2data(root_dir, image_id):
        image = cv2.imread(root_dir + f'{args.domain}/image/' + image_id + '.jpg')

        wsss_mask = Image.open(root_dir + f'{domain}/{args.wsss}/' + image_id + '.png').convert('P')
        wsss_mask = np.asarray(wsss_mask)

        fmap = pickle_utils.load_pickle(root_dir + f'{domain}/pickles_using_{args.usss}/' + image_id + '.pkl')
        fmap = F.normalize(fmap, dim=1)

        _, classes = xml_utils.read_xml(root_dir + f'{args.domain}/xml/' + image_id + '.xml', keep_difficult=True)
        tags = np.unique(classes)

        return image, fmap, wsss_mask, tags

    image_ids = os.listdir(args.root_dir + f'{args.domain}/image/')
    image_ids = [image_id.replace('.jpg', '') for image_id in image_ids]    
    
    gpus = args.gpus.split(',')

    if args.reverse:
        image_ids = image_ids[::-1]

    i = 0
    params = []

    for image_id in tqdm(image_ids, 'Debiased Masks'):
        if os.path.isfile(ib_dir + image_id + '.pkl'):
            continue

        image, fmap, wsss_mask, tags = id2data(args.root_dir, image_id)

        if len(tags) == 0:
            continue
        
        debiased_dict_per_image = {tag: debiased_dict[tag].clone() for tag in tags if tag in debiased_dict}

        param = (image, fmap, debiased_dict_per_image, ib_dir + image_id + '.pkl')
        # generate(*param)
        # continue

        params.append(param)
        
        if len(params) >= (args.num_workers * 8):
            Parallel(n_jobs=args.num_workers, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
            params = []

    if len(params) > 0:
        Parallel(n_jobs=args.num_workers, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
