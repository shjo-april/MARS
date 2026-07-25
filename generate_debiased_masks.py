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

def generate(image, fmap, debiased_centroids, ib_path):
    h, w, _ = image.shape

    tags = []
    attention_maps = []
    
    for tag in debiased_centroids:
        debiased_centroid = debiased_centroids[tag]
        attention_map = F.relu(F.cosine_similarity(fmap, debiased_centroid, dim=1))[0]
        
        tags.append(tag)
        attention_maps.append(attention_map)

    pickle_utils.dump_pickle(ib_path, {'tags': tags, 'maps':attention_maps, 'wh': (w, h)})

if __name__ == '__main__':
    parser = io_utils.Parser()
    
    # cpu
    parser.add('cpus', 1, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train_aug', str)

    # wsss and uss methods
    parser.add('wsss', 'RS+EPM', str)
    parser.add('usss', 'STEGO', str)

    # mars
    parser.add('ours', 'RS+EPM@STEGO@K_bg=2@K_fg=2', str)

    # selecting
    parser.add('alpha', 0.40, float)

    args = parser.get_args()

    colors = cv_utils.get_colors_from_length()[..., ::-1]

    ib_dir = io_utils.create_directory(args.root_dir + f'{args.domain}/{args.ours}@alpha({args.alpha:.02f})/')
    intra_dir = args.root_dir + f'{args.domain}/{args.ours}/'
    
    # 1. Stack centroids for all foreground and background classes
    target_dict = {}
    bg_centroids = []

    image_ids = [n.replace('.pkl', '') for n in os.listdir(intra_dir)]

    for image_id in tqdm(image_ids):
        intra_dict = pickle_utils.load_pickle(intra_dir + image_id + '.pkl')
        
        for tag in intra_dict.keys():
            centroids = list(intra_dict[tag])

            if tag == 'background':
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

    # 2. Calculate the distance per centroid
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

            class_target_dict[tag]['distances'] += [fgbg.item()]
            class_target_dict[tag]['centroids'] += [centroid.numpy()]

    # 3. Select debiased centroids
    debiased_dict = {}

    for tag in class_target_dict.keys():
        distances = np.asarray(class_target_dict[tag]['distances'])
        distances = np.argsort(distances)[::-1]

        indices = distances[:int(len(distances)*args.alpha)]
        centroids = np.asarray(class_target_dict[tag]['centroids'])[indices]

        debiased_dict[tag] = torch.from_numpy(centroids).unsqueeze(-1).unsqueeze(-1)
        debiased_dict[tag] = torch.mean(debiased_dict[tag], dim=0, keepdim=True)

        print(tag, len(class_target_dict[tag]['centroids']), len(centroids), debiased_dict[tag].shape)
    
    # Generate debiased 
    def id2data(root_dir, image_id):
        image = cv2.imread(root_dir + f'{args.domain}/image/' + image_id + '.jpg')

        wsss_mask = Image.open(root_dir + f'{args.domain}/{args.wsss}/' + image_id + '.png').convert('P')
        wsss_mask = np.asarray(wsss_mask)

        fmap = pickle_utils.load_pickle(root_dir + f'{args.domain}/pickles_using_{args.usss}/' + image_id + '.pkl')
        fmap = F.normalize(fmap, dim=1)

        _, classes = xml_utils.read_xml(root_dir + f'{args.domain}/xml/' + image_id + '.xml', keep_difficult=True)
        tags = np.unique(classes)

        return image, fmap, wsss_mask, tags

    image_ids = os.listdir(args.root_dir + f'{args.domain}/image/')
    image_ids = [image_id.replace('.jpg', '') for image_id in image_ids]    

    i = 0
    params = []
    
    for image_id in tqdm(image_ids, 'Debiased Masks'):
        if os.path.isfile(ib_dir + image_id + '.pkl'):
            continue

        image, fmap, wsss_mask, tags = id2data(args.root_dir, image_id)

        if len(tags) == 0:
            continue
        
        debiased_centroids = {tag: debiased_dict[tag].clone() for tag in tags if tag in debiased_dict}

        param = (image, fmap, debiased_centroids, ib_dir + image_id + '.pkl')
        params.append(param)
        
        if len(params) >= (args.cpus * 8):
            Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
            params = []

    if len(params) > 0:
        Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
