import os
import cv2
import copy
import faiss
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F

from PIL import Image
from joblib import Parallel, delayed

from tools.general import io_utils, cv_utils, pickle_utils, xml_utils, json_utils

def get_tags(xml_path):
    _, classes = xml_utils.read_xml(xml_path, keep_difficult=True)
    tags = np.unique(classes)
    return tags

def generate(shape, fmap, wsss_mask, path, class_names, patch_size, K_bg, K_fg):
    h, w = shape
    _, _, fh, fw = fmap.shape

    fmap = F.normalize(fmap, dim=1)

    # decomposition
    if patch_size > 0:
        while w % patch_size != 0: w += 1
        while h % patch_size != 0: h += 1
        wsss_mask = cv2.resize(wsss_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    emb_dict = {}

    for fy in range(fh):
        y = int(fy / fh * h)
        for fx in range(fw):
            x = int(fx / fw * w)

            class_indices = np.unique(wsss_mask[y:y+patch_size, x:x+patch_size])
            if len(class_indices) > 1:
                continue
            
            class_index = class_indices[0]
            
            emb = fmap[0, :, fy, fx].numpy()
            tag = class_names[class_index]

            try: emb_dict[tag].append(emb)
            except KeyError: emb_dict[tag] = [emb]
    
    # clustering
    data_dict = {}

    for tag in emb_dict.keys():
        if tag == 'background': K = K_bg
        else: K = K_fg

        if len(emb_dict[tag]) > K:
            embs = copy.deepcopy(emb_dict[tag])
            embs = np.asarray(embs)

            if K == 1:
                embs = np.asarray(embs)
                embs = np.mean(embs, axis=0, keepdims=True)

                data_dict[tag] = embs
            else:
                kmeans = faiss.Kmeans(
                    embs.shape[1], k=K, niter=300, nredo=10, spherical=False, 
                    verbose=False, min_points_per_centroid=1, max_points_per_centroid=10000000, seed=0
                )
                kmeans.train(embs)
                
                data_dict[tag] = kmeans.centroids
    
    if len(data_dict.keys()) > 0:
        pickle_utils.dump_pickle(path, data_dict)

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

    parser.add('uss', 'STEGO', str)
    parser.add('patch_size', 8, int)

    # clustering
    parser.add('K_fg', 2, int)
    parser.add('K_bg', 2, int)
    
    args = parser.get_args()

    image_dir = args.root_dir + f'{args.domain}/image/'
    xml_dir = args.root_dir + f'{args.domain}/xml/'
    uss_dir = args.root_dir + f'{args.domain}/pickles_using_{args.uss}/'
    wsss_dir = args.root_dir + f'{args.domain}/{args.wsss}/'

    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    ib_key = f'K_bg={args.K_bg}@K_fg={args.K_fg}'
    ib_dir = io_utils.create_directory(args.root_dir + f'{args.domain}/{args.wsss}@{args.uss}@{ib_key}/')
    
    image_ids = os.listdir(image_dir)
    image_ids = [image_id.replace('.jpg', '') for image_id in image_ids]
    
    i = 0
    params = []
    
    colors = cv_utils.get_colors_from_length()[..., ::-1]

    for image_id in tqdm(image_ids, 'preprocess'):
        path = ib_dir + image_id + '.pkl'
        tags = get_tags(xml_dir + image_id + '.xml')

        if len(tags) == 0:
            continue
        
        wsss_mask = Image.open(wsss_dir + image_id + '.png').convert('P')
        wsss_mask = np.asarray(wsss_mask)

        uss_fmap = pickle_utils.load_pickle(uss_dir + image_id + '.pkl')
        
        param = (wsss_mask.shape, uss_fmap, wsss_mask, path, data_dict['class_names'], args.patch_size, args.K_bg, args.K_fg)
        params.append(param)

        if len(params) >= (args.cpus * 8):
            Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
            params = []

    if len(params) > 0:
        Parallel(n_jobs=args.cpus, verbose=10, pre_dispatch='all')([delayed(generate)(*param) for param in params])
