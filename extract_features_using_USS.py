import os
import glob
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from core.USS import STEGO
from tools.general import io_utils, pickle_utils

def preprocess(image, patch_size=8):
    w, h = image.size
    while w % patch_size != 0: w += 1
    while h % patch_size != 0: h += 1

    transform = transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    return transform(image)

if __name__ == '__main__':
    parser = io_utils.Parser()

    # gpu
    parser.add('gpus', '0', str)

    # dataset
    parser.add('root_dir', '../VOC2012/', str)
    parser.add('domain', 'train_aug', str)

    # uss
    parser.add('uss', 'STEGO', str)
    parser.add('weight_path', './weights/stego_voc.ckpt', str)
    
    args = parser.get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.uss == 'STEGO':
        patch_size = 8
        model = STEGO.load_from_checkpoint(args.weight_path).net
    else:
        raise Exception("Please implement an unsupervised method")

    model.eval()
    model.cuda()

    res_dir = io_utils.create_directory(f'{args.root_dir}{args.domain}/pickles_using_{args.uss}/')
    image_paths = glob.glob(f'{args.root_dir}{args.domain}/image/*')
    
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        extension = image_name.split('.')[-1]
        image_id = image_name.replace('.' + extension, '')

        if os.path.isfile(res_dir + f'{image_id}.pkl'):
            continue

        image = Image.open(image_path).convert('RGB')
        image = preprocess(image, patch_size)
        image = image.cuda().unsqueeze(0)
        
        with torch.no_grad():
            feats1 = model(image)
            feats2 = model(image.flip(dims=[3]))

            feats = (feats1 + feats2.flip(dims=[3])) / 2
            feats = feats.cpu()
        
        pickle_utils.dump_pickle(res_dir + f'{image_id}.pkl', feats)
    