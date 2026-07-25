import os
import torch
import shutil
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from core import networks, datasets, losses, generators

from tools.ai import torch_utils, trainers, optimizers, evaluators, augment_utils
from tools.general import io_utils, log_utils, json_utils

def collate(batch):
    images = []
    weak_images = []
    masks = []
    labels = []
    crop_bboxes = []

    for image, weak_image, mask, label, crop_bbox in batch:
        images.append(torch.from_numpy(image))
        weak_images.append(torch.from_numpy(weak_image))
        masks.append(torch.from_numpy(mask))
        labels.append(torch.from_numpy(label))
        crop_bboxes.append(crop_bbox)
        
    data = {
        'images': torch.stack(images),
        'weak_images': torch.stack(weak_images),
        'masks': torch.stack(masks),
        'labels': torch.stack(labels),
        'crop_bboxes': np.stack(crop_bboxes),
    }
    return data

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(args):
    torch_utils.set_seed(args.seed)

    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    gpus = args.gpus.split(',')
    device = torch.device('cuda', 0)

    if RANK in [-1, 0]:
        for i, d in enumerate(gpus):
            p = torch.cuda.get_device_properties(i)
            print(f'[i] CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)')

    if LOCAL_RANK != -1:
        from torch import distributed as dist

        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)

        from datetime import timedelta
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))

        assert args.batch_size % WORLD_SIZE == 0, '--batch_size must be multiple of CUDA device count'
    
    # set directories
    txt_dir = './experiments/txt/'
    txt_path = txt_dir + f'{args.tag}.txt'

    model_dir = f'./experiments/models/{args.tag}/'
    tensorboard_dir = f'./experiments/tensorboards/{args.tag}/'
    
    if os.path.isfile(txt_path) and RANK in [-1, 0]:
        result = input('Found existing logs. yes=remove, no=keyboardinterept')
        if result == 'no':
            raise KeyboardInterrupt
        else:
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
            if os.path.isdir(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)

    txt_dir = io_utils.create_directory(txt_dir) if RANK in [-1, 0] else txt_dir
    model_dir = io_utils.create_directory(model_dir) if RANK in [-1, 0] else model_dir
    tensorboard_dir = io_utils.create_directory(tensorboard_dir) if RANK in [-1, 0] else tensorboard_dir

    log_func = lambda string='': log_utils.log_print(string, txt_path)

    # read dataset information
    data_dict = json_utils.read_json(f'./data/{args.dataset}.json')

    # create model
    if args.decoder == 'deeplabv3+':
        model = networks.DeepLabv3_Plus(
            args.backbone, 
            data_dict['num_classes']
        ).to(device)
    
    if RANK in [-1, 0]:
        log_func('[i] Backbone is {}'.format(args.backbone))
        log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
        log_func()

    # define loss functions
    class_loss_fn = losses.WeightedCrossEntropyLoss(data_dict['num_classes'], data_dict['ignore_index']).to(device)

    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, 
                True, args.ema_decay,
                args.max_epochs, tensorboard_dir, RANK
            )
            super().__init__(model, device, param)

            self.com_module = generators.Complementing_Module(
                device=device,
                ignore_index=data_dict['ignore_index'],
                refine_params={
                    'ema': self.ema,
                    'bias_class': args.bias_class,
                }
            )

        def prepare_dataset(self):
            train_transform, valid_transform = augment_utils.get_transform(args)

            if RANK in [-1, 0]:
                log_func('[i] train transform : {}'.format(train_transform))
                log_func('[i] test transform : {}'.format(valid_transform))

            self.train_dataset = datasets.Dataset_For_WSSS(
                args.root_dir, 
                args.train_domain, 
                train_transform, 
                tag=args.pseudo_tag,
                name=args.dataset
            )
            self.valid_dataset = datasets.Dataset_For_Segmentation(
                args.root_dir, 
                args.train_domain, 
                valid_transform, 
                name=args.dataset
            )
        
        def prepare_loader(self):
            if RANK != -1:
                shuffle = False
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            else:
                shuffle = True
                train_sampler = None
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size // WORLD_SIZE, num_workers=args.cpus, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=train_sampler, collate_fn=collate)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=max(args.cpus // 4, 1), shuffle=False, drop_last=False, pin_memory=True)
            
            if RANK in [-1, 0]:
                log_func('[i] The size of training set is {}'.format(len(self.train_dataset)))
        
        def configure_optimizers(self):
            self.optimizer = optimizers.SGD(
                params=[
                    {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                ],
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                scheduler_option={
                    'scheduler': args.scheduler,
                    'power': 0.9,
                    'max_iterations': self.train_iterations * self.param.max_epochs
                }
            )

        def forward(self, data, training: bool=True):
            if training:
                images = data['images'].to(self.device)
                masks = data['masks'].to(self.device)

                output_dict = self.model(images, same=True)
                
                if self.epoch > int(args.max_epochs * 0.60):
                    pseudo_dict = self.com_module(data['weak_images'], masks, data['labels'], data['crop_bboxes'])
                    for key in pseudo_dict.keys():
                        pseudo_dict[key] = pseudo_dict[key].detach().to(device)

                    masks = pseudo_dict['pseudo_masks']
                    if args.use_certainty:
                        certain_masks = pseudo_dict['certain_masks']
                    else:
                        certain_masks = None
                else:
                    certain_masks = None

                masks[masks == args.bias_class] = 0

                loss = class_loss_fn(output_dict['seg_logits'], masks, certain_masks)
                
                tb_dict = {
                    'Training/CE': loss.item(),
                    'Training/Learning rate': self.get_learning_rate(),
                }
                
                return loss, tb_dict
            else:
                images, masks = data

                if self.ema is not None:
                    model = self.ema.get_model()
                else:
                    model = torch_utils.de_parallel(self.model)

                images = images.to(self.device)
                masks = masks.to(self.device)

                _, c, h, w = images.size()

                params = {
                    "image": images[0],
                    "image_size": (h, w),
                    "scales": [1.0],
                    "hflip": False
                }
                output_dict = model.forward_with_scales(**params)

                return {
                    'pred_masks': output_dict['pred_mask'],
                    'gt_masks': masks,
                }

        def evaluation_step(self, debug=False):
            if self.ema is None:
                self.model.eval()
            
            self.evaluator = evaluators.Evaluator_For_Semantic_Segmentation(data_dict['class_names'])

            valid_time = super().evaluation_step(debug, self.valid_loader)
            valid_mIoU, valid_FP, valid_FN = self.evaluator.get()

            train_time = super().evaluation_step(debug, self.valid_loader_for_train)
            train_mIoU, train_FP, train_FN = self.evaluator.get()
            
            if self.ema is None:
                self.model.train()
            
            if self.best_valid_score < valid_mIoU:
                self.best_valid_score = valid_mIoU

            self.update_tensorboard(
                {
                    'Evaluation/Best mIoU': self.best_valid_score,

                    'Evaluation/valid mIoU': valid_mIoU,
                    'Evaluation/valid FP': valid_FP,
                    'Evaluation/valid FN': valid_FN,

                    'Evaluation/train mIoU': train_mIoU,
                    'Evaluation/train FP': train_FP,
                    'Evaluation/train FN': train_FN,
                }
            )
            
            return {
                'best_valid_mIoU': self.best_valid_score, 

                'valid_mIoU': valid_mIoU, 
                'valid_FP': valid_FP, 
                'valid_FN': valid_FN, 

                'train_mIoU': train_mIoU, 
                'train_FP': train_FP, 
                'train_FN': train_FN, 

                'time': train_time + valid_time
            }
    
    # train
    trainer = Trainer()

    for epoch in range(trainer.epoch, args.max_epochs+1):
        train_dict = trainer.training_step(args.debug)

        train_dict['epoch'] = epoch
        train_dict['CE'] = train_dict['Training/CE']
        train_dict['lr'] = train_dict['Training/Learning rate']
        
        log_func('[i] epoch={epoch:,}, lr={lr:.6f}, CE={CE:.4f}, time={time:.0f}sec'.format(**train_dict))
    
    trainer.save_model(model_dir + 'last.pth')

if __name__ == '__main__':
    parser = io_utils.Parser()

    # environment
    parser.add('local_rank', -1, int)
    parser.add('seed', 0, int)

    parser.add('gpus', '0', str)
    parser.add('cpus', 8, int)

    # dataset
    parser.add('dataset', 'VOC', str)
    parser.add('root_dir', '../VOC2012/', str)

    parser.add('train_domain', 'train_aug', str)

    parser.add('pseudo_tag', 'RS+EPM@STEGO@K_bg=2@K_fg=2@alpha(0.40)@MARS', str)
    parser.add('bias_class', 254, int)

    # network architecture
    parser.add('backbone', 'resnet101', str)
    parser.add('decoder', 'deeplabv3+', str)

    parser.add('image_size', 512, int)
    
    # training configuration
    parser.add('batch_size', 16, int)

    parser.add('tag', 'ResNet-101@VOC@RS+EPM_with_MARS', str)

    parser.add('max_epochs', 100, int)
    parser.add('valid_ratio', 5.0, float)

    parser.add('lr', 1e-3, float)
    parser.add('wd', 4e-5, float)
    parser.add('momentum', 0.9, float)
    parser.add('nesterov', True, bool)

    parser.add('optimizer', 'SGD', str)
    parser.add('scheduler', 'PolyLR', str)

    parser.add('ema_decay', 0.999, float)
    parser.add('debug', False, bool)
    
    # complementing debiased labels
    parser.add('use_certainty', True, bool)

    # data augmentation
    parser.add('bg_mask', 255, int)

    parser.add('min_scale', 0.5, float) 
    parser.add('max_scale', 2.0, float) 

    parser.add('min_image_size', 256, int)
    parser.add('max_image_size', 1024, int)

    parser.add('b_factor', 0.5, float)
    parser.add('c_factor', 0.5, float)
    parser.add('s_factor', 0.5, float)
    parser.add('h_factor', 0.3, float)

    parser.add('zoom_factor', 1, int)

    parser.add('train_augment', 'minmax_scale-hflip-colorjitter-gray-blur-normalize-random_crop-transpose', str) 
    parser.add('test_augment', 'normalize-transpose', str)
    
    main(parser.get_args())