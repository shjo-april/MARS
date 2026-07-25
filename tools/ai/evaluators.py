import numpy as np

from tools.ai import torch_utils

class Evaluator_For_Semantic_Segmentation:
    def __init__(self, class_names, ignore_index=255):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index

        self.clear()

    def clear(self):
        self.meter_dict = {
            'P' : np.zeros(self.num_classes, dtype=np.float32),
            'T' : np.zeros(self.num_classes, dtype=np.float32),
            'TP' : np.zeros(self.num_classes, dtype=np.float32)
        }
    
    def add(self, data):
        if isinstance(data, dict):
            pred_mask = torch_utils.get_numpy(data['pred_masks'])[0]
            gt_mask = torch_utils.get_numpy(data['gt_masks'])[0]
        else:
            pred_mask, gt_mask = data
        
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)

        obj_mask = gt_mask != self.ignore_index
        if self.ignore_index in pred_mask:
            pred_mask[pred_mask == self.ignore_index] = 0

        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.num_classes):
            self.meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
            self.meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
            self.meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
    
    def add_from_data(self, meter_dict):
        for i in range(self.num_classes):
            self.meter_dict['P'][i] += meter_dict['P'][i]
            self.meter_dict['T'][i] += meter_dict['T'][i]
            self.meter_dict['TP'][i] += meter_dict['TP'][i]
    
    def calculate_per_image(self, pred_mask, gt_mask, label, threshold=None, ignore_index=255, num_classes=21):
        if threshold is not None:
            pred_mask = np.pad(pred_mask, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            pred_mask = np.argmax(pred_mask, axis=0)

        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
        
        obj_mask = (gt_mask != ignore_index) 
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        IoUs = {}
        FPs = {}
        FNs = {}
        
        for i in range(num_classes):
            if i > 0:
                if label[i - 1] == 0:
                    continue
            
            P = np.sum((pred_mask==i)*obj_mask)
            T = np.sum((gt_mask==i)*obj_mask)
            TP = np.sum((gt_mask==i)*correct_mask)

            union = (T + P - TP)
            if union == 0:
                continue

            IoU = TP / union
            FP = (P - TP) / union
            FN = (T - TP) / union

            IoUs[self.class_names[i]] = float(IoU)
            FPs[self.class_names[i]] = float(FP)
            FNs[self.class_names[i]] = float(FN)
        
        return {
            'IoUs': IoUs,

            'FPs': FPs,
            'FNs': FNs
        }
    
    def get(self, clear=True):
        IoU_list = []
        FP_list = [] # over activation
        FN_list = [] # under activation

        TP = self.meter_dict['TP']
        P = self.meter_dict['P']
        T = self.meter_dict['T']

        detail_dict = {}
        
        for i in range(self.num_classes):
            union = (T[i] + P[i] - TP[i])

            IoU = TP[i] / union * 100
            FP = (P[i] - TP[i]) / union
            FN = (T[i] - TP[i]) / union

            detail_dict[self.class_names[i]] = {
                'IoU': IoU,
                'FP': FP,
                'FN': FN,
            }
            
            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.nanmean(IoU_list)
        FP = np.nanmean(FP_list[1:])
        FN = np.nanmean(FN_list[1:])
        
        if clear:
            self.clear()
        
        return mIoU
    
    def print(self, tag):
        mIoU = self.get(clear=False)
        print('[{}] mIoU: {:.2f}%'.format(tag, mIoU))