import sys
import numpy as np

if sys.platform == 'linux':
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    from pydensecrf.utils import unary_from_softmax

class DenseCRF(object):
    def __init__(self, iter_max=10, pos_w=3, pos_xy_std=1, bi_w=4, bi_xy_std=67, bi_rgb_std=3, for_seg=False):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.for_seg = for_seg
    
    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        if not self.for_seg:
            bg = np.power(1 - np.max(probmap, axis=0, keepdims=True), 4)
            probmap = np.concatenate([bg, probmap], axis=0)

            C += 1
        
        U = unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        
        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))
        
        return Q

    def run_label(self, image, label, num_classes=21, gt_prob=0.7):
        """
        {
            'iter_max':10,
            'bi_w':10, 'bi_xy_std':50, 'bi_rgb_std':5, 
            'pos_w':3, 'pos_xy_std':3
        }
        """
        H, W = label.shape

        U = unary_from_labels(label, num_classes, gt_prob=gt_prob, zero_unsure=False)
        U = np.ascontiguousarray(U)
        
        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, num_classes)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((num_classes, H, W))
        Q = np.argmax(Q, axis=0)

        return Q