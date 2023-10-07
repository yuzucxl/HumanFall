import os
import cv2
import torch

from SPPE.src.main_fast_inference import InferenNet_fast, InferenNet_fastRes50
from SPPE.src.utils.img import crop_dets
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction

# pose_estimator = SPPE_FastPose(inp_h, inp_w)
class SPPE_FastPose(object):
    def __init__(self,
                 backbone='resnet50',
                 input_height=320,
                 input_width=256,
                 device='cuda'):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device

        if backbone == 'resnet101':
            self.model = InferenNet_fast(weights_file='./Models1/sppe/fast_res101_320x256.pth').to(device)
        else:
            self.model = InferenNet_fastRes50(weights_file='./Models1/sppe/fast_res50_256x192.pth').to(device)
        self.model.eval()
        # print('show pose-model device')
        # print(next(self.model.parameters()).device)   cuda:0

    def predict(self, image, bboxs, bboxs_scores):
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        # pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores) #将结果进行NMS处理
        return result