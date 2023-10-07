import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.plots import plot_one_box
import random

import cv2
from tqdm import tqdm

class Detector(object):
    def __init__(self, weights_path = 'yolov5s.pt', device='', colors=None):
        # self.weights = weights_path
        self.thresh = 0.3
        self.cls = ['person']
        self.colors = {0: (0, 0, 255), 5: (0, 255, 0)}
        self.weights = weights_path
        # self.device = gpu_id
        self.device = select_device(device)
        print('loading yolov5 model...')
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.float()
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        self.colors_random = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.colors = colors
        if self.colors:
            for color in self.colors:
                self.colors_random[color] = self.colors[color]

    def preprocess(self, img, img_size):
        img0 = img.copy()
        img = letterbox(img, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    # Process predictions 预测过程
    def detect(self, im, thresh=0.3, cls=None, img_size=640):
        if cls is None:
            cls = ['person']
        im0, img = self.preprocess(im, img_size)
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, thresh, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                # 将boxes大小从img_size调整为im0
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 打印类别
                for *x, conf, cls_id in det:
                    label_name = self.names[int(cls_id)]
                    if not label_name in cls:
                        continue
                    x1, y1 = int(x[0].cpu().detach().numpy().tolist()), int(x[1].cpu().detach().numpy().tolist())
                    x2, y2 = int(x[2].cpu().detach().numpy().tolist()), int(x[3].cpu().detach().numpy().tolist())
                    cls_id = int(cls_id.cpu().detach().numpy().tolist())
                    conf = round((conf.cpu().detach().numpy().tolist()), 4)
                    # pred_boxes.append(
                    #     (x1, y1, x2, y2, conf, cls_id, label_name))
                    pred_boxes.append(
                        (x1, y1, x2, y2, conf))
        # return im, pred_boxes
                    # pred_boxes = torch.tensor(pred_boxes,dtype=torch.float32)
        return torch.tensor(pred_boxes,dtype=torch.float32)

    # 画框
    def draw_box(self, im, pred_boxes):
        if not pred_boxes:
            print("can not draw box cause no objects!!")
        else:
            for pred_box in pred_boxes:
                plot_one_box(pred_box, im, label=pred_box[5]+" "+str(pred_box[6]), color=self.colors_random[int(pred_box[4])], line_thickness=3)
        return im

    # 显示结果
    def print_result(self, pred_boxes):
        if not pred_boxes:
            print("can not print result cause no objects!!")
        else:
            for i in range(len(pred_boxes)):
                print("{}st object is {}, class id is {}, x:{},y:{},w:{},h:{},conf:{}".format(i, pred_boxes[i][5], pred_boxes[i][4],
                                                                                              pred_boxes[i][0], pred_boxes[i][1],
                                                                                              (pred_boxes[i][2]-pred_boxes[i][0]),
                                                                                              (pred_boxes[i][3]-pred_boxes[i][1]),
                                                                                              pred_boxes[i][6]))





