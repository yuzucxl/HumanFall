import time
import torch
import numpy as np
import torchvision.transforms as transforms

from queue import Queue
from threading import Thread

from Detection.Models import Darknet
from Detection.Utils import non_max_suppression, ResizePadding


class TinyYOLOv3_onecls(object):
    """加载训练好的 Tiny-YOLOv3 一类（人）检测模型。
    参数:
        input_size: (int)   输入图像的大小必须能被 32 整除。默认值：416,
        config_file: (str)  Yolo 模型结构配置文件的路径.,
        weight_file: (str)  训练权重文件的路径.,
        nms: (float) Non-Maximum Suppression overlap threshold 非最大抑制重叠阈值.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off 要切断的预测 bbox 的最小置信度阈值.,
        device: (str) 在“cpu”或“cuda”上加载模型的设备.
    """

    def __init__(self,
                 input_size=416,
                 config_file='./Models1/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                 weight_file='./Models1/yolo-tiny-onecls/best-model1.pth',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size
        self.model = Darknet(config_file).to(device)
        print('loading yolov3 model...') #自己添加的
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()

    # def detect(self, image, need_resize=True, expand_bb=5):
    def detect(self, image, need_resize=False, expand_bb=10):
        """Feed forward to the model.前馈到模型
        Args:参数
            image: (numpy array) Single RGB image to detect.,检测单个RGB图像。
            need_resize: (bool) Resize to input_size before feed and will return bboxs
                with scale to image original size.,
                在喂食前将大小调整为input_size，并将返回bboxs与图像原始大小的比例
            expand_bb: (int) Expand boundary of the bboxs.扩展bboxes的边界
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        每个检测到的对象包含一个[上，左，下，右，bbox_score, class_score，类]，如果没有检测到，返回' None '。
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)

        image = self.transf_fn(image)[None, ...]
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]
        detected = self.model(image.to(self.device))
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]

        if detected is not None:
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2  #不变
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2  #不变
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

            detected = detected[:, 0:5]

        return detected


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







