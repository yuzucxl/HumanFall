import os
import torch
import numpy as np
from processor.io import IO
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose
from net.st_gcn import Model


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """

    def __init__(self,
                 weight_file='./Models1/TSSTG/tsstg-model.pth',
                 device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down']
        self.num_class = len(self.class_names)
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        print('loading stgcn model...')
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)

        out = self.model((pts, mot))

        return out.detach().cpu().numpy()


class YuzuActionModel():

    def __init__(self,
                 weight_file='./models/kinetics-st_gcn.pt ',
                 device='cuda'):
        self.graph_args = {'layout': 'openpose_new', 'strategy': 'spatial'}
        self.class_names = []
        self.num_class = len(self.class_names)
        self.device = device
        self.model = Model(
            in_channels=3,
            num_class=400,
            edge_importance_weighting=True,
            graph_args=self.graph_args
        ).to(self.device)
        # self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        print('loading stgcn model...')
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

    def load_label_names(self):
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
        # print(label_name)
        self.class_names = label_name

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]

        # mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        # mot = mot.to(self.device)
        # pts = pts.to(self.device)
        N, C, T, V = pts.shape

        # 扩充第5维,大小为1
        pts = pts.unsqueeze(-1)
        pts = pts.to(self.device)
        out = self.model(pts)

        return out.detach().cpu().numpy()