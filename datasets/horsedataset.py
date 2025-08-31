import torch
import numpy as np
import os
from PIL import Image
import cv2
from datasets.pointdataset import PointDataset
import pickle
import utils.data

class HorseDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/horse10',
            crop_size=(384, 512),
            seq_len=None,
            only_first=False,
    ):
        super(HorseDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )
        print("loading horse dataset...")

        self.seq_len = seq_len
        self.only_first = only_first
        self.dname = 'hor'

        self.dataset_location = data_root
        self.anno_path = os.path.join(self.dataset_location, "seq_annotation.pkl")
        with open(self.anno_path, "rb") as f:
            self.annotation = pickle.load(f)

        self.video_names = []

        for video_name in list(self.annotation.keys()):
            video = self.annotation[video_name]

            rgbs = []
            trajs = []
            visibs = []
            for sample in video:
                img_path = sample["img_path"]
                img_path = self.dataset_location + '/' + img_path
                rgb = Image.open(img_path)
                rgbs.append(rgb)
                trajs.append(np.squeeze(sample["keypoints"], 0))
                visibs.append(np.squeeze(sample["keypoints_visible"], 0))

            rgbs = np.stack(rgbs, axis=0)
            trajs = np.stack(trajs, axis=0)
            visibs = np.stack(visibs, axis=0)
            valids = visibs.copy()

            S, H, W, C = rgbs.shape
            _, N, D = trajs.shape

            for si in range(S):
                # avoid 2px edge, since these are not really visible (according to adam)
                oob_inds = np.logical_or(
                    np.logical_or(trajs[si, :, 0] < 2, trajs[si, :, 0] >= W-2),
                    np.logical_or(trajs[si, :, 1] < 2, trajs[si, :, 1] >= H-2),
                )
                visibs[si, oob_inds] = 0

            rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
                rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

            N = trajs.shape[1]
            if N > 0:
                self.video_names.append(video_name)
            
        print(f"found {len(self.annotation)} unique videos in {self.dataset_location}")
        
    def getitem_helper(self, index):
        video_name = self.video_names[index]
        video = self.annotation[video_name]

        rgbs = []
        trajs = []
        visibs = []
        for sample in video:
            img_path = sample["img_path"]
            img_path = self.dataset_location + '/' + img_path
            rgb = Image.open(img_path)
            rgbs.append(rgb)
            trajs.append(np.squeeze(sample["keypoints"], 0))
            visibs.append(np.squeeze(sample["keypoints_visible"], 0))

        rgbs = np.stack(rgbs, axis=0)
        trajs = np.stack(trajs, axis=0)
        visibs = np.stack(visibs, axis=0)
        valids = visibs.copy()
        
        S, H, W, C = rgbs.shape
        _, N, D = trajs.shape

        for si in range(S):
            # avoid 2px edge, since these are not really visible (according to adam)
            oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < 2, trajs[si, :, 0] >= W-2),
                np.logical_or(trajs[si, :, 1] < 2, trajs[si, :, 1] >= H-2),
            )
            visibs[si, oob_inds] = 0

        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] /= W-1
        trajs[:,:,1] /= H-1
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgbs = np.stack(rgbs)
        H,W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()
        trajs = torch.from_numpy(trajs)
        visibs = torch.from_numpy(visibs)
        valids = torch.from_numpy(valids)

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids, 
            dname=self.dname,
        )
        return sample, True

    def __len__(self):
        return len(self.video_names)
