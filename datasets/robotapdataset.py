import torch
import numpy as np
import pickle
from datasets.pointdataset import PointDataset
import utils.data
import cv2

class RobotapDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/robotap',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(RobotapDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        self.dname = 'robo'
        self.only_first = only_first
        
        # self.train_pkls = ['robotap_split0.pkl', 'robotap_split1.pkl', 'robotap_split2.pkl']
        self.val_pkls = ['robotap_split3.pkl', 'robotap_split4.pkl']

        print("loading robotap dataset...")
        # self.vid_pkls = self.train_pkls if is_training else self.val_pkls
        self.data = []
        for vid_pkl in self.val_pkls:
            print(vid_pkl)
            input_path = "%s/%s" % (data_root, vid_pkl)
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            keys = list(data.keys())
            self.data += [data[key] for key in keys]
        print("found %d videos in %s" % (len(self.data), data_root))

    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        dat = self.data[index]
        rgbs = dat["video"]  # list of H,W,C uint8 images
        trajs = dat["points"]  # N,S,2 array
        visibs = 1 - dat["occluded"]  # N,S array

        # note the annotations are only valid when not occluded
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = visibs.copy()

        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
        
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1
        
        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        visibs = torch.from_numpy(visibs).float() # S,N
        valids = torch.from_numpy(valids).float() # S,N

        if self.seq_len is not None:
            rgbs = rgbs[:self.seq_len]
            trajs = trajs[:self.seq_len]
            valids = valids[:self.seq_len]
            visibs = visibs[:self.seq_len]

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )
        return sample, True
