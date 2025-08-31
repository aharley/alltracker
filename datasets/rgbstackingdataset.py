import torch
import numpy as np
import pickle
from datasets.pointdataset import PointDataset
import utils.data
import cv2

class RGBStackingDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/tapvid_rgbstacking',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(RGBStackingDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        print('loading TAPVID-RGB-Stacking dataset...')

        self.dname = 'rgbstacking'
        self.only_first = only_first
        
        input_path = '%s/tapvid_rgb_stacking.pkl' % data_root
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data
        print('found %d videos in %s' % (len(self.data), data_root))
        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        visibs = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when visib
        valids = visibs.copy()
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = valids.transpose(1,0) # S,N

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

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids, 
            dname=self.dname,
        )
        return sample, True

    def __len__(self):
        return len(self.data)


