import torch
import numpy as np
import pickle
from datasets.pointdataset import PointDataset
import utils.data
import cv2
from pathlib import Path
import io
from PIL import Image

def decode(frame):
    byteio = io.BytesIO(frame)
    img = Image.open(byteio)
    return np.array(img)

class KineticsDataset(PointDataset):
    def __init__(
        self,
        data_root='../datasets/tapvid_kinetics',
        crop_size=(384,512),
        seq_len=None,
        only_first=False,
    ):
        super(KineticsDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        print('loading TAPVID-Kinetics dataset...')

        self.dname = 'kinetics'
        self.only_first = only_first
        
        self.data = []
        for vid_pkl in sorted(list(Path(data_root).glob('*.pkl')))[:]:
            vid_pkl = vid_pkl.name
            print(vid_pkl)
            input_path = "%s/%s" % (data_root, vid_pkl)
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            self.data += data
        print("found %d videos in %s" % (len(self.data), data_root))
        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        if isinstance(rgbs[0], bytes):  # decode if needed
            rgbs = [decode(frame) for frame in rgbs]
        trajs = dat['points'] # N,S,2 array
        visibs = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when visib
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = visibs.copy()

        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
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


