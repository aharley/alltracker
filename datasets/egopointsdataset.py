import torch
import numpy as np
from datasets.pointdataset import PointDataset
import utils.data
import cv2
from pathlib import Path

class EgoPointsDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/ego_points',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(EgoPointsDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        print('loading egopoints dataset...')

        self.dname = 'egopoints'
        self.only_first = only_first
        
        self.data = []
        for subfolder in Path(data_root).iterdir():
            if subfolder.is_dir():
                annot_fn = subfolder / 'annot.npz'
                if not annot_fn.exists():
                    continue
                data = np.load(annot_fn)
                trajs_2d, valids, visibs, vis_valids = data['trajs_2d'], data['valids'], data['visibs'], data['vis_valids']

                self.data.append({
                    'rgb_paths': sorted(subfolder.glob('rgbs/*.jpg')),
                    'trajs_2d': trajs_2d,
                    'valids': valids,
                    'visibs': visibs,
                    'vis_valids': vis_valids,
                })
                
        print('found %d videos in %s' % (len(self.data), data_root))
        
    def __getitem__(self, index):
        dat = self.data[index]
        rgb_paths = dat['rgb_paths']
        trajs = dat['trajs_2d'] # S,N,2
        valids = dat['valids'] # S,N
        visibs = valids.copy() # we don't use this 

        rgbs = [cv2.imread(str(rgb_path))[..., ::-1] for rgb_path in rgb_paths]
        
        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
            
        # resize
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgb0_raw = cv2.imread(str(rgb_paths[0]))[..., ::-1]
        trajs = trajs / (np.array([rgb0_raw.shape[1], rgb0_raw.shape[0]]) - 1)
        trajs = np.maximum(np.minimum(trajs, 1.), 0.)
        # 1.0,1.0 should map to the bottom-right corner pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        valids = torch.from_numpy(valids).float() # S,N
        visibs = torch.from_numpy(visibs).float()

        if self.seq_len is not None:
            rgbs = rgbs[:self.seq_len]
            trajs = trajs[:self.seq_len]
            valids = valids[:self.seq_len]
            visibs = visibs[:self.seq_len]

        # req at least one timestep valid (after cutting)
        val_ok = torch.sum(valids, axis=0) > 0
        trajs = trajs[:,val_ok]
        valids = valids[:,val_ok]
        visibs = visibs[:,val_ok]

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            valids=valids,
            visibs=visibs,
            dname=self.dname,
        )
        return sample, True

    def __len__(self):
        return len(self.data)


