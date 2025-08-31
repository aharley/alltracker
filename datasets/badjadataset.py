import torch
import numpy as np
from datasets.pointdataset import PointDataset
import utils.data
import cv2
import glob
import imageio
import pandas as pd

# this is badja with annotations hand-cleaned by adam in 2022

class BadjaDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/badja',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(BadjaDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        self.dname = 'badja'
        self.seq_len = seq_len
        self.only_first = only_first
        
        npzs = glob.glob('%s/complete_aa/*.npz' % self.data_root)
        npzs = sorted(npzs)
        df = pd.read_csv('%s/picks_and_coords.txt' % self.data_root, sep=' ', header=None)
        track_names = df[0].tolist()
        pick_frames = np.array(df[1])

        self.animal_names = []
        self.animal_trajs = []
        self.animal_visibs = []
        self.animal_valids = []
        self.animal_picks = []
        
        for ind in range(len(npzs)):
            o = np.load(npzs[ind])

            animal_name = o['animal_name']
            trajs = o['trajs_g']
            valids = o['valids_g']

            S, N, D = trajs.shape

            assert(D==2)
            
            N = trajs.shape[1]

            # hand-picked frame where it's fair to start tracking this kp
            pick_g = np.zeros((N), dtype=np.int32)

            for n in range(N):
                short_name = '%s_%02d' % (animal_name, n)
                txt_id = track_names.index(short_name)
                pick_id = pick_frames[txt_id]
                pick_g[n] = pick_id

                # discard annotations before the pick
                valids[:pick_id,n] = 0
                valids[pick_id,n] = 2

            self.animal_names.append(animal_name)
            self.animal_trajs.append(trajs)
            self.animal_valids.append(valids)

    def __getitem__(self, index):
        animal_name = self.animal_names[index]
        trajs = self.animal_trajs[index].copy()
        valids = self.animal_valids[index]
        valids = (valids==2) * 1.0
        visibs = valids.copy()
            
        S,N,D = trajs.shape

        filenames = glob.glob('%s/videos/%s/*.png' % (self.data_root, animal_name)) + glob.glob('%s/videos/%s/*.jpg' % (self.data_root, animal_name))
        filenames = sorted(filenames)
        S = len(filenames)
        filenames_short = [fn.split('/')[-1] for fn in filenames]
        
        rgbs = []
        for s in range(S):
            filename_actual = filenames[s]
            rgb = imageio.imread(filename_actual)
            rgbs.append(rgb)

        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

        S = len(rgbs)
        H, W, C = rgbs[0].shape
        N = trajs.shape[1]
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        sx = W / self.crop_size[1]
        sy = H / self.crop_size[0]
        trajs[:,:,0] /= sx
        trajs[:,:,1] /= sy
        rgbs = np.stack(rgbs, 0)
        H, W, C = rgbs[0].shape
        
        rgbs = torch.from_numpy(rgbs).reshape(S, H, W, 3).permute(0,3,1,2).float()
        trajs = torch.from_numpy(trajs).reshape(S, N, 2).float()
        visibs = torch.from_numpy(visibs).reshape(S, N).float()
        valids = torch.from_numpy(valids).reshape(S, N).float()

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )
        return sample, True
        
    def __len__(self):
        return len(self.animal_names)
    
