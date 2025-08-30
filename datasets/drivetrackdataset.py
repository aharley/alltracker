import torch
import numpy as np
import os
import glob
import cv2
from datasets.pointdataset import PointDataset
import pickle
import utils.data
from pathlib import Path

class DrivetrackDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/drivetrack',
            crop_size=(384, 512),
            seq_len=None,
            traj_per_sample=512,
            only_first=False,
    ):
        super(DrivetrackDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
        )
        print("loading drivetrack dataset...")

        self.dname = 'drivetrack'
        self.only_first = only_first
        S = seq_len

        self.dataset_location = Path(data_root)
        self.S = S
        video_fns = sorted(list(self.dataset_location.glob('*.npz')))

        self.video_fns = []
        for video_fn in video_fns[:100]: # drivetrack is huge and self-similar, so we trim to 100
            ds = np.load(video_fn, allow_pickle=True)
            rgbs, trajs, visibs = ds['video'], ds['tracks'], ds['visibles']
            # rgbs is T,1280,1920,3
            # trajs is N,T,2
            # visibs is N,T
            
            trajs = np.transpose(trajs, (1,0,2)).astype(np.float32) # S,N,2
            visibs = np.transpose(visibs, (1,0)).astype(np.float32) # S,N
            valids = visibs.copy()
            # print('N0', trajs.shape[1])

            # discard tracks with any inf/nan
            idx = np.nonzero(np.isfinite(trajs.sum(0).sum(1)))[0] # N
            trajs = trajs[:,idx]
            visibs = visibs[:,idx]
            valids = valids[:,idx]
            # print('N1', trajs.shape[1])

            if trajs.shape[1] < self.traj_per_sample:
                continue

            # shuffle and trim
            inds = np.random.permutation(trajs.shape[1])
            inds = inds[:10000]
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            # print('N2', trajs.shape[1])

            S,H,W,C = rgbs.shape

            # set OOB to invisible
            visibs[trajs[:, :, 0] > W-1] = False
            visibs[trajs[:, :, 0] < 0] = False
            visibs[trajs[:, :, 1] > H-1] = False
            visibs[trajs[:, :, 1] < 0] = False
            
            rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
                rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
            # print('N3', trajs.shape[1])

            if trajs.shape[1] < self.traj_per_sample:
                continue

            trajs = torch.from_numpy(trajs)
            visibs = torch.from_numpy(visibs)
            valids = torch.from_numpy(valids)
            # discard tracks that go far OOB
            crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
            close_pts_inds = torch.all(
                torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1) < max(H,W)*2,
                dim=0,
            )
            trajs = trajs[:, close_pts_inds]
            visibs = visibs[:, close_pts_inds]
            valids = valids[:, close_pts_inds]
            # print('N4', trajs.shape[1])

            if trajs.shape[1] < self.traj_per_sample:
                continue

            visible_inds = (valids[0]*visibs[0]).nonzero(as_tuple=False)[:, 0]
            trajs = trajs[:, visible_inds].float()
            visibs = visibs[:, visible_inds].float()
            valids = valids[:, visible_inds].float()
            # print('N5', trajs.shape[1])

            if trajs.shape[1] >= self.traj_per_sample:
                self.video_fns.append(video_fn)
        
        print(f"found {len(self.video_fns)} unique videos in {self.dataset_location}")
        
    def getitem_helper(self, index):
        video_fn = self.video_fns[index]
        ds = np.load(video_fn, allow_pickle=True)
        rgbs, trajs, visibs = ds['video'], ds['tracks'], ds['visibles']
        # rgbs is T,1280,1920,3
        # trajs is N,T,2
        # visibs is N,T

        trajs = np.transpose(trajs, (1,0,2)).astype(np.float32) # S,N,2
        visibs = np.transpose(visibs, (1,0)).astype(np.float32) # S,N
        valids = visibs.copy()

        # discard inf/nan
        idx = np.nonzero(np.isfinite(trajs.sum(0).sum(1)))[0] # N
        trajs = trajs[:,idx]
        visibs = visibs[:,idx]
        valids = valids[:,idx]

        # shuffle and trim
        inds = np.random.permutation(trajs.shape[1])
        inds = inds[:10000]
        trajs = trajs[:,inds]
        visibs = visibs[:,inds]
        valids = valids[:,inds]
        N = trajs.shape[1]
        # print('N2', trajs.shape[1])

        S,H,W,C = rgbs.shape
        # set OOB to invisible
        visibs[trajs[:, :, 0] > W-1] = False
        visibs[trajs[:, :, 0] < 0] = False
        visibs[trajs[:, :, 1] > H-1] = False
        visibs[trajs[:, :, 1] < 0] = False
        
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
        
        trajs = torch.from_numpy(trajs)
        visibs = torch.from_numpy(visibs)
        valids = torch.from_numpy(valids)

        # discard tracks that go far OOB
        crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1) < max(H,W)*2,
            dim=0,
        )
        trajs = trajs[:, close_pts_inds]
        visibs = visibs[:, close_pts_inds]
        valids = valids[:, close_pts_inds]
        # print('N3', trajs.shape[1])
        
        visible_pts_inds = (valids[0]*visibs[0]).nonzero(as_tuple=False)[:, 0]
        point_inds = torch.randperm(len(visible_pts_inds))[: self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            return None, False
        visible_inds_sampled = visible_pts_inds[point_inds]
        trajs = trajs[:, visible_inds_sampled].float()
        visibs = visibs[:, visible_inds_sampled].float()
        valids = valids[:, visible_inds_sampled].float()
        # print('N4', trajs.shape[1])

        trajs = trajs[:, :self.traj_per_sample]
        visibs = visibs[:, :self.traj_per_sample]
        valids = valids[:, :self.traj_per_sample]
        # print('N5', trajs.shape[1])

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )
        return sample, True

    def __len__(self):
        return len(self.video_fns)
