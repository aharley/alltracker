import torch
import numpy as np
import os
from PIL import Image
import cv2
from datasets.pointdataset import PointDataset
import utils.data

class CrohdDataset(PointDataset):
    def __init__(
            self,
            data_root='../datasets/crohd',
            crop_size=(384, 512),
            seq_len=None,
            only_first=False,
    ):
        super(CrohdDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        self.dname = 'crohd'
        self.seq_len = seq_len
        self.only_first = only_first

        dataset_dir = "%s/HT21/train" % self.data_root
        label_location = "%s/HT21Labels/train" % self.data_root
        subfolders = ["HT21-01", "HT21-02", "HT21-03", "HT21-04"]

        print("loading data from {0}".format(dataset_dir))
        self.dataset_dir = dataset_dir
        self.subfolders = subfolders
        print("found %d samples" % len(self.subfolders))

    def getitem_helper(self, index):
        subfolder = self.subfolders[index]

        label_path = os.path.join(self.dataset_dir, subfolder, "gt/gt.txt")
        labels = np.loadtxt(label_path, delimiter=",")

        n_frames = int(labels[-1, 0])
        n_heads = int(labels[:, 1].max())

        bboxes = np.zeros((n_frames, n_heads, 4))
        visibs = np.zeros((n_frames, n_heads))

        for i in range(labels.shape[0]):
            (
                frame_id,
                head_id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                conf,
                cid,
                vis,
            ) = labels[i]
            frame_id = int(frame_id) - 1  # convert 1-indexing to 0-indexing
            head_id = int(head_id) - 1  # convert 1-indexing to 0-indexing

            visibs[frame_id, head_id] = vis
            box_cur = np.array(
                [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]
            )  # convert xywh to x1, y1, x2, y2
            bboxes[frame_id, head_id] = box_cur

        prescale = 0.75 # to save memory

        # take the center of each head box as a coordinate
        trajs = np.stack([bboxes[:, :, [0, 2]].mean(2), bboxes[:, :, [1, 3]].mean(2)], axis=2)  # S,N,2
        trajs = trajs * prescale
        valids = visibs.copy()

        S, N = valids.shape

        rgbs = []
        for ii in range(S):
            rgb_path = os.path.join(self.dataset_dir, subfolder, "img1", str(ii + 1).zfill(6) + ".jpg")
            rgb = Image.open(rgb_path) # 1920x1080
            rgb = rgb.resize((int(rgb.size[0] * prescale), int(rgb.size[1] * prescale)), Image.BILINEAR)  # save memory by downsampling here
            rgbs.append(rgb)
        rgbs = np.stack(rgbs) # S,H,W,3

        rgbs, trajs, visibs, valids = utils.data.standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

        H, W = rgbs[0].shape[:2]
        S, N = trajs.shape[:2]
        sx = W / self.crop_size[1]
        sy = H / self.crop_size[0]
        trajs[:,:,0] /= sx
        trajs[:,:,1] /= sy
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgbs = np.stack(rgbs)
        H,W = rgbs[0].shape[:2]
        
        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()
        trajs = torch.from_numpy(trajs).float()
        visibs = torch.from_numpy(visibs).float()
        valids = torch.from_numpy(valids).float()

        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )
        return sample, True
        
    def __len__(self):
        return len(self.subfolders)


