import os
import gzip
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from typing import List, Optional, Any, Dict, Tuple, IO, TypeVar, Type, get_args, get_origin, Union, Any
from datasets.pointdataset import PointDataset
import json
import dataclasses
from dataclasses import dataclass, Field, MISSING
import utils.data
import cv2
import random

_X = TypeVar("_X")


def load_dataclass(f: IO, cls: Type[_X], binary: bool = False) -> _X:
    """
    Loads to a @dataclass or collection hierarchy including dataclasses
    from a json recursively.
    Call it like load_dataclass(f, typing.List[FrameAnnotationAnnotation]).
    raises KeyError if json has keys not mapping to the dataclass fields.

    Args:
        f: Either a path to a file, or a file opened for writing.
        cls: The class of the loaded dataclass.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)

    # in the list case, run a faster "vectorized" version
    cls = get_args(cls)[0]
    res = list(_dataclass_list_from_dict_list(asdict, cls))

    return res


def _resolve_optional(type_: Any) -> Tuple[bool, Any]:
    """Check whether `type_` is equivalent to `typing.Optional[T]` for some T."""
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_


def _unwrap_type(tp):
    # strips Optional wrapper, if any
    if get_origin(tp) is Union:
        args = get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):  # noqa: E721
            # this is typing.Optional
            return args[0] if args[1] is type(None) else args[1]  # noqa: E721
    return tp


def _get_dataclass_field_default(field: Field) -> Any:
    if field.default_factory is not MISSING:
        # pyre-fixme[29]: `Union[dataclasses._MISSING_TYPE,
        #  dataclasses._DefaultFactory[typing.Any]]` is not a function.
        return field.default_factory()
    elif field.default is not MISSING:
        return field.default
    else:
        return None


def _dataclass_list_from_dict_list(dlist, typeannot):
    """
    Vectorised version of `_dataclass_from_dict`.
    The output should be equivalent to
    `[_dataclass_from_dict(d, typeannot) for d in dlist]`.

    Args:
        dlist: list of objects to convert.
        typeannot: type of each of those objects.
    Returns:
        iterator or list over converted objects of the same length as `dlist`.

    Raises:
        ValueError: it assumes the objects have None's in consistent places across
            objects, otherwise it would ignore some values. This generally holds for
            auto-generated annotations, but otherwise use `_dataclass_from_dict`.
    """

    cls = get_origin(typeannot) or typeannot

    if typeannot is Any:
        return dlist
    if all(obj is None for obj in dlist):  # 1st recursion base: all None nodes
        return dlist
    if any(obj is None for obj in dlist):
        # filter out Nones and recurse on the resulting list
        idx_notnone = [(i, obj) for i, obj in enumerate(dlist) if obj is not None]
        idx, notnone = zip(*idx_notnone)
        converted = _dataclass_list_from_dict_list(notnone, typeannot)
        res = [None] * len(dlist)
        for i, obj in zip(idx, converted):
            res[i] = obj
        return res

    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        return _dataclass_list_from_dict_list(dlist, contained_type)

    # otherwise, we dispatch by the type of the provided annotation to convert to
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        # For namedtuple, call the function recursively on the lists of corresponding keys
        types = cls.__annotations__.values()
        dlist_T = zip(*dlist)
        res_T = [
            _dataclass_list_from_dict_list(key_list, tp) for key_list, tp in zip(dlist_T, types)
        ]
        return [cls(*converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, (list, tuple)):
        # For list/tuple, call the function recursively on the lists of corresponding positions
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(dlist[0])
        dlist_T = zip(*dlist)
        res_T = (
            _dataclass_list_from_dict_list(pos_list, tp) for pos_list, tp in zip(dlist_T, types)
        )
        if issubclass(cls, tuple):
            return list(zip(*res_T))
        else:
            return [cls(converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, dict):
        # For the dictionary, call the function recursively on concatenated keys and vertices
        key_t, val_t = get_args(typeannot)
        all_keys_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.keys()], key_t
        )
        all_vals_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.values()], val_t
        )
        indices = np.cumsum([len(obj) for obj in dlist])
        assert indices[-1] == len(all_keys_res)

        keys = np.split(list(all_keys_res), indices[:-1])
        all_vals_res_iter = iter(all_vals_res)
        return [cls(zip(k, all_vals_res_iter)) for k in keys]
    elif not dataclasses.is_dataclass(typeannot):
        return dlist

    # dataclass node: 2nd recursion base; call the function recursively on the lists
    # of the corresponding fields
    assert dataclasses.is_dataclass(cls)
    fieldtypes = {
        f.name: (_unwrap_type(f.type), _get_dataclass_field_default(f))
        for f in dataclasses.fields(typeannot)
    }

    # NOTE the default object is shared here
    key_lists = (
        _dataclass_list_from_dict_list([obj.get(k, default) for obj in dlist], type_)
        for k, (type_, default) in fieldtypes.items()
    )
    transposed = zip(*key_lists)
    return [cls(*vals_as_tuple) for vals_as_tuple in transposed]


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]

@dataclass
class DynamicReplicaFrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    meta: Optional[Dict[str, Any]] = None

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None


class DynamicReplicaDataset(PointDataset):
    def __init__(
            self,
            data_root,
            split="train",
            traj_per_sample=256,
            traj_max_factor=24, # multiplier on traj_per_sample
            crop_size=None,
            use_augs=False,
            seq_len=64,
            strides=[2,3],
            shuffle_frames=False,
            shuffle=False,
            only_first=False,
    ):
        super(DynamicReplicaDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
        )
        print('loading dynamicreplica dataset...')
        self.data_root = data_root
        self.only_first = only_first
        self.traj_max_factor = traj_max_factor
        self.seq_len = seq_len
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.crop_size = crop_size
        self.shuffle_frames = shuffle_frames
        frame_annotations_file = f"frame_annotations_{split}.jgz"
        self.sample_list = []
        with gzip.open(
            os.path.join(data_root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])
        seq_annot = defaultdict(list)
        for frame_annot in frame_annots_list:
            if frame_annot.camera_name == "left":
                seq_annot[frame_annot.sequence_name].append(frame_annot)
            # if os.path.isfile(traj_2d_file) and os.path.isfile(visib_file) and os.path.isfile(valid_file):
            #     self.sequences.append(seq)
                
        clip_step = 64

        for seq_name in seq_annot.keys():
            S_local = len(seq_annot[seq_name])
            # print(seq_name, 'S_local', S_local)

            traj_path = os.path.join(self.data_root, self.split, seq_annot[seq_name][0].trajectories['path'])
            if os.path.isfile(traj_path):
                for stride in strides:
                    for ref_idx in range(0, S_local-seq_len*stride, clip_step):
                        full_idx = ref_idx + np.arange(seq_len)*stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        full_idx = np.array(full_idx).astype(np.int32)
                        if len(full_idx)==seq_len:
                            sample = [seq_annot[seq_name][fi] for fi in full_idx]
                            self.sample_list.append(sample)
        print('found %d unique videos in %s (split=%s)' % (len(self.sample_list), data_root, split))
        self.dname = 'dynrep%d' % seq_len

        if shuffle:
            random.shuffle(self.sample_list)
                
    def __len__(self):
        return len(self.sample_list)

    def getitem_helper(self, index):
        sample = self.sample_list[index]
        T = len(sample)
        rgbs, visibs, trajs = [], [], []

        H, W = sample[0].image.size
        image_size = (H, W)

        for i in range(T):
            traj_path = os.path.join(self.data_root, self.split, sample[i].trajectories["path"])
            traj = torch.load(traj_path, weights_only=False)

            visibs.append(traj["verts_inds_vis"].numpy())

            rgbs.append(traj["img"].numpy())
            trajs.append(traj["traj_2d"].numpy()[..., :2])

        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        trajs = np.stack(trajs)
        visibs = np.stack(visibs)
        T, N, D = trajs.shape

        H,W = rgbs[0].shape[:2]
        visibs[trajs[:, :, 0] > W-1] = False
        visibs[trajs[:, :, 0] < 0] = False
        visibs[trajs[:, :, 1] > H-1] = False
        visibs[trajs[:, :, 1] < 0] = False
        

        if self.use_augs and np.random.rand() < 0.5:
            # time flip
            rgbs = np.flip(rgbs, axis=[0]).copy()
            trajs = np.flip(trajs, axis=[0]).copy()
            visibs = np.flip(visibs, axis=[0]).copy()
            
        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]
            
        assert(trajs.shape[0] == self.seq_len)

        if self.only_first:
            vis_ok = np.nonzero(visibs[0]==1)[0]
            trajs = trajs[:,vis_ok]
            visibs = visibs[:,vis_ok]

        N = trajs.shape[1]
        if N < self.traj_per_sample:
            print('dyn: N after vis0', N)
            return None, False

        # the data is quite big: 720x1280
        if H > self.crop_size[0]*2 and W > self.crop_size[1]*2 and np.random.rand() < 0.5:
            scale = 0.5
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            H, W = rgbs[0].shape[:2]
            rgbs = np.stack(rgbs, axis=0) # S,H,W,3
            trajs = trajs * scale
            # print('resized rgbs', rgbs.shape)
            
        if H > self.crop_size[0]*2 and W > self.crop_size[1]*2 and np.random.rand() < 0.5:
            scale = 0.5
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            H, W = rgbs[0].shape[:2]
            rgbs = np.stack(rgbs, axis=0) # S,H,W,3
            trajs = trajs * scale
            # print('resized rgbs', rgbs.shape)

        if self.use_augs and np.random.rand() < 0.98:
            H, W = rgbs[0].shape[:2]
            
            rgbs, trajs, visibs = self.add_photometric_augs(
                rgbs, trajs, visibs, replace=False,
            )
            if np.random.rand() < 0.2:
                rgbs, trajs = self.add_spatial_augs(rgbs, trajs, visibs, self.crop_size)
            else:
                rgbs, trajs = self.follow_crop(rgbs, trajs, visibs, self.crop_size)
            if np.random.rand() < self.rot_prob:
                # note this is OK since B==1
                # otw we would do it before this func
                rgbs = [np.transpose(rgb, (1,0,2)).copy() for rgb in rgbs]
                rgbs = np.stack(rgbs)
                trajs = np.flip(trajs, axis=2).copy()
            H, W = rgbs[0].shape[:2]
            if np.random.rand() < self.h_flip_prob:
                rgbs = [rgb[:, ::-1].copy() for rgb in rgbs]
                trajs[:, :, 0] = W - trajs[:, :, 0]
                rgbs = np.stack(rgbs)
            if np.random.rand() < self.v_flip_prob:
                rgbs = [rgb[::-1].copy() for rgb in rgbs]
                trajs[:, :, 1] = H - trajs[:, :, 1]
                rgbs = np.stack(rgbs)
        else:
            rgbs, trajs = self.crop(rgbs, trajs, self.crop_size)

        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames (again)
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]

        H,W = rgbs[0].shape[:2]

        visibs[trajs[:, :, 0] > W-1] = False
        visibs[trajs[:, :, 0] < 0] = False
        visibs[trajs[:, :, 1] > H-1] = False
        visibs[trajs[:, :, 1] < 0] = False
        
        # ensure no crazy values
        all_valid = np.nonzero(np.sum(np.sum(np.abs(trajs), axis=-1)<100000, axis=0)==self.seq_len)[0]
        trajs = trajs[:,all_valid]
        visibs = visibs[:,all_valid]
        
        if self.only_first:
            vis_ok = np.nonzero(visibs[0]==1)[0]
            trajs = trajs[:,vis_ok]
            visibs = visibs[:,vis_ok]

        N = trajs.shape[1]
        if N < self.traj_per_sample:
            print('dyn: N after aug', N)
            return None, False

        trajs = torch.from_numpy(trajs)
        visibs = torch.from_numpy(visibs)
        
        rgbs = np.stack(rgbs, 0)
        rgbs = torch.from_numpy(rgbs).reshape(T, H, W, 3).permute(0, 3, 1, 2).float()

        # discard tracks that go far OOB
        crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1) < max(H,W)*2,
            dim=0,
        )
        trajs = trajs[:, close_pts_inds]
        visibs = visibs[:, close_pts_inds]
        
        visible_pts_inds = (visibs[0]).nonzero(as_tuple=False)[:, 0]
        point_inds = torch.randperm(len(visible_pts_inds))[:self.traj_per_sample*self.traj_max_factor]
        if len(point_inds) < self.traj_per_sample:
            # print('not enough trajs')
            return None, False

        visible_inds_sampled = visible_pts_inds[point_inds]
        trajs = trajs[:, visible_inds_sampled].float()
        visibs = visibs[:, visible_inds_sampled]
        valids = torch.ones_like(visibs)

        trajs = trajs[:, :self.traj_per_sample*self.traj_max_factor]
        visibs = visibs[:, :self.traj_per_sample*self.traj_max_factor]
        valids = valids[:, :self.traj_per_sample*self.traj_max_factor]
        
        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            dname=self.dname,
        )
        return sample, True
