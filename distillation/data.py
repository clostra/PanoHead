from pathlib import Path
import numpy as np
import torch
import sys
from distillation.utils import read_mp4
from torchvision.transforms.functional import resize

sys.path.insert(0, "deps/neural-head-avatars")
from nha.data.real import RealDataset, RealDataModule


class Full1Partial2Sampler:
    def __init__(self, num_full, num_partial, batch_size, drop_last=False):
        self.partial_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(range(num_full, num_full + num_partial)),
            batch_size,
            drop_last=drop_last,
        )
        self.num_full = num_full
        self.batch_size = num_full + batch_size
        self.drop_last = drop_last

    def __len__(self):
        return len(self.partial_sampler)

    def __iter__(self):
        for sample in self.partial_sampler:
            yield list(range(self.num_full)) + sample


class PanoheadPostDataset(RealDataset):
    def __init__(self, path, pti_out_path, mode="train", **kwargs):
        super().__init__(path, **kwargs)
        self.mode = mode
        if mode == "val":
            return
        self.pti_out_path = Path(pti_out_path)
        pti_render = self.pti_out_path / "PTI_render"
        post_c_path = pti_render / "post_c.npy"
        post_mp4_path = pti_render / "post.mp4"
        post_mask_path = pti_render / "post_mask.npy"
        post_c = np.load(str(post_c_path))
        self.post_mp4 = torch.from_numpy(np.stack(read_mp4(post_mp4_path)))
        self.post_mask = torch.from_numpy(np.load(str(post_mask_path))).clamp(0, 1)
        self.pti_c2w = torch.from_numpy(post_c[:, :16].reshape(-1, 4, 4))
        self.pti_K = torch.from_numpy(post_c[:, 16:].reshape(-1, 3, 3))

    def __len__(self):
        if self.mode == "val":
            return super().__len__()
        if self.mode == "train":
            return super().__len__() + len(self.post_mp4)

    def __getitem__(self, idx):
        if idx < super().__len__():
            sample = super().__getitem__(idx)
            sample["is_annotated"] = True
            sample["frame"] = 0
            sample.pop("subject")
            sample.pop("valid_eyes")
            return sample
        else:
            idx -= super().__len__()
            H, W = self.post_mp4.shape[1:3]
            mask = self.post_mask[idx]
            mask = resize(mask, (H, W), antialias=True)
            mask = mask > 0.5

            # print(H, W, flush=True)
            return {
                "cam_intrinsic": self.pti_K[idx]
                * torch.tensor([W, H, 1]).to(self.pti_K)[..., None],
                "cam_extrinsic": torch.linalg.inv(self.pti_c2w[idx])[:3],
                "rgb": self.post_mp4[idx].permute(2, 0, 1).float() / 255 * 2 - 1,
                "is_annotated": False,
                "frame": 0,
                # dummies for collate
                "seg": mask,
                "lmk2d": torch.zeros(68, 3),
                "lmk2d_iris": torch.zeros(2, 3),
                "normal": torch.zeros(3, H, W),
                "parsing": torch.zeros(1, H, W, dtype=torch.int32),
                "eye_distance": [0, 0],
            }

    @classmethod
    def add_argparse_args(cls, parser):
        """
        Adds dataset specific parameters to parser
        :param parser:
        :return:
        """
        parser = super(PanoheadPostDataset, cls).add_argparse_args(parser)
        parser.add_argument("--pti_out_path", type=str, required=True)

        return parser


class PanoheadPostDataModule(RealDataModule):
    def __init__(self, pti_out_path, **kwargs):
        super().__init__(**kwargs)
        self.pti_out_path = pti_out_path

    def setup(self, stage=None):
        train_split, val_split = self._read_splits(self._split_config)

        self._train_set = PanoheadPostDataset(
            self._path,
            self.pti_out_path,
            mode="train",
            frame_filter=train_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

        self._val_set = PanoheadPostDataset(
            self._path,
            self.pti_out_path,
            mode="val",
            frame_filter=val_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

    @classmethod
    def add_argparse_args(cls, parser):
        parser = super(PanoheadPostDataModule, cls).add_argparse_args(parser)
        return parser

    def train_dataloader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(
            self._train_set,
            num_workers=self._workers,
            batch_sampler=Full1Partial2Sampler(
                len(self._train_set.frame_list),
                len(self._train_set.post_mp4),
                batch_size,
            ),
        )
