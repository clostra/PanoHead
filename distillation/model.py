from pathlib import Path
import cv2
import numpy as np
import pytorch_lightning as pl
import scipy
from scipy.spatial.transform import Rotation as R
import torch
import sys

from distillation.utils import get_lmks, read_mp4

sys.path.insert(0, "deps/neural-head-avatars")
from nha.util.general import dict_2_device, stack_dicts
from nha.models.nha_optimizer import NHAOptimizer


def slice_array(arr, idx):
    if type(arr) is torch.Tensor:
        return arr[idx]
    elif type(arr) is list:
        return [slice_array(el, idx) for el in arr]
    else:
        raise ValueError(f"Unsupported type {type(arr)}")


def slice_batch(batch, idx):
    return {k: slice_array(v, idx) for k, v in batch.items()}


class NHAStaticTrainer(NHAOptimizer):
    def __init__(self, body_part_weights, pti_out_path, **kwargs):
        super().__init__(0, body_part_weights, **kwargs)

        neutral = self._flame.get_neutral_joint_rotations()
        self._neck_pose.data[0] = neutral["neck"].detach().clone()
        self._jaw_pose.data[0] = neutral["jaw"].detach().clone()
        neutral_eyes_pose = torch.cat([neutral["eyes"], neutral["eyes"]], 0)
        self._eyes_pose.data[0] = neutral_eyes_pose

        self.pti_out_path = Path(pti_out_path)
        self.initialize_rt_from_pti()

    def initialize_rt_from_pti(self):
        # Load Panohead render data
        post_c_path = self.pti_out_path / "PTI_render" / "post_c.npy"
        post_c = np.load(str(post_c_path))
        self.c2w = post_c[:, :16].reshape(-1, 4, 4)
        self.K = post_c[:, 16:].reshape(-1, 3, 3)
        post_mp4_path = self.pti_out_path / "PTI_render" / "post.mp4"
        self.post_mp4 = np.stack(read_mp4(post_mp4_path))

        # Get front frames
        angle = np.arctan2(self.c2w[:, 2, 3], self.c2w[:, 0, 3]) / np.pi * 180
        front_idx = np.arange(len(angle))[np.abs(angle - angle[0]) <= 45]

        # Get landmarks
        i0, i1 = front_idx[0], front_idx[30]
        with torch.no_grad():
            lmks1 = get_lmks(self.post_mp4[i0])
            lmks2 = get_lmks(self.post_mp4[i1])

        # Triangulate
        proj1 = self.K[i0] @ np.linalg.inv(self.c2w[i0])[:3] * np.array([512, 512, 1])[:, None]
        proj2 = self.K[i1] @ np.linalg.inv(self.c2w[i1])[:3] * np.array([512, 512, 1])[:, None]
        lmk3d_gt = cv2.triangulatePoints(proj1, proj2, lmks1[:2], lmks2[:2])

        lmk3d_gt = lmk3d_gt[:3] / lmk3d_gt[3:]
        lmk3d_gt = lmk3d_gt.T

        with torch.no_grad():
            flame_results = self._flame(
                self._shape,
                self._expr,
                torch.zeros(1, 3),
                self._neck_pose,
                self._jaw_pose,
                self._eyes_pose,
                zero_centered=True,
                use_rotation_limits=True,
                return_landmarks="static",
                return_mouth_conditioning=False,
            )
            lmks3d = flame_results["landmarks"][0, :68].detach()
        lmks3d = lmks3d.cpu().numpy()

        gt_norm = (lmk3d_gt.std(0).mean(), lmk3d_gt.mean(0))
        flame_norm = (lmks3d.std(0).mean(), lmks3d.mean(0))

        # Find best rotation
        mat, _ = scipy.linalg.orthogonal_procrustes(
            (lmks3d - flame_norm[1]) / flame_norm[0],
            (lmk3d_gt - gt_norm[1]) / gt_norm[0],
        )

        mat = mat.T

        # !!!
        self._log_scale_resid.data = torch.log(torch.tensor(gt_norm[0] / flame_norm[0]))

        mat_rotvec = R.from_matrix(mat)
        # flip_xy = R.from_rotvec([0, 0, np.pi])
        # mat_rotvec = flip_xy * mat_rotvec
        mat_rotvec = mat_rotvec.as_rotvec()
        t = gt_norm[1] - mat @ flame_norm[1] * gt_norm[0] / flame_norm[0]

        # !!!
        self._rotation.data[0] = torch.from_numpy(mat_rotvec).to(self._log_scale_resid)
        self._translation.data[0] = torch.from_numpy(t).to(self._log_scale_resid)

    def _get_current_optimizer(self, epoch=None):
        (
            flame_optim,
            offset_optim,
            tex_optim,
            joint_flame_optim,
            off_resid_optim,
            all_resid_optim,
        ) = self.optimizers()
        return [flame_optim]

    def flame_step(self, batch, stage="train"):
        if stage == "train":
            optim = self._get_current_optimizer()
            self.toggle_optimizer(optim)  # automatically toggle right requires_grads

        flame_params_offsets = self._create_flame_param_batch(batch)
        offsets_verts, pred_lmks = self._forward_flame(flame_params_offsets)

        idx_annotated = batch["is_annotated"]
        batch_annotated = slice_batch(batch, idx_annotated)

        loss_dict = {}

        loss_dict["lmk_loss"] = self._compute_lmk_loss(batch_annotated, pred_lmks[idx_annotated])
        loss_dict["silh_loss"] = self._compute_silhouette_loss(batch, offsets_verts)

        loss = sum(loss_dict.values())

        if stage == "train":
            self.manual_backward(loss)

            for opt in optim:
                opt.step()

            for opt in optim:
                opt.zero_grad(set_to_none=True)  # saving gpu memoryk

            self.untoggle_optimizer()

        return loss, loss_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.is_train = True
        self.prepare_batch(batch)
        with torch.set_grad_enabled(True):
            loss, log_dict = self.flame_step(batch, stage="train")
        self.log_dict(log_dict)
        self.is_train = False
        return loss

    def validation_step(self, batch, batch_idx):
        self.prepare_batch(batch)
        batch["is_annotated"] = torch.tensor([True] * len(batch["rgb"]))
        with torch.no_grad():
            loss, log_dict = self.flame_step(batch, stage="val")
        # self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        log_images = self.current_epoch % self.hparams["image_log_period"] == 0
        if batch_idx == 0 and log_images:
            dataset = self.trainer.val_dataloaders[0].dataset
            interesting_samples = [
                dataset[i] for i in np.linspace(0, len(dataset), 4).astype(int)[1:3]
            ]
            vis_batch = dict_2_device(stack_dicts(*interesting_samples), self.device)
            vis_batch = self.prepare_batch(vis_batch)
            self._visualize_head(vis_batch, max_samples=2, title="val")

    @staticmethod
    def add_argparse_args(parser):
        parser = super(NHAStaticTrainer, NHAStaticTrainer).add_argparse_args(parser)
        parser.add_argument("--pti_out_path", type=str, required=True)
        return parser
