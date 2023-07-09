"""
Trains a regressor that precits the camera pose from a given image.

1. Extracts DINO features from the image (frozen DINO model)
2. Feeds the features to a regressor (trained from scratch)
3. The regressor outputs the camera pose
4. Loss is computed between the predicted and ground truth camera pose

- Camera pose is represented as a 4x4 matrix from blender. It needs to be converted to OpenGL coordinate system and then to a view matrix. From the view matrix, we can get the camera position and rotation. The rotation is represented as a quaternion.

classes:

- Dataset
- Trainer
- Regressor


Intantiate with hydra, use neptune.ai for logging.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import neptune
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from . import utils


def dino_features_to_image(
    patch_key, dino_pca_mat, h=256, w=256, dino_feature_recon_dim=3
):
    dino_feat_im = patch_key.reshape(-1, patch_key.shape[-1]).cpu().numpy()
    dims = dino_feat_im.shape[:-1]
    dino_feat_im = dino_feat_im / np.linalg.norm(dino_feat_im, axis=1, keepdims=True)
    dino_feat_im = (
        torch.from_numpy(dino_pca_mat.apply_py(dino_feat_im))
        .to(patch_key.device)
        .reshape(*dims, -1)
    )
    dino_feat_im = (
        dino_feat_im.reshape(-1, 32, 32, dino_feat_im.shape[-1])
        .permute(0, 3, 1, 2)
        .clip(-1, 1)
        * 0.5
        + 0.5
    )
    # TODO: is it needed?
    dino_feat_im = torch.nn.functional.interpolate(
        dino_feat_im, size=[h, w], mode="bilinear"
    )[:, :dino_feature_recon_dim]
    return dino_feat_im


def rotation_loss(rotation_pred, rotation_gt):
    """
    rotation_pred and rotation_gt are quaternions, shape: (B, 4)
    """
    return (1 - torch.abs(torch.sum(rotation_pred * rotation_gt, dim=1))).mean()


class InfiniteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        return self.dataset[i % len(self.dataset)]

    def __len__(self):
        return 10**7  # a large number


@dataclass
class Trainer:
    train_dataset: Dataset
    val_dataset: Dataset
    model: nn.Module
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    learning_rate: float = 1e-4
    num_iterations: int = 10000
    num_eval_iterations: int = 100
    num_vis_iterations: int = 50
    num_log_iterations: int = 10
    evaluate_n_visuals: int = 10
    save_checkpoint_freq: int = 200
    device: str = "cuda"
    translation_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    checkpoint_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_name: Optional[str] = None
    shuffle_val: bool = False
    resume: bool = False
    resume_with_latest: bool = False
    test_obj_path: str = (
        "data_generation/examples/data/horse_009_arabian_galgoPosesV1.obj"
    )
    neptune_project: str = "tomj/camera-regressor"
    neptune_api_token: str = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Y2MwOTJmNS01NmRlLTRjNmItYWNkYi05NmZmMGY4NzA4N2MifQ=="
    # TODO: move somewhere else
    dino_feat_pca_path: str = "/work/tomj/dove/dino/horses-12c-4s-5k_rnd-cos-gt_mask-pca16-2-pad2-nfix/pca.faiss"

    def __post_init__(self):
        self.train_dataset = InfiniteDataset(self.train_dataset)
        self.val_dataset = self.val_dataset
        self.model = self.model.to(self.device)
        # TODO: move somewhere else
        self.dino_pca_mat = faiss.read_VectorTransform(self.dino_feat_pca_path)

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if optim:
            warnings.warn("Loading optimizer state is not implemented yet.")

        def get_latest_checkpoint():
            if self.checkpoint_dir is None:
                return None
            checkpoints = sorted(Path(self.checkpoint_dir).glob("*.pth"))
            if len(checkpoints) == 0:
                return None
            return checkpoints[-1]

        latest_checkpoint = get_latest_checkpoint()

        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
        elif self.checkpoint_name is not None:
            checkpoint_path = Path(self.checkpoint_dir) / self.checkpoint_name
        else:
            checkpoint_path = latest_checkpoint

        if self.resume_with_latest and latest_checkpoint is not None:
            checkpoint_path = latest_checkpoint

        if checkpoint_path is None:
            return 0

        self.checkpoint_name = Path(checkpoint_path).name

        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(cp["model"])
        # TODO: implement
        # if optim:
        #     self.model.load_optimizer_state(cp)
        # self.metrics_trace = cp["metrics_trace"]
        total_iter = cp["total_iter"]
        return total_iter

    def save_checkpoint(self, total_iter, optim=True):
        # TODO: update docstring
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"checkpoint-{total_iter:07}.pth"
        checkpoint_path = Path(self.checkpoint_dir) / checkpoint_name
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        # TODO: implement
        # if optim:
        #     optimizer_state = self.model.get_optimizer_state()
        #     state_dict = {**state_dict, **optimizer_state}
        # TODO: implement
        # state_dict["metrics_trace"] = self.metrics_trace
        state_dict["total_iter"] = total_iter
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        # TODO: implement
        # if self.keep_num_checkpoint > 0:
        #     misc.clean_checkpoint(
        #         self.checkpoint_dir, keep_num=self.keep_num_checkpoint
        #     )

    def forward(
        self,
        batch,
        neptune_run=None,
        log_prefix=None,
        log=False,
        visualize=False,
        iteration=None,
        n_visuals=1,
    ):
        # TODO: where it should actually be done
        batch = {k: v.to(self.device) for k, v in batch.items()}
        rotation, translation, forward_aux = self.model(batch)

        if "camera_matrix" in batch:
            # TODO: where it should actually be computed?
            camera_matrix = batch["camera_matrix"]
            # convert the camera matrix from Blender to OpenGL coordinate system
            camera_matrix = utils.blender_to_opengl(camera_matrix)
            # convert the camera matrix to view matrix
            view_matrix_gt = camera_matrix.inverse()
            # convert the view matrix to camera position and rotation (as a quaternion)
            rotation_gt, translation_gt = utils.matrix_to_rotation_translation(
                view_matrix_gt
            )
            # Compute loss between predicted and ground truth camera pose
            loss = self.rotation_loss_weight * rotation_loss(
                rotation, rotation_gt
            ) + self.translation_loss_weight * torch.nn.functional.mse_loss(
                translation, translation_gt
            )
        else:
            loss = 0

        if visualize:
            n_visuals = min(n_visuals, len(batch["image"]))
            neptune_run[log_prefix + "/input_image"].append(
                utils.tensor_to_image(batch["image"][:n_visuals]), step=iteration
            )
            neptune_run[log_prefix + "/input_mask"].append(
                utils.tensor_to_image(batch["mask"][:n_visuals]), step=iteration
            )
            neptune_run[log_prefix + "/input_dino"].append(
                utils.tensor_to_image(
                    dino_features_to_image(
                        forward_aux["patch_key_dino"][:n_visuals], self.dino_pca_mat
                    )
                ),
                step=iteration,
            )
            mesh_renderer = utils.MeshRenderer(
                obj_path=self.test_obj_path, device=self.device
            )
            # render the predicted camera pose
            view_matrix = utils.rotation_translation_to_matrix(rotation, translation)
            rendered_view_pred = mesh_renderer.render(view_matrix[:n_visuals])
            neptune_run[log_prefix + "/rendered_view_pred"].append(
                utils.tensor_to_image(rendered_view_pred[:n_visuals], chw=False),
                step=iteration,
            )
            if "camera_matrix" in batch:
                # render the ground truth camera pose
                rendered_view_gt = mesh_renderer.render(view_matrix_gt[:n_visuals])
                neptune_run[log_prefix + "/rendered_view_gt"].append(
                    utils.tensor_to_image(rendered_view_gt, chw=False),
                    step=iteration,
                )

        return loss

    def evaluate(self, iteration, neptune_run):
        print("Evaluating...")
        val_batch_size = self.val_batch_size or 2 * self.batch_size
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        total_loss = 0
        num_batches = 0

        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if i == 0:
                visualize = True
            else:
                visualize = False
            loss = self.forward(
                batch,
                neptune_run=neptune_run,
                log_prefix="val",
                visualize=visualize,
                iteration=iteration,
                n_visuals=self.evaluate_n_visuals,
            )
            total_loss += loss
            num_batches += 1

        # Compute average validation loss
        avg_loss = total_loss / num_batches
        neptune_run["val/loss"].append(value=avg_loss, step=iteration)

    def train(self, config=None):
        """
        config: dict TODO: config for logging, might be better to move it to elsewhere
        """
        # Initialize Neptune logger
        neptune_run = neptune.init_run(
            project=self.neptune_project,
            api_token=self.neptune_api_token,
        )  # your credentials
        if config is not None:
            neptune_run["parameters"] = config

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        start_total_iter = 0
        # resume from checkpoint
        if self.resume:
            start_total_iter = self.load_checkpoint(optim=True)

        self.model.train()

        # Training loop
        for run_iteration, batch in enumerate(train_loader):
            iteration = start_total_iter + run_iteration

            if iteration % self.save_checkpoint_freq == 0:
                self.save_checkpoint(iteration, optim=True)

            if iteration > self.num_iterations:
                break

            visualize = iteration % self.num_vis_iterations == 0

            loss = self.forward(
                batch,
                neptune_run=neptune_run,
                log_prefix="train",
                visualize=visualize,
                iteration=iteration,
            )
            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"iter: {iteration}, loss: {loss.item():.4f}")
            if iteration % self.num_log_iterations == 0:
                neptune_run["train/loss"].append(value=loss.item(), step=iteration)

            # Evaluate the model
            if iteration % self.num_eval_iterations == 0:
                self.model.eval()
                self.evaluate(iteration, neptune_run)
                self.model.train()

        # Close Neptune logger
        neptune_run.stop()
