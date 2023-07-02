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
from dataclasses import dataclass

import neptune
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional
from PIL import Image

from . import utils


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
    device: str = "cuda"
    translation_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    test_obj_path: str = (
        "data_generation/examples/data/horse_009_arabian_galgoPosesV1.obj"
    )
    neptune_project: str = "tomj/camera-regressor"
    neptune_api_token: str = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Y2MwOTJmNS01NmRlLTRjNmItYWNkYi05NmZmMGY4NzA4N2MifQ=="

    def __post_init__(self):
        self.train_dataset = InfiniteDataset(self.train_dataset)
        self.val_dataset = self.val_dataset
        self.model = self.model.to(self.device)

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
        rotation, translation = self.model(batch)

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
            for i in range(min(n_visuals, len(batch["image"]))):
                neptune_run[log_prefix + "/input_image"].append(
                    utils.tensor_to_image(batch["image"][i]), step=iteration
                )
                mesh_renderer = utils.MeshRenderer(
                    obj_path=self.test_obj_path, device=self.device
                )
                # render the predicted camera pose
                view_matrix = utils.rotation_translation_to_matrix(
                    rotation, translation
                )
                rendered_view_pred = mesh_renderer.render(view_matrix[i])
                neptune_run[log_prefix + "/rendered_view_pred"].append(
                    utils.tensor_to_image(rendered_view_pred[0], no_permute=True),
                    step=iteration,
                )
                if "camera_matrix" in batch:
                    # render the ground truth camera pose
                    rendered_view_gt = mesh_renderer.render(view_matrix_gt[i])
                    neptune_run[log_prefix + "/rendered_view_gt"].append(
                        utils.tensor_to_image(rendered_view_gt[0], no_permute=True),
                        step=iteration,
                    )

        return loss

    def evaluate(self, iteration, neptune_run):
        print("Evaluating...")
        val_batch_size = self.val_batch_size or 2 * self.batch_size
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        total_loss = 0
        num_batches = 0

        for i, batch in tqdm(enumerate(val_loader)):
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
                n_visuals=10,
            )
            total_loss += loss
            num_batches += 1

        # Compute average validation loss
        avg_loss = total_loss / num_batches
        neptune_run["val/loss"].append(value=avg_loss, step=iteration)

    def train(self):
        # Initialize Neptune logger
        neptune_run = neptune.init_run(
            project=self.neptune_project,
            api_token=self.neptune_api_token,
        )  # your credentials

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        # Training loop
        for iteration, batch in enumerate(train_loader):
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
            if iteration % self.num_eval_iterations == 0 and iteration > 0:
                self.model.eval()
                self.evaluate(iteration, neptune_run)
                self.model.train()

        # Close Neptune logger
        neptune_run.stop()
