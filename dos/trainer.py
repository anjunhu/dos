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
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import neptune

from .utils import utils
import ipdb
import matplotlib.pyplot as plt


class InfiniteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        # return self.dataset[i % len(self.dataset)]
        return self.dataset[i % len(self.dataset)]

    def __len__(self):
        return 10**3  # 10**7 a large number


@dataclass
class Trainer:
    train_dataset: Dataset
    model: nn.Module
    val_dataset: Optional[Dataset] = None
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    learning_rate: float = 1e-4
    num_iterations: int = 10000
    num_eval_iterations: int = 100
    num_vis_iterations: int = 50
    num_log_iterations: int = 10
    evaluate_num_visuals: int = 12 # changed - before it was 10
    save_checkpoint_freq: int = 200
    device: str = "cuda"
    translation_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    experiment_name: Optional[str] = None
    checkpoint_root_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_name: Optional[str] = None
    modules_to_load: Optional[list] = None
    shuffle_val: bool = False
    test_only: bool = False
    resume: bool = False
    resume_with_latest: bool = False
    neptune_project: Optional[str] = None
    neptune_api_token: Optional[str] = None

    def __post_init__(self):
        self.val_dataset = self.val_dataset or self.train_dataset

        self.train_dataset_collate_fn = self.train_dataset.collate_fn
        self.val_dataset_collate_fn = self.val_dataset.collate_fn

        self.train_dataset = InfiniteDataset(self.train_dataset)
        self.model = self.model.to(self.device)

        # either both experiment_name and checkpoint_root_dir or only checkpoint_dir should be specified
        if self.checkpoint_dir is not None:
            assert (
                self.experiment_name is None and self.checkpoint_root_dir is None
            ), "Either both experiment_name and checkpoint_root_dir or only checkpoint_dir should be specified."
            self.checkpoint_dir = Path(self.checkpoint_dir)
            self.experiment_name = self.checkpoint_dir.name
            self.checkpoint_root_dir = self.checkpoint_dir.parent
        else:
            assert (
                self.experiment_name is not None
                and self.checkpoint_root_dir is not None
            ), "Either both experiment_name and checkpoint_root_dir or only checkpoint_dir should be specified."
            self.checkpoint_root_dir = Path(self.checkpoint_root_dir)
            self.checkpoint_dir = self.checkpoint_root_dir / self.experiment_name

    def load_module(self, name, path):
        """Load a module from a checkpoint."""
        module = utils.rgetattr(self.model, name)
        module.load_state_dict(torch.load(path))

    def load_modules(self, modules_to_load):
        """Load modules from checkpoints."""
        for module in modules_to_load:
            self.load_module(module["name"], module["path"])

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
        num_visuals=12,
    ):
        
        # TODO: where it should actually be done
        # non_blocking=True is needed for async data loading
        batch = utils.safe_batch_to_device(batch, self.device, non_blocking=True)

        # rotation, translation, forward_aux = self.model(batch)
        
        model_outputs = self.model(batch)
        
        # metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        # loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        loss_dict = self.model.get_loss_dict(model_outputs, batch)
        

        if visualize:
            
            num_visuals = min(num_visuals, len(batch["image"]))
            
            """
            batch.keys() are dict_keys(['mesh', 'image', 'mask', 'pose', 'texture_features', 'name'])
            model_outputs.keys() are dict_keys(['image_pred', 'mask_pred', 'albedo', 'shading', 'rendered_kps', 'rendered_image_with_kps', 'target_image_with_kps', 'target_corres_kps'])
            visuals_dict.keys() are dict_keys(['image', 'image_pred'])
            """
            
            # visuals_dict = self.model.get_visuals_dict(
            #     model_outputs, batch, num_visuals=num_visuals
            # )
            
            visuals_dict = {key: model_outputs[key] for key in ['target_image_with_kps', 'rendered_image_with_kps']}
            
            for index, (visual_name, visual) in enumerate(visuals_dict.items()):

                visual.save(f'{visual_name}_{index}.png', bbox_inches='tight')                
                neptune_run[log_prefix + "/" + visual_name].append(visual, step=iteration)
                
        return loss_dict["loss"]

    def evaluate(self, iteration, neptune_run):
        print("Evaluating...")
        
        print('len of self.val_dataset', len(self.val_dataset))
        
        val_batch_size = self.batch_size
        # val_batch_size = self.val_batch_size or 2 * self.batch_size
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset_collate_fn,
        )

        total_loss = 0
        num_batches = 0
        
        
        # len(val_loader) is 13
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
                num_visuals=self.evaluate_num_visuals,
            )
            total_loss += loss
            print('loss', loss)
            num_batches += 1

        # Compute average validation loss
        avg_loss = total_loss / num_batches
        neptune_run["val/loss"].append(value=avg_loss, step=iteration) # ORIGINAL CODE
        
        #with neptune.create_experiment() as neptune_run:                                                   # ADDED
        #    neptune_run.set_property('parameter-3', ["val/loss"].append(value=avg_loss, step=iteration))   # ADDED

    def train(self, config=None):
        print("Training...")
        """
        config: dict TODO: config for logging, might be better to move it to elsewhere
        """
        # set random seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize Neptune logger                  
        neptune_run = neptune.init_run(
            project=self.neptune_project,
            api_token=self.neptune_api_token,
        )  # your credentials
            
        # COMMENTED
        # if config is not None:
        #     neptune_run["parameters"] = config
            
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset_collate_fn,
        )

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        start_total_iter = 0
        # resume from checkpoint
        if self.resume:
            start_total_iter = self.load_checkpoint(optim=True)

        # load individual modules
        if self.modules_to_load is not None:
            self.load_modules(self.modules_to_load)

        # TODO: consider better way to do this
        if self.test_only:
            self.model.eval()
            iteration = start_total_iter
            self.evaluate(iteration, neptune_run)
            self.model.train()
            return

        self.model.train()
        

        # Training loop
        for run_iteration, batch in tqdm(enumerate(train_loader)):
            iteration = start_total_iter + run_iteration

            if iteration % self.save_checkpoint_freq == 0 and iteration > 0:
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
            if loss.requires_grad:
                loss.backward()
            else:
                warnings.warn("loss.backward() is not called.")
            optimizer.step()

            print(f"iter: {iteration}, loss: {loss.item():.4f}")
            if iteration % self.num_log_iterations == 0:   # ORIGINAL CODE
                neptune_run["train/loss"].append(value=loss.item(), step=iteration) # ORIGINAL CODE
                
                #with neptune.create_experiment() as neptune_run:  # ADDED
                #    neptune_run.set_property('parameter-4', ["train/loss"].append(value=loss.item(), step=iteration)) # ADDED

            # Evaluate the model
            if iteration % self.num_eval_iterations == 0:
                self.model.eval()
                self.evaluate(iteration, neptune_run)
                self.model.train()

        # Close Neptune logger
        neptune_run.stop()
