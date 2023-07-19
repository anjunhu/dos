import faiss
import torch

from . import utils
from .losses import rotation_loss
from .networks.misc import Encoder32
from .networks.vit import ViTEncoder
from .utils import dino_features_to_image


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        raise NotImplementedError()

    def get_metrics_dict(self, model_outputs, batch):
        raise NotImplementedError()

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        # TODO: check what nerfstudio does with the metrics_dict
        raise NotImplementedError()

    def get_visuals_dict(self, model_outputs, batch):
        raise NotImplementedError()


class CameraRegressor(BaseModel):
    def __init__(
        self,
        encoder=None,
        random_mask_occluder=None,
        rotation_loss_weight=1.0,
        translation_loss_weight=1.0,
        test_obj_path="data_generation/examples/data/horse_009_arabian_galgoPosesV1.obj",
        dino_feat_pca_path="/work/tomj/dove/dino/horses-12c-4s-5k_rnd-cos-gt_mask-pca16-2-pad2-nfix/pca.faiss",  # TODO: move to config
    ):
        super().__init__()
        if encoder is None:
            self.encoder = ViTEncoder()
        else:
            self.encoder = encoder
        # TODO: is random_mask_occluder the best place to put this?
        self.random_mask_occluder = random_mask_occluder
        self.rotation_loss_weight = rotation_loss_weight
        self.translation_loss_weight = translation_loss_weight
        self.test_obj_path = test_obj_path
        self.dino_feat_pca_path = dino_feat_pca_path

        pose_cout = 7  # 4 for rotation, 3 for translation
        if "vits" in self.encoder.model_type:
            dino_feat_dim = 384
        elif "vitb" in self.encoder.model_type:
            dino_feat_dim = 768
        else:
            raise NotImplementedError()
        self.netPose = Encoder32(
            cin=dino_feat_dim, cout=pose_cout, nf=256, activation=None
        )

        # TODO: move somewhere else
        self.dino_pca_mat = faiss.read_VectorTransform(self.dino_feat_pca_path)

    def forward_pose(self, patch_key):
        pose = self.netPose(patch_key)  # Shape: (B, latent_dim)
        # rotation as a quaternion, translation as a vector
        rotation = pose[:, :4]
        # normalize the quaternion
        rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)
        # translation
        translation = pose[:, 4:]
        return rotation, translation

    def forward(self, batch):
        images = batch["image"]
        masks = batch["mask"]
        models_outputs = {}

        # TODO: is this the best place to put this? should there be another wrapper class for the model that does these things?
        if self.random_mask_occluder is not None:
            # randomly occlude the masks
            masks = self.random_mask_occluder(masks)
            models_outputs.update({"masks_randomly_occluded": masks})

        images = images * masks
        patch_key = self.encoder(images)
        # patch_key torch.Size([B, 384, 32, 32])
        # resize to 32 (for DINO v2) TODO: find a better way to do this
        if patch_key.shape[-1] != 32:
            assert patch_key.shape[-1] == patch_key.shape[-2]
            patch_key = torch.nn.functional.interpolate(
                patch_key, size=(32, 32), mode="bilinear", align_corners=False
            )
        rotation, translation = self.forward_pose(patch_key)
        # TODO: see if we really need to output the patch_key_dino
        patch_key_dino = patch_key.permute(0, 2, 3, 1)
        patch_key_dino = patch_key_dino.reshape(
            patch_key_dino.shape[0], 1, -1, patch_key_dino.shape[-1]
        )
        # patch_key_dino torch.Size([B, 1, 1024, 384])
        models_outputs.update({"patch_key_dino": patch_key_dino})
        models_outputs.update(
            {
                "rotation": rotation,
                "translation": translation,
            }
        )
        return models_outputs

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        rotation = model_outputs["rotation"]
        translation = model_outputs["translation"]
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
        loss_dict = {"loss": loss}
        return loss_dict

    def get_metrics_dict(self, model_outputs, batch):
        return {}

    def get_visuals_dict(self, model_outputs, batch, num_visuals=1):
        # TODO: should return tensor or PIL image?

        def _get_visuals_dict(input_dict, names):
            return {
                name: utils.tensor_to_image(input_dict[name][:num_visuals])
                for name in names
                if name in input_dict
            }

        visuals_dict = {}

        batch_visuals_names = ["image", "mask"]
        visuals_dict.update(_get_visuals_dict(batch, batch_visuals_names))

        model_outputs_visuals_names = ["masks_randomly_occluded"]
        visuals_dict.update(
            _get_visuals_dict(model_outputs, model_outputs_visuals_names)
        )

        patch_key_dino = model_outputs["patch_key_dino"][:num_visuals]
        dino_image = dino_features_to_image(patch_key_dino, self.dino_pca_mat)
        patch_key_dino_img = utils.tensor_to_image(dino_image)
        visuals_dict.update({"patch_key_dino": patch_key_dino_img})

        # TODO: cache the mesh_renderer so it doesn't load the obj file every time
        mesh_renderer = utils.MeshRenderer(
            obj_path=self.test_obj_path, device=model_outputs["rotation"].device
        )
        # render the predicted camera pose
        view_matrix = utils.rotation_translation_to_matrix(
            model_outputs["rotation"], model_outputs["translation"]
        )
        rendered_view_pred = mesh_renderer.render(view_matrix[:num_visuals])
        rendered_view_img = utils.tensor_to_image(
            rendered_view_pred[:num_visuals], chw=False
        )
        visuals_dict.update({"rendered_view_pred": rendered_view_img})

        # utils.tensor_to_image(rendered_view_pred[:num_visuals], chw=False),
        if "camera_matrix" in batch:
            # render the ground truth camera pose
            # convert the camera matrix from Blender to OpenGL coordinate system
            camera_matrix = utils.blender_to_opengl(batch["camera_matrix"])
            # convert the camera matrix to view matrix
            view_matrix_gt = camera_matrix.inverse()
            rendered_view_gt = mesh_renderer.render(view_matrix_gt[:num_visuals])
            visuals_dict_img = utils.tensor_to_image(rendered_view_gt, chw=False)
            visuals_dict.update({"rendered_view_gt": visuals_dict_img})

        return visuals_dict
