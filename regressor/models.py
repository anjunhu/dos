import torch
from .networks.misc import Encoder32
from .networks.vit import ViTEncoder


class CameraRegressor(torch.nn.Module):
    def __init__(
        self,
        encoder=ViTEncoder(),
        random_mask_occluder=None,  # TODO: is this the best place to put this?
    ):
        super().__init__()
        self.random_mask_occluder = random_mask_occluder
        self.netEncoder = encoder
        pose_cout = 7  # 4 for rotation, 3 for translation
        if "vits" in self.netEncoder.model_type:
            dino_feat_dim = 384
        elif "vitb" in self.netEncoder.model_type:
            dino_feat_dim = 768
        else:
            raise NotImplementedError()
        self.netPose = Encoder32(
            cin=dino_feat_dim, cout=pose_cout, nf=256, activation=None
        )

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
        aux_out = {}
        # TODO: is this the best place to put this? should there be another wrapper class for the model that does these things?
        if self.random_mask_occluder is not None:
            # randomly occlude the masks
            masks = self.random_mask_occluder(masks)
            aux_out.update({"masks_randomly_occluded": masks})
        images = images * masks
        patch_key = self.netEncoder(images)
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
        aux_out.update({"patch_key_dino": patch_key_dino})
        return rotation, translation, aux_out
