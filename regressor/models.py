import torch
from . import networks


class CameraRegressor(torch.nn.Module):
    # TODO: abstract away the encoder and its parameters
    def __init__(
        self,
        vit_name="dino_vits8",
        vit_final_layer_type="conv",
        encoder_latent_dim=256,
        encoder_pretrained=True,
        encoder_frozen=True,
        in_image_size=256,
    ):
        super().__init__()
        self.netEncoder = networks.ViTEncoder(
            cout=encoder_latent_dim,
            which_vit=vit_name,
            pretrained=encoder_pretrained,
            frozen=encoder_frozen,
            in_size=in_image_size,
            final_layer_type=vit_final_layer_type,
        )
        pose_cout = 7  # 4 for rotation, 3 for translation
        if vit_name == "dino_vits8":
            dino_feat_dim = 384
        elif vit_name == "dino_vitb8":
            dino_feat_dim = 768
        self.netPose = networks.Encoder32(
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

    def forward_encoder(self, images):
        images_in = images * 2 - 1  # Rescale to (-1, 1)
        feat_out, feat_key, patch_out, patch_key, patch_key_dino = self.netEncoder(
            images_in, return_patches=True
        )
        return feat_out, feat_key, patch_out, patch_key, patch_key_dino

    def forward(self, batch):
        images = batch["image"]
        masks = batch["mask"]
        images = images * masks
        (
            feat_out,
            feat_key,
            patch_out,
            patch_key,
            patch_key_dino,
        ) = self.forward_encoder(images)

        rotation, translation = self.forward_pose(patch_key)
        return rotation, translation
