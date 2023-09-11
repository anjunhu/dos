from dataclasses import dataclass

import nvdiffrast.torch as dr
import torch

from ..nvdiffrec.render import render
from ..utils.geometry import get_camera_extrinsics_and_mvp_from_pose


@dataclass
class Renderer(object):
    znear: float = 0.1
    zfar: float = 1000.0
    fov: float = 25.0  # MagicPony default
    cam_pos_z_offset: float = 10.0  # MagicPony default
    resolution: tuple = (256, 256)  # MagicPony default

    def __post_init__(self) -> None:
        self.glctx = dr.RasterizeGLContext()

    # TODO: what should be moved to config?
    def __call__(
        self,
        shape,
        material=None,
        # either
        pose=None,
        # or
        mvp=None,
        w2c=None,
        campos=None,
        #
        im_features=None,
        light=None,
        prior_shape=None,
        dino_pred=None,
        render_mode="diffuse",
        two_sided_shading=True,
        spp=1,
        background=None,
    ):
        """
        Either (pose, self.fov, self.cam_pos_z_offset) or (mvp, w2c, campos) should be provided
        """
        resolution = self.resolution
        fov = self.fov
        cam_pos_z_offset = self.cam_pos_z_offset

        # either pose or mvp
        pose_enabled = (
            pose is not None and fov is not None and cam_pos_z_offset is not None
        )
        mvp_enabled = mvp is not None and w2c is not None and campos is not None
        assert pose_enabled != mvp_enabled, "Either pose or mvp should be provided"

        if pose_enabled:
            # TODO: remove cam_pos_z_offset dependency
            mvp, w2c, campos = get_camera_extrinsics_and_mvp_from_pose(
                pose,
                fov,
                znear=self.znear,
                zfar=self.zfar,
                cam_pos_z_offset=cam_pos_z_offset,
            )

        h, w = resolution
        batch_size = len(mvp)
        if background is None:
            background = torch.zeros((batch_size, h, w, 3), device=mvp.device)
        else:
            # expects channels last
            background = background.permute(0, 2, 3, 1)

        frame_rendered = render.render_mesh(
            self.glctx,
            shape,
            mtx_in=mvp,
            w2c=w2c,
            view_pos=campos,
            material=material,
            lgt=light,
            resolution=resolution,
            spp=spp,
            msaa=True,
            background=background,
            bsdf=render_mode,
            feat=im_features,
            prior_mesh=prior_shape,
            two_sided_shading=two_sided_shading,
            dino_pred=dino_pred,
        )
        shaded = frame_rendered["shaded"].permute(0, 3, 1, 2)
        image_pred = shaded[:, :3, :, :]
        mask_pred = shaded[:, 3, :, :]
        albedo = frame_rendered["kd"].permute(0, 3, 1, 2)[:, :3, :, :]
        if "shading" in frame_rendered:
            shading = frame_rendered["shading"].permute(0, 3, 1, 2)[:, :1, :, :]
        else:
            shading = None
        if dino_pred is not None:
            dino_feat_im_pred = frame_rendered["dino_feat_im_pred"]
            dino_feat_im_pred = dino_feat_im_pred.permute(0, 3, 1, 2)[:, :-1]
        else:
            dino_feat_im_pred = None

        output_dict = {
            "image_pred": image_pred,
            "mask_pred": mask_pred,
            "albedo": albedo,
            "shading": shading,
        }
        if dino_feat_im_pred is not None:
            output_dict["dino_feat_im_pred"] = dino_feat_im_pred

        return output_dict
