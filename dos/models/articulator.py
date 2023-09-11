import torch

from ..components.skinning.bones_estimation import BonesEstimator
from ..components.skinning.skinning import mesh_skinning
from ..modules.renderer import Renderer
from ..nvdiffrec.render.mesh import load_mesh
from ..predictors.texture import TexturePredictor
from ..utils import geometry as geometry_utils
from ..utils import mesh as mesh_utils
from ..utils import visuals as visuals_utils
from .base import BaseModel


class Articulator(BaseModel):
    """
    Articulator predicts instance shape parameters (instance shape) - optimisation based - predictor takes only id as input
    """

    # TODO: set default values for the parameters (dataclasses have a nice way of doing it
    #   but it's not compatible with torch.nn.Module)
    def __init__(
        self,
        encoder=None,
        enable_texture_predictor=True,
        texture_predictor=None,
        bones_predictor=None,
        articulation_predictor=None,
        renderer=None,
        shape_template_path=None,
    ):
        super().__init__()
        self.encoder = encoder  # encoder TODO: should be part of the predictor?
        self.enable_texture_predictor = enable_texture_predictor
        self.texture_predictor = (
            texture_predictor if texture_predictor is not None else TexturePredictor()
        )
        self.bones_predictor = (
            bones_predictor if bones_predictor is not None else BonesEstimator()
        )
        # TODO: implement articulation predictor
        self.articulation_predictor = articulation_predictor
        self.renderer = renderer if renderer is not None else Renderer()

        if shape_template_path is not None:
            self.shape_template = self._load_shape_template(shape_template_path)
        else:
            self.shape_template = None

    def _load_shape_template(self, shape_template_path):
        return load_mesh(shape_template_path)

    def compute_correspondences(
        self, mesh, pose, renderer, bones, source_image, target_image
    ):
        # TODO: implement keypoint loss (in a separate function)
        # 1. Extract features from the rendered image and the target image
        # 2. Sample source keypoints
        #  - sample keypoints along the bones
        #  - find the closest visible point on the mesh in 3D (the visibility is done in 2D)
        # 3. Extract features from the source keypoints
        # 4. Find corresponding target keypoints (TODO: some additional tricks e.g. optimal transport etc.)

        # get visible vertices
        mvp, _, _ = geometry_utils.get_camera_extrinsics_and_mvp_from_pose(
            pose,
            renderer.fov,
            renderer.znear,
            renderer.zfar,
            renderer.cam_pos_z_offset,
        )
        visible_vertices = mesh_utils.get_visible_vertices(
            mesh, mvp, renderer.resolution
        )
        # TODO: continue here

        # project vertices/keypoints example
        projected_vertices = geometry_utils.project_points(mesh.v_pos, mvp)

        output_dict = {}
        return output_dict

    def forward(self, batch):
        batch_size = batch["image"].shape[0]
        if self.shape_template is not None:
            mesh = self.shape_template.extend(batch_size)
        else:
            mesh = batch["mesh"]  # rest pose

        # estimate bones
        bones_predictor_outputs = self.bones_predictor(mesh.v_pos)

        # TODO: predict articulation (just a look up table by the sample name for now as we do optimisation only)
        # bones_rotations = self.predictor(batch)
        batch_size = bones_predictor_outputs["bones_pred"].shape[0]
        num_bones = bones_predictor_outputs["bones_pred"].shape[1]
        bones_rotations = torch.zeros(
            batch_size, num_bones, 3, device=mesh.v_pos.device
        )
        # pertrub the bones rotations (only to test the implementation)
        bones_rotations = bones_rotations + torch.randn_like(bones_rotations) * 0.1

        # apply articulation to mesh
        articulated_mesh, _ = mesh_skinning(
            mesh,
            bones_predictor_outputs["bones_pred"],
            bones_predictor_outputs["kinematic_chain"],
            bones_rotations,
            bones_predictor_outputs["skinnig_weights"],
            output_posed_bones=False,
        )

        # render mesh
        if "texture_features" in batch:
            texture_features = batch["texture_features"]
        else:
            texture_features = None

        if self.enable_texture_predictor:
            material = self.texture_predictor
        else:
            # if texture predictor is not enabled, use the loaded material from the mesh
            material = mesh.material

        # if pose not provided, compute it from the camera matrix
        if "pose" not in batch:
            pose = geometry_utils.blender_camera_matrix_to_magicpony_pose(
                batch["camera_matrix"]
            )

        if "background" in batch:
            background = batch["background"]
        else:
            background = None

        renderer_outputs = self.renderer(
            articulated_mesh,
            material=material,
            pose=pose,
            im_features=texture_features,
            background=background,
        )

        # compute_correspondences for keypoint loss
        correspondences_dict = self.compute_correspondences(
            articulated_mesh,
            pose,
            self.renderer,
            bones_predictor_outputs["bones_pred"],
            renderer_outputs["image_pred"],
            batch["image"],
        )

        # esamble outputs
        outputs = {}
        # TODO: probaly rename the ouputs of the renderer
        outputs.update(renderer_outputs)
        outputs.update(correspondences_dict)

        return outputs

    def get_metrics_dict(self, model_outputs, batch):
        return {}

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        """ """
        # TODO: implement keypoint loss (in a separate function)
        # 5. Compute the loss between the source and target keypoints
        return {"loss": torch.tensor(0.0)}

    def get_visuals_dict(self, model_outputs, batch, num_visuals=1):
        def _get_visuals_dict(input_dict, names):
            return visuals_utils.get_visuals_dict(input_dict, names, num_visuals)

        visuals_dict = {}

        batch_visuals_names = ["image"]
        visuals_dict.update(_get_visuals_dict(batch, batch_visuals_names))

        model_outputs_visuals_names = ["image_pred"]
        visuals_dict.update(
            _get_visuals_dict(model_outputs, model_outputs_visuals_names)
        )

        # TODO: render also rest pose

        return visuals_dict
