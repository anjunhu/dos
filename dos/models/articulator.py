import torch
import os

from ..components.skinning.bones_estimation import BonesEstimator
from ..components.skinning.skinning import mesh_skinning
from ..modules.renderer import Renderer
from ..nvdiffrec.render.mesh import load_mesh
from ..predictors.texture import TexturePredictor
from ..predictors.articulation_predictor import ArticulationPredictor
from ..utils import geometry as geometry_utils
from ..utils import mesh as mesh_utils
from ..utils import visuals as visuals_utils
from .base import BaseModel
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import torch.nn.functional as nn_functional

import matplotlib.pyplot as plt
from ..utils import utils
from dos.components.fuse.compute_correspond import compute_correspondences_sd_dino
from dos.utils.utils_correspondence import resize, draw_kp_on_image, draw_lines_on_img


def identify_edge_vertices(mesh_v_pos, distance_threshold=0.05, chunk_size=500):
    num_vertices = mesh_v_pos.size(1)
    neighbor_counts = torch.zeros(mesh_v_pos.size(0), num_vertices, device=mesh_v_pos.device)

    for i in range(0, num_vertices, chunk_size):
        end = min(i + chunk_size, num_vertices)
        chunk = mesh_v_pos[:, i:end, :]
        chunk_exp = chunk.unsqueeze(2)
        mesh_v_pos_exp = mesh_v_pos.unsqueeze(1)
        dists_chunk = ((chunk_exp - mesh_v_pos_exp) ** 2).sum(-1).sqrt()
        neighbor_mask = (dists_chunk < distance_threshold) & (dists_chunk > 0)
        neighbor_counts[:, i:end] += neighbor_mask.sum(dim=2)

    neighbor_threshold = neighbor_counts.median(dim=1).values  # Here I'm using median as a threshold, you can adjust based on your needs
    edge_vertices = neighbor_counts < neighbor_threshold.unsqueeze(1)

    return edge_vertices

def closest_visible_points(bones_midpts, mesh_v_pos, visible_vertices):
    
    # mesh_v_pos shape ([10, 31070, 3])
    # Expand dimensions for broadcasting
    bones_midpts_exp = bones_midpts.unsqueeze(2)  # bones_midpts_exp shape [Batch size, 20, 1, 3]
    mesh_v_pos_exp = mesh_v_pos.unsqueeze(1)  # mesh_v_pos_exp shape [Batch size, 1, 31070, 3]
    
    # Calculating squared distance between each bones_midpt and all mesh vertices
    dists = ((bones_midpts_exp - mesh_v_pos_exp) ** 2).sum(-1)  # dists shape [Batch size, 20, 31070]
    
    del bones_midpts_exp
    del mesh_v_pos_exp
    # Set the distance of occluded vertices to a high value (to ignore them)
    # By adding 1 (or any positive number) to the maximum value already present in dists, we are guaranteeing that this max_val is strictly greater 
    # than any other distance in the tensor. 
    # This makes it a safe value to use for masking, because it will always be larger than any genuine squared distance in the dists tensor.
    max_val = torch.max(dists).item() + 1
    
    # edge_mask = identify_edge_vertices(mesh_v_pos)
    
    # occluded_or_edge_mask = ((1 - visible_vertices).bool() | edge_mask.bool()).unsqueeze(1)   # occluded_mask shape [Batch size, 1, 31070]
    
    occluded_or_edge_mask = (1 - visible_vertices).bool().unsqueeze(1) 
    
    # wherever the occluded_mask is True (i.e., for occluded vertices), the value in dists will be set to this maximum value (max_val). 
    dists.masked_fill_(occluded_or_edge_mask, max_val)
    
    del max_val
    del occluded_or_edge_mask
    
    # Get the index of the minimum distance along the last dimension
    _, closest_idx = dists.min(-1)  # closest_idx shape [Batch size, 20]
    
    del dists
    # Use the index to gather the closest visible points from mesh_v_pos
    # creates a tensor of indices that represent each sample in the batch. It's required for advanced indexing in the next step.
    
    # batch_idx shape is ([10, 1])
    batch_idx = torch.arange(bones_midpts.size(0)).unsqueeze(1).to(closest_idx.device)
    
    bone_idx = torch.arange(bones_midpts.size(1)).unsqueeze(0).to(closest_idx.device)
    
    # closest_points shape ([10, 20, 3])
    closest_points = mesh_v_pos[batch_idx, closest_idx, :]
    
    return closest_points
    

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
        self.articulation_predictor = (articulation_predictor if articulation_predictor else ArticulationPredictor())
        
        self.renderer = renderer if renderer is not None else Renderer()

        if shape_template_path is not None:
            self.shape_template = self._load_shape_template(shape_template_path)
        else:
            self.shape_template = None

    def _load_shape_template(self, shape_template_path):
        return load_mesh(shape_template_path)

    def compute_correspondences(
        self, articulated_mesh, pose, renderer, bones, rendered_image, target_image
    ):
        # 1. Extract features from the rendered image and the target image
        # 2. Sample keypoints from the rendered image
        #  - sample keypoints along the bones
        #  - find the closest visible point on the articulated_mesh in 3D (the visibility is done in 2D)
        # 3. Extract features from the source keypoints
        # 4. Find corresponding target keypoints (TODO: some additional tricks e.g. optimal transport etc.)
        
        target_image_path = f'/users/oishideb/dos/target_images/'

        # get visible vertices
        mvp, _, _ = geometry_utils.get_camera_extrinsics_and_mvp_from_pose(
            pose,
            renderer.fov,
            renderer.znear,
            renderer.zfar,
            renderer.cam_pos_z_offset,
        )
        
        # visible_vertices.shape is torch.Size([2, 31070]) 
        visible_vertices = mesh_utils.get_visible_vertices(                                                  
            articulated_mesh, mvp, renderer.resolution
        )
        
        # project vertices/keypoints example
        # mesh.v_pos shape is torch.Size([2, 31070, 3])
        # mvp.shape is torch.Size([Batch size, 4, 4])
        # projected_vertices.shape is torch.Size([4, 31070, 2]), # mvp is model-view-projection
        # projected_vertices = geometry_utils.project_points(articulated_mesh.v_pos, mvp)  
        
        #projected_vertices = projected_vertices[:, :100, :]
        #kps_img = projected_vertices[:,:,:][0] * rendered_image.shape[2]

        bone_end_pt_1_3D = bones[:, :, 0, :]  # one end of the bone in 3D
        bone_end_pt_2_3D = bones[:, :, 1, :]  # other end of the bone in 3D
        
        bones_in_3D_all_kp40 = torch.cat((bone_end_pt_1_3D, bone_end_pt_2_3D), dim=1)
        bones_2D_proj_all_kp40 = geometry_utils.project_points(bones_in_3D_all_kp40, mvp)
        
        bone_end_pt_1_projected_in_2D = geometry_utils.project_points(bone_end_pt_1_3D, mvp)
        bone_end_pt_2_projected_in_2D = geometry_utils.project_points(bone_end_pt_2_3D, mvp)
        
        bones_midpts_in_3D = (bones[:, :, 0, :] + bones[:, :, 1, :]) / 2.0        # This is in 3D the shape is torch.Size([2, 20, 3])
        bones_midpts_projected_in_2D = geometry_utils.project_points(bones_midpts_in_3D, mvp)
        
        # edge_mask = identify_edge_vertices(articulated_mesh.v_pos)
        
        closest_midpts = closest_visible_points(bones_midpts_in_3D, articulated_mesh.v_pos, visible_vertices)
        ## shape of bones_closest_pts_2D_proj is ([Batch-size, 20, 2])
        bones_closest_midpts_projected_in_2D_all_kp20 = geometry_utils.project_points(closest_midpts, mvp)
        
        bone_end_pt_1 = closest_visible_points(bone_end_pt_1_3D, articulated_mesh.v_pos, visible_vertices)
        bone_end_pt_2 = closest_visible_points(bone_end_pt_2_3D, articulated_mesh.v_pos, visible_vertices)
        
        bone_end_pt_1_in_2D_cls = geometry_utils.project_points(bone_end_pt_1, mvp)
        bone_end_pt_2_in_2D_cls = geometry_utils.project_points(bone_end_pt_2, mvp)
        
        bones_mid_pt_in_2D = (bone_end_pt_1_in_2D_cls + bone_end_pt_2_in_2D_cls) / 2.0 
        
        bones_all = torch.cat((bone_end_pt_1, bone_end_pt_2), dim=1)
        
        bones_all = closest_visible_points(bones_all, articulated_mesh.v_pos, visible_vertices)
        bones_closest_pts_2D_proj_all_kp40 = geometry_utils.project_points(bones_all, mvp)
        
        # Define a list to hold the converted images
        rendered_images_PIL_list = []
        target_images_PIL_list = []
        output_dict = {}
        
        
        # Iterate over the batch dimension and convert each sample
        for index in range(rendered_image.shape[0]):
            
            # rendered_image_PIL is 256*256
            rendered_image_PIL = F.to_pil_image(rendered_image[index])
            print('rendered_image_PIL', rendered_image_PIL)
            
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            #rendered_images_PIL_list.append(rendered_image_PIL)
            #rendered_image_PIL.save(f'{index}_image_gt.png', bbox_inches='tight') 
            
            # target_image_PIL = F.to_pil_image(target_image[index])
            # target_image_PIL = resize(target_image_PIL, target_res = 840, resize=True, to_pil=True)
            # #target_images_PIL_list.append(target_image_PIL)
            
            
            target_image_updated = os.path.join(target_image_path+f'{index}_image_gt_save.png')
            
            target_image_updated = Image.open(target_image_updated).convert('RGB')
            
            midpts_in_2D = False
            if midpts_in_2D:
                kps_img_resolu_mid_kp20 = (bones_mid_pt_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_all_kp40 = (bones_closest_pts_2D_proj_all_kp40[index] + 1) * rendered_image_PIL.size[0]/2
                
                rendered_image_with_kps = draw_kp_on_image(kps_img_resolu_mid_kp20, rendered_image_PIL, color='yellow')                  #[-6:]
                rendered_image_with_kps = draw_kp_on_image(kps_img_resolu_all_kp40, rendered_image_PIL, color='blue')  
                
                kps_img_resolu_bone_1_cls_2D = (bone_end_pt_1_in_2D_cls[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_bone_2_cls_2D = (bone_end_pt_2_in_2D_cls[index] + 1) * rendered_image_PIL.size[0]/2
        
                x1, y1 = kps_img_resolu_bone_1_cls_2D[:,0], kps_img_resolu_bone_1_cls_2D[:,1]
                x2, y2 = kps_img_resolu_bone_2_cls_2D[:,0], kps_img_resolu_bone_2_cls_2D[:,1]
                
                rendered_image_with_kps = draw_lines_on_img(rendered_image_with_kps, bone_end_pt_1_3D, bone_end_pt_2_3D)
                # rendered_image_with_kps.save(f'img_rendered_with_kps_midpts_{index}.png', bbox_inches='tight') 
            
                
            midpts_calculated_in_3D = True
            if midpts_calculated_in_3D:
                
                # MID-POINTS ORIGINAL (NOT CLOSEST)
                kps_img_resolu_mid_kp20 = (bones_midpts_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_all_kp40 = (bones_2D_proj_all_kp40[index] + 1) * rendered_image_PIL.size[0]/2
                
                #kps_img_resolu_all_31070_vertices = (projected_vertices[index] + 1) * rendered_image_PIL.size[0]/2
                
                rendered_image_with_kps = draw_kp_on_image(kps_img_resolu_mid_kp20, rendered_image_PIL) #, color='yellow')              #[-6:]
                rendered_image_with_kps = draw_kp_on_image(kps_img_resolu_all_kp40, rendered_image_with_kps) #, color='blue')  
                
                kps_img_resolu_bone_1_projected_in_2D = (bone_end_pt_1_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_bone_2_projected_in_2D = (bone_end_pt_2_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
    
                rendered_image_with_kps = draw_lines_on_img(rendered_image_with_kps, kps_img_resolu_bone_1_projected_in_2D, kps_img_resolu_bone_2_projected_in_2D)
                rendered_image_with_kps.save(f'output_folder/img_render_kps_mid3D_NEW_{index}.png', bbox_inches='tight')       
                
                # MID-POINTS CLOSEST 
                rendered_image_PIL = F.to_pil_image(rendered_image[index])
                #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
                kps_img_resolu_mid_kp20_closest = (bones_closest_midpts_projected_in_2D_all_kp20[index] + 1) * rendered_image_PIL.size[0]/2
                rendered_image_with_kps_closest = draw_kp_on_image(kps_img_resolu_mid_kp20_closest, rendered_image_PIL) #, color='yellow')              #[-6:]
                print(f'(kps_img_resolu_mid_kp20_closest: {kps_img_resolu_mid_kp20_closest} and index is {index}')
                rendered_image_with_kps_closest = draw_kp_on_image(kps_img_resolu_all_kp40, rendered_image_with_kps_closest) #, color='blue')  
                
                rendered_image_with_kps_closest = draw_lines_on_img(rendered_image_with_kps_closest, kps_img_resolu_bone_1_projected_in_2D, kps_img_resolu_bone_2_projected_in_2D)
                #rendered_image_with_kps_closest.save(f'img_render_kps_all_10deg_mid3D_closest_{index}.png', bbox_inches='tight') 
                
                rendered_image_with_kps_closest.save(f'output_folder/img_render_kps_NEW{index}.png', bbox_inches='tight') 
                
            # target_dict = compute_correspondences_sd_dino(img1=rendered_image_PIL, img1_kps=kps_img_resolu_mid_kp20, img2=target_image_PIL, index = index) #[-6:]
            
            target_dict = compute_correspondences_sd_dino(img1=rendered_image_PIL, img1_kps=kps_img_resolu_mid_kp20_closest, img2=target_image_updated, index = index)
            
            output_dict = {
            "rendered_kps": kps_img_resolu_mid_kp20_closest,  #[-6:]
            "rendered_image_with_kps": rendered_image_with_kps
            }
            
            output_dict.update(target_dict)
        
        
        return output_dict

    
    def forward(self, batch):
        
        # mesh is of shape <class 'dos.nvdiffrec.render.mesh.Mesh'>
        mesh = batch["mesh"]  # rest pose

        # estimate bones
        # bones_predictor_outputs is dictionary with keys - ['bones_pred', 'skinnig_weights', 'kinematic_chain', 'aux'])
        bones_predictor_outputs = self.bones_predictor(mesh.v_pos)

        batch_size, num_bones = bones_predictor_outputs["bones_pred"].shape[:2]
        
        # BONE ROTATIONS
        bones_rotations = self.articulation_predictor(batch, num_bones)
        
        # NO BONE ROTATIONS
        # bones_rotations = torch.zeros(batch_size, num_bones, 3, device=mesh.v_pos.device)
        
        # DUMMY BONE ROTATIONS - pertrub the bones rotations (only to test the implementation)
        # bones_rotations = bones_rotations + torch.randn_like(bones_rotations) * 0.1
        
        # apply articulation to mesh
        articulated_mesh, aux = mesh_skinning(
            mesh,
            bones_predictor_outputs["bones_pred"],
            bones_predictor_outputs["kinematic_chain"],
            bones_rotations,
            bones_predictor_outputs["skinnig_weights"],
            output_posed_bones=True,
        )

        #articulated_bones_predictor_outputs = self.bones_predictor(articulated_mesh.v_pos)
        
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
        # compute_correspondences for keypoint loss
        correspondences_dict = self.compute_correspondences(
            articulated_mesh,
            batch["pose"],                             # batch["pose"].shape is torch.Size([Batch size, 12])
            self.renderer,
            aux["posed_bones"],                        # predicted articulated bones
            #bones_predictor_outputs["bones_pred"],    # this is a rest pose    # bones_predictor_outputs["bones_pred"].shape is torch.Size([4, 20, 2, 3]), 4 is batch size, 20 is number of bones, 2 are the two endpoints of the bones and 3 means the 3D point defining one of the end points of the line segment in 3D that defines the bone 
            renderer_outputs["image_pred"],            # renderer_outputs["image_pred"].shape is torch.Size([4, 3, 256, 256]), 4 is batch size, 3 is RGB channels, 256 is image resolution
            batch["image"],
        )

        # esamble outputs
        outputs = {}
        # TODO: probaly rename the ouputs of the renderer
        
        outputs.update(renderer_outputs)        # renderer_outputs keys are dict_keys(['image_pred', 'mask_pred', 'albedo', 'shading'])
        outputs.update(correspondences_dict)

        return outputs

    def get_metrics_dict(self, model_outputs, batch):
        return {}

    def get_loss_dict(self, model_outputs, batch):
        
        # TODO: implement keypoint loss
        # 5. Compute the loss between the source and target keypoints
        print('Calculating l1 loss')
        # loss = nn_functional.mse_loss(rendered_keypoints, target_keypoints, reduction='mean')
        
        loss = nn_functional.l1_loss(model_outputs["rendered_kps"], model_outputs["target_corres_kps"], reduction='mean')
        
        return {"loss": loss}

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
