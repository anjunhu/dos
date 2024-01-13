import random
random.seed(42) 

import os
# # Set TORCH_HOME to a custom directory
# os.environ['TORCH_HOME'] = '/work/oishideb/cache/torch_hub'

import time
import timeit
import torch
import cv2
import numpy as np
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
from dos.utils.correspondence import resize, draw_correspondences_1_image #, draw_lines_on_img, plot_points

from ..components.sd_model_text_to_image.diffusion_sds import Stable_Diffusion_Text_to_Target_Img

# UNCOMMENT IT LATER
# from ..components.DeepFlyod_or_sdXL_text2image_inference import StableDiffusionXL, DeepFloydIF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def closest_visible_points(bones_midpts, mesh_v_pos, visible_vertices):
    """
    Find the closest visible points in the mesh to the given bone midpoints.
    
    Parameters:
    - bones_midpts: Tensor of bone midpoints with shape [Batch size, 20, 3]
    - mesh_v_pos: Tensor of mesh vertex positions with shape [Batch size, 31070, 3]
    - visible_vertices: Tensor indicating visibility of each vertex with shape [Batch size, 31070]
    
    Returns:
    - closest_points: Tensor of closest visible points with shape [Batch size, 20, 3]
    """
    
    # Expand dimensions for broadcasting
    bones_midpts_exp = bones_midpts.unsqueeze(2)
    mesh_v_pos_exp = mesh_v_pos.unsqueeze(1)
    
    # Compute squared distances between each bone midpoint and all mesh vertices
    dists = ((bones_midpts_exp - mesh_v_pos_exp) ** 2).sum(-1)
    
    # Mask occluded vertices by setting their distance to a high value
    max_val = torch.max(dists).item() + 1
    occluded_mask = (1 - visible_vertices).bool().unsqueeze(1)
    dists.masked_fill_(occluded_mask, max_val)
    
    # Get the index of the minimum distance for each bone midpoint
    _, closest_idx = dists.min(-1)
    
    # Gather the closest visible points from mesh_v_pos using the computed indices
    batch_indices = torch.arange(bones_midpts.size(0), device=closest_idx.device).unsqueeze(1)
    closest_points = mesh_v_pos[batch_indices, closest_idx, :]
    
    return closest_points

    
def mask_erode_tensor(batch_of_masks):

    # batch_of_masks is a tensor of shape (batch_size, channels, height, width) containing binary masks
    # Set the kernel size for erosion (e.g., 3x3)
    kernel_size = (1, 1)
    
    erode_off_half = False
    if erode_off_half:
        kernel_size = (15, 1)

    # Create a custom erosion function for binary masks
    def binary_erosion(mask, kernel_size):
        # Pad the mask to handle border pixels
        padding = [k // 2 for k in kernel_size]
        mask = nn_functional.pad(mask, padding, mode='constant', value=0)

        # Convert mask to a binary tensor (0 or 1)
        binary_mask = (mask > 0).float()

        # Create a tensor of ones as the kernel
        kernel = torch.ones(1, 1, *kernel_size).to(mask.device)

        # Perform erosion using a convolution
        eroded_mask = nn_functional.conv2d(binary_mask, kernel)

        # Set eroded values to 1 and the rest to 0
        eroded_mask = (eroded_mask == kernel.numel()).float()
        
        # ADDED next two lines
        # Mask out the upper part of the image to keep only the bottom part (legs)
        if erode_off_half:
            height = eroded_mask.shape[2]
            eroded_mask[:, :height//2, :] = 0

        return eroded_mask

    # Loop through the batch and apply erosion to each mask
    eroded_masks = []
    for i in range(batch_of_masks.shape[0]):
        mask = batch_of_masks[i:i+1]  # Extract a single mask from the batch
        eroded_mask = binary_erosion(mask, kernel_size)
        eroded_masks.append(eroded_mask)

    # Stack the results into a single tensor
    eroded_masks = torch.cat(eroded_masks, dim=0)
    
    # eroded_masks is a tensor of shape (batch_size, height, width) containing binary masks
    # Convert the tensors to numpy arrays and scale them to 0-255 range
    eroded_masks_np = (eroded_masks.cpu().numpy() * 255).astype(np.uint8)

    # Loop through the batch and save each mask as an image
    for i in range(eroded_masks_np.shape[0]):
        mask = eroded_masks_np[i]
        pil_image = Image.fromarray(mask)
        #pil_image.save(f'eroded_mask_{i}.png')

        #Now, eroded_masks contains the eroded masks for each mask in the batch

    # eroded_masks shape is torch.Size([4, 254, 256])
    return eroded_masks


def get_vertices_inside_mask_old(projected_visible_v_in_2D, eroded_mask):
    # Resultant list
    vertices_inside_mask = []
    
    # Iterate over the batch size
    for i in range(projected_visible_v_in_2D.shape[0]):
        # Filter the vertices for the current batch
        current_vertices = projected_visible_v_in_2D[i]
        
        # Make sure the vertex coordinates are in int format and within bounds
        valid_vertices = current_vertices.int().clamp(min=0, max=255).long()
        
        # Check if these vertices lie inside the mask
        mask_values = eroded_mask[i, valid_vertices[:, 1], valid_vertices[:, 0]]
        
        # Filter out the vertices based on the mask
        inside_mask = current_vertices[mask_values == 1]
        
        # Append to the resultant list
        vertices_inside_mask.append(inside_mask)
    
    return vertices_inside_mask


def get_vertices_inside_mask(projected_visible_v_in_2D, eroded_mask):
    # Resultant list
    vertices_inside_mask = []
    
    # To determine the maximum number of vertices that are inside the mask for all batches
    max_vertices = 0
    
    # Iterate over the batch size
    for i in range(projected_visible_v_in_2D.shape[0]):
        # Filter the vertices for the current batch
        current_vertices = projected_visible_v_in_2D[i]
        
        # Make sure the vertex coordinates are in int format and within bounds
        valid_vertices = current_vertices.int().clamp(min=0, max=255).long()
        
        # Check if these vertices lie inside the mask
        mask_values = eroded_mask[i, valid_vertices[:, 1], valid_vertices[:, 0]]
        
        # Filter out the vertices based on the mask
        inside_mask = current_vertices[mask_values == 1]
        
        # Update the max_vertices value
        max_vertices = max(max_vertices, inside_mask.shape[0])
        
        # Append to the resultant list
        vertices_inside_mask.append(inside_mask)
    
    # Pad each tensor in the list to have max_vertices vertices
    for i in range(len(vertices_inside_mask)):
        padding = max_vertices - vertices_inside_mask[i].shape[0]
        if padding > 0:
            padding_tensor = torch.zeros((padding, 2)).to(vertices_inside_mask[i].device)
            vertices_inside_mask[i] = torch.cat([vertices_inside_mask[i], padding_tensor], dim=0)
    
    # Convert the list of tensors to a single tensor
    vertices_inside_mask = torch.stack(vertices_inside_mask, dim=0)
    
    return vertices_inside_mask

def get_closest_vertices_fr_mask_old(projected_visible_v_in_2D, eroded_mask):
    # Resultant tensor with same shape as projected_visible_v_in_2D
    closest_vertices = torch.zeros_like(projected_visible_v_in_2D)

    # Iterate over the batch size
    for i in range(projected_visible_v_in_2D.shape[0]):
        # Get the indices of all points in the eroded_mask where the value is 1
        mask_indices = torch.nonzero(eroded_mask[i] == 1, as_tuple=False).float()

        # Iterate over the 20 keypoints
        for j in range(projected_visible_v_in_2D.shape[1]):
            # Compute the squared distance from the current keypoint to all points in the mask
            distances = torch.norm((mask_indices - projected_visible_v_in_2D[i, j, :]), dim=1)

            # Find the index of the minimum distance
            min_distance_idx = torch.argmin(distances)

            # Store the x,y coordinate of the closest point in the result tensor
            closest_vertices[i, j, :] = mask_indices[min_distance_idx]
    
            # print(f'closest_vertices {closest_vertices[i, j, :]} and projected_visible_v_in_2D {projected_visible_v_in_2D[i, j, :]}')

    return closest_vertices

def get_closest_vertices_fr_mask_old(projected_visible_v_in_2D, eroded_mask):
    # Resultant tensor with same shape as projected_visible_v_in_2D
    closest_vertices = torch.zeros_like(projected_visible_v_in_2D)

    # Iterate over the batch size
    for i in range(projected_visible_v_in_2D.shape[0]):
        # Get the indices of all points in the eroded_mask where the value is 1
        mask_indices = torch.nonzero(eroded_mask[i] == 1, as_tuple=False).float()

        # Iterate over the 20 keypoints
        for j in range(projected_visible_v_in_2D.shape[1]):
            # Compute the squared distance from the current keypoint to all points in the mask
            distances = torch.norm((mask_indices - projected_visible_v_in_2D[i, j, :]), dim=1)

            # Find the index of the minimum distance
            min_distance_idx = torch.argmin(distances)

            # Store the x,y coordinate of the closest point in the result tensor
            closest_vertices[i, j, :] = mask_indices[min_distance_idx]
            
            # Ensure that the closest vertex is inside the mask
            y, x = closest_vertices[i, j, :].long()
            while eroded_mask[i, y, x] != 1:
                distances[min_distance_idx] = float('inf')  # Set the current minimum distance to infinity
                min_distance_idx = torch.argmin(distances)  # Find the new minimum distance index
                closest_vertices[i, j, :] = mask_indices[min_distance_idx]
                y, x = closest_vertices[i, j, :].long()

            # print(f'closest_vertices {closest_vertices[i, j, :]} and projected_visible_v_in_2D {projected_visible_v_in_2D[i, j, :]}')

    return closest_vertices


def get_closest_vertices_fr_mask(projected_visible_v_in_2D, eroded_mask):
    # Resultant tensor with same shape as projected_visible_v_in_2D
    closest_vertices = torch.zeros_like(projected_visible_v_in_2D)

    # Iterate over the batch size
    for i in range(projected_visible_v_in_2D.shape[0]):
        # Get the indices of all points in the eroded_mask where the value is 1
        mask_indices = torch.nonzero(eroded_mask[i] == 1, as_tuple=False).float()

        # Iterate over the keypoints
        for j in range(projected_visible_v_in_2D.shape[1]):
            # Initialize distances
            distances = torch.cdist(projected_visible_v_in_2D[i, j, :].unsqueeze(0), mask_indices).squeeze(0)
            min_distance_idx = torch.argmin(distances)

            # Store the x,y coordinate of the closest point in the result tensor
            closest_vertices[i, j, :] = mask_indices[min_distance_idx]

            # Ensure that the closest vertex is inside the mask
            y, x = closest_vertices[i, j, :].long()
            while eroded_mask[i, y, x] != 1:
                distances[min_distance_idx] = float('inf')  # Set the current minimum distance to infinity
                min_distance_idx = torch.argmin(distances)  # Find the new minimum distance index
                closest_vertices[i, j, :] = mask_indices[min_distance_idx]
                y, x = closest_vertices[i, j, :].long()

            # print(f'closest_vertices {closest_vertices[i, j, :]} and projected_visible_v_in_2D {projected_visible_v_in_2D[i, j, :]}')

    return closest_vertices

def sample_points_on_line(pt1, pt2, num_samples):
    """
    Sample points on lines defined by pt1 and pt2, excluding the endpoints.
    
    Parameters:
    - pt1: Tensor of shape [Batch size, 20, 3] representing the first endpoints
    - pt2: Tensor of shape [Batch size, 20, 3] representing the second endpoints
    - num_samples: Number of points to sample on each line
    
    Returns:
    - sampled_points: Tensor of shape [Batch size, 20, num_samples, 3] containing the sampled points
    """
    
    # Create a tensor for linear interpolation
    alpha = torch.linspace(0, 1, num_samples + 2)[1:-1].to(pt1.device)  # Exclude 0 and 1 to avoid endpoints
    alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 1, num_samples, 1]
    
    # Linear interpolation formula: (1 - alpha) * pt1 + alpha * pt2
    sampled_points = (1 - alpha) * pt1.unsqueeze(2) + alpha * pt2.unsqueeze(2)
    
    # Reshape to [Batch size, 100, 3]
    batch_size = pt1.size(0)
    sampled_points = sampled_points.reshape(batch_size, -1, 3)
    
    return sampled_points


class Articulator(BaseModel):
    """
    Articulator predicts instance shape parameters (instance shape) - optimisation based - predictor takes only id as input
    """

    # TODO: set default values for the parameters (dataclasses have a nice way of doing it
    #   but it's not compatible with torch.nn.Module)
    
    def __init__(
        self,
        path_to_save_images,
        cache_dir,
        num_pose,
        sd_Text_to_Target_Img=None,
        encoder=None,
        enable_texture_predictor=True,
        texture_predictor=None,
        bones_predictor=None,
        articulation_predictor=None,
        renderer=None,
        shape_template_path=None
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
        
        self.path_to_save_images = path_to_save_images
        
        self.cache_dir = cache_dir
    
        # self.sd_Text_to_Target_Img = sd_Text_to_Target_Img if sd_Text_to_Target_Img is not None else Stable_Diffusion_Text_to_Target_Img(cache_dir = self.cache_dir)
        
        self.sd_Text_to_Target_Img = Stable_Diffusion_Text_to_Target_Img(cache_dir = self.cache_dir)
        
        self.num_pose= num_pose

    def _load_shape_template(self, shape_template_path):
        return load_mesh(shape_template_path)

    def compute_correspondences(
        self, articulated_mesh, pose, renderer, bones, rendered_mask, rendered_image, target_image, sd_model, sd_aug
    ):
        # 1. Extract features from the rendered image and the target image
        # 2. Sample keypoints from the rendered image
        #  - sample keypoints along the bones
        #  - find the closest visible point on the articulated_mesh in 3D (the visibility is done in 2D)
        # 3. Extract features from the source keypoints
        # 4. Find corresponding target keypoints (TODO: some additional tricks e.g. optimal transport etc.)
        
        # target_image_path = f'/users/oishideb/dos/target_images/'

        start_time = time.time()
         
        # get visible vertices
        mvp, _, _ = geometry_utils.get_camera_extrinsics_and_mvp_from_pose(
            pose,
            renderer.fov,
            renderer.znear,
            renderer.zfar,
            renderer.cam_pos_z_offset,
        )
        end_time = time.time()
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'get_camera_extrinsics_and_mvp_from_pose' compute took {end_time - start_time} seconds to run.\n")
            
        print(f"The 'get_camera_extrinsics_and_mvp_from_pose' function took {end_time - start_time} seconds to run.")
        
        
        start_time = time.time()
        # All the visible_vertices in 2d
        # visible_vertices.shape is torch.Size([2, 31070]) 
        visible_vertices = mesh_utils.get_visible_vertices(                                                  
            articulated_mesh, mvp, renderer.resolution
        )
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'get_visible_vertices' took {end_time - start_time} seconds to run.\n")
            
        print(f"The get_visible_vertices function took {end_time - start_time} seconds to run.")
        
        # Reshape visible_vertices to match the shape of v_pos
        # visible_vertices = visible_vertices.unsqueeze(2).expand(-1, -1, 3)
        
        eroded_mask = mask_erode_tensor(rendered_mask)
            
        Sample_farthest_points = False
        if Sample_farthest_points:
                
            visible_v_coordinates_list = []

            # Loop over each batch
            for i in range(visible_vertices.size(0)):
                # Extract the visible vertex coordinates using the boolean mask from visible_vertices
                visible_v_coordinates = articulated_mesh.v_pos[i][visible_vertices[i].bool()]
                visible_v_coordinates_list.append(visible_v_coordinates)

            # Find the maximum number of visible vertices across all batches
            max_visible_vertices = max([tensor.size(0) for tensor in visible_v_coordinates_list])

            # Pad each tensor in the list to have shape [max_visible_vertices, 3]
            padded_tensors = []
            for tensor in visible_v_coordinates_list:
                # Calculate the number of padding rows required
                padding_rows = max_visible_vertices - tensor.size(0)
                # Create a padding tensor of shape [padding_rows, 3] filled with zeros (or any other value)
                padding = torch.zeros((padding_rows, 3), device=tensor.device, dtype=tensor.dtype)
                # Concatenate the tensor and the padding
                padded_tensor = torch.cat([tensor, padding], dim=0)
                padded_tensors.append(padded_tensor)

            # Convert the list of padded tensors to a single tensor
            visible_v_position = torch.stack(padded_tensors, dim=0)
            
            #### Sample farthest points
            visible_v_position = visible_v_position.permute(0,2,1)
            
            num_samples = 100
            visible_v_position = geometry_utils.sample_farthest_points(visible_v_position, num_samples)
            visible_v_position = visible_v_position.permute(0,2,1)
            
            projected_visible_v_in_2D = geometry_utils.project_points(visible_v_position, mvp)  
        
    
            # Convert to Pixel Coordinates of the mask
            pixel_projected_visible_v_in_2D = (projected_visible_v_in_2D + 1) * eroded_mask.size(1)/2
            vertices_inside_mask = get_vertices_inside_mask(pixel_projected_visible_v_in_2D, eroded_mask)
            kps_img_resolu = (vertices_inside_mask/256) * 840
            
            # project vertices/keypoints example
            # mesh.v_pos shape is torch.Size([2, 31070, 3])
            # mvp.shape is torch.Size([Batch size, 4, 4])
            projected_vertices.shape is torch.Size([4, 31070, 2]), # mvp is model-view-projection
            projected_vertices = geometry_utils.project_points(articulated_mesh.v_pos, mvp)  
            
            projected_vertices = projected_vertices[:, :100, :]
            kps_img = projected_vertices[:,:,:][0] * rendered_image.shape[2]

        bone_kps = True
        if bone_kps:
            bone_end_pt_1_3D = bones[:, :, 0, :]  # one end of the bone in 3D
            bone_end_pt_2_3D = bones[:, :, 1, :]  # other end of the bone in 3D
            
            bones_in_3D_all_kp40 = torch.cat((bone_end_pt_1_3D, bone_end_pt_2_3D), dim=1)
            bones_2D_proj_all_kp40 = geometry_utils.project_points(bones_in_3D_all_kp40, mvp)
            
            bone_end_pt_1_projected_in_2D = geometry_utils.project_points(bone_end_pt_1_3D, mvp)
            bone_end_pt_2_projected_in_2D = geometry_utils.project_points(bone_end_pt_2_3D, mvp)
            
            bones_midpts_in_3D = (bones[:, :, 0, :] + bones[:, :, 1, :]) / 2.0        # This is in 3D the shape is torch.Size([2, 20, 3])
            
            # NEWLY ADDED: SAMPLE POINTS ON BONE LINE
            bones_midpts_in_3D = sample_points_on_line(bone_end_pt_1_3D, bone_end_pt_2_3D, 3)
            
            bones_midpts_projected_in_2D = geometry_utils.project_points(bones_midpts_in_3D, mvp)
        
            start_time = time.time()
            closest_midpts = closest_visible_points(bones_midpts_in_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
            
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'closest_visible_points' took {end_time - start_time} seconds to run.\n")

            print(f"The closest_visible_points function took {end_time - start_time} seconds to run.")
            
            ## shape of bones_closest_pts_2D_proj is ([Batch-size, 20, 2])
            bones_closest_midpts_projected_in_2D_all_kp20 = geometry_utils.project_points(closest_midpts, mvp)
            
            # ADDED next 3 lines: Convert to Pixel Coordinates of the mask
            pixel_projected_visible_v_in_2D = (bones_closest_midpts_projected_in_2D_all_kp20 + 1) * eroded_mask.size(1)/2
            img_pixel_reso = (pixel_projected_visible_v_in_2D/256) * 840
            kps_img_resolu = (pixel_projected_visible_v_in_2D/256) * 840
            # print('img_pixel_reso ', img_pixel_reso)
            
            ###  Commented Out
            start_time = time.time()
            ## get_vertices_inside_mask
            vertices_inside_mask = get_vertices_inside_mask(pixel_projected_visible_v_in_2D, eroded_mask)
            
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'get_vertices_inside_mask' took {end_time - start_time} seconds to run.\n")
            
            print(f"The get_vertices_inside_mask function took {end_time - start_time} seconds to run.")
        
            kps_img_resolu = (vertices_inside_mask/256) * 840
            
            bone_end_pt_1 = closest_visible_points(bone_end_pt_1_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
            bone_end_pt_2 = closest_visible_points(bone_end_pt_2_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
            
            # bone_end_pt_1_in_2D_cls = geometry_utils.project_points(bone_end_pt_1, mvp)
            # bone_end_pt_2_in_2D_cls = geometry_utils.project_points(bone_end_pt_2, mvp)
            
            # bones_mid_pt_in_2D = (bone_end_pt_1_in_2D_cls + bone_end_pt_2_in_2D_cls) / 2.0
            
            bones_all = torch.cat((bone_end_pt_1, bone_end_pt_2), dim=1)
            
            bones_all = closest_visible_points(bones_all, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
            # bones_closest_pts_2D_proj_all_kp40 = geometry_utils.project_points(bones_all, mvp)
            
            # Define a list to hold the converted images
            rendered_images_PIL_list = []
            target_images_PIL_list = []
            
        output_dict = {}
        
        corres_target_kps_tensor_stack = torch.empty(0, kps_img_resolu.shape[1], 2, device=kps_img_resolu.device)
        cycle_consi_kps_tensor_stack = torch.empty(0, kps_img_resolu.shape[1], 2, device=kps_img_resolu.device)
            
        target_image_with_kps_list = []
        rendered_image_with_kps_list = []
        cycle_consi_image_with_kps_list = []
        
        target_image_with_kps_list_after_cyc_check = []
        rendered_image_with_kps_list_after_cyc_check =[]
         
        kps_img_resolu_list = []
        corres_target_kps_list = []
        
        # shape of rendered_image and target_image is [batch size, 3, 256, 256]
        print('rendered_image.shape', rendered_image.shape)
        print('target_image.shape', target_image.shape)
        
        # Iterate over the batch dimension and convert each sample
        # for index in range(rendered_image.shape[0]):
        
        for index, kps_1_batch in enumerate(kps_img_resolu):
            
            # rendered_image_PIL is 256*256 (default size)
            rendered_image_PIL = F.to_pil_image(rendered_image[index])
            rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            rendered_image_PIL.save(f'{self.path_to_save_images}/{index}_rendered_image_only.png', bbox_inches='tight')
            
            eroded_mask_PIL = F.to_pil_image(eroded_mask[index])
            # eroded_mask_PIL = resize(eroded_mask_PIL, target_res = 840, resize=True, to_pil=True)
            eroded_mask_PIL.save(f'{self.path_to_save_images}/{index}_eroded_mask.png', bbox_inches='tight')
            
            #rendered_images_PIL_list.append(rendered_image_PIL)
            
            print('target_image.shape', target_image.shape)
            print('target_image[0].shape', target_image[0].shape)
            target_image_PIL = F.to_pil_image(target_image[0])
            #target_image_PIL = resize(target_image_PIL, target_res = 840, resize=True, to_pil=True)
            #target_images_PIL_list.append(target_image_PIL)
            
            # target_image_updated = os.path.join(target_image_path+f'{index}_image_gt_save.png')
            # target_image_updated = Image.open(target_image_updated).convert('RGB')
            
            # num_random_kp = 29
            sample_fr_mask = False
            if sample_fr_mask:
                
                # # Sample 'num_random_kp' random points for each batch
                # sampled_vertices = torch.zeros(rendered_image.shape[0], num_random_kp, 2)
                # for i in range(kps_img_resolu.shape[0]):
                #     indices = torch.randperm(kps_img_resolu.shape[1])[:num_random_kp]
                #     sampled_vertices[i] = kps_img_resolu[i][indices]
                    
                # sampled_vertices = (sampled_vertices[index]/256) * 840
                #rendered_image_with_kps = draw_correspondences_1_image(sampled_vertices, rendered_image_PIL, index=0) 
                
                rendered_image_with_kps = draw_correspondences_1_image(kps_1_batch, rendered_image_PIL, index=0)  # , num_samples
                # rendered_image_with_kps = plot_points(kps_img_resolu, rendered_image_PIL)
                
                # Set the background color to grey
                plt.gcf().set_facecolor('grey')
                rendered_image_with_kps.savefig(f'{self.path_to_save_images}/{index}_rendered_image_vert_fr_mask.png', bbox_inches='tight')
                
                
            midpts_calculated_in_3D = True
            if midpts_calculated_in_3D:
                
                # MID-POINTS ORIGINAL (NOT CLOSEST)
                kps_img_resolu_mid_kp20 = (bones_midpts_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_all_kp40 = (bones_2D_proj_all_kp40[index] + 1) * rendered_image_PIL.size[0]/2
                
                #kps_img_resolu_all_visible_vertices = (projected_visible_v_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                # rendered_image_with_kps = draw_correspondences_1_image(kps_img_resolu_mid_kp20, rendered_image_PIL, index=0) #, color='yellow')              #[-6:]
            
                kps_img_resolu_bone_1_projected_in_2D = (bone_end_pt_1_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2
                kps_img_resolu_bone_2_projected_in_2D = (bone_end_pt_2_projected_in_2D[index] + 1) * rendered_image_PIL.size[0]/2

                # showing error atm
                #rendered_image_with_kps = draw_lines_on_img(rendered_image_with_kps, kps_img_resolu_bone_1_projected_in_2D, kps_img_resolu_bone_2_projected_in_2D)
                #rendered_image_with_kps.save(f'{self.path_to_save_images}/img_render_kps_mid3D{index}.png', bbox_inches='tight')       
                
            # MID-POINTS CLOSEST 
            rendered_image_PIL = F.to_pil_image(rendered_image[index])
            rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            rendered_image_PIL.save(f'{self.path_to_save_images}/{index}_rendered_image_only.png', bbox_inches='tight')
            
            # kps_img_resolu = (bones_closest_midpts_projected_in_2D_all_kp20[index] + 1) * rendered_image_PIL.size[0]/2
            
            
            # bones mid pts
            # rendered_image_with_kps_2 = draw_correspondences_1_image(img_pixel_reso[index], rendered_image_PIL, index=0) 
            # Set the background color to grey
            # plt.gcf().set_facecolor('grey')
            # rendered_image_with_kps_2.savefig(f'{self.path_to_save_images}/{index}_rendered_image_with_midpts.png', bbox_inches='tight')
            
            
            #rendered_image_with_kps_closest.save(f'{self.path_to_save_images}/img_render_kps_NEW{index}.png', bbox_inches='tight') 
            
            # target_dict = compute_correspondences_sd_dino(img1=rendered_image_PIL, img1_kps=kps_img_resolu_mid_kp20, img2=target_image_PIL, index = index) #[-6:]
            # target_dict = compute_correspondences_sd_dino(img1=rendered_image_PIL, img1_kps=kps_img_resolu_mid_kp20_closest, img2=target_image_PIL, index = index)
            
            start_time = time.time()
            # shape is ([10, 60, 2])
            print('kps_1_batch.shape', kps_1_batch.shape)
            
            target_image_with_kps, corres_target_kps, cycle_consi_image_with_kps, cycle_consi_corres_kps = compute_correspondences_sd_dino(img1=rendered_image_PIL, img1_kps=kps_1_batch, img2=target_image_PIL, index = index ,model=sd_model, aug=sd_aug)
            
            end_time = time.time()  # Record the end time

            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'compute_correspondences_sd_dino' took {end_time - start_time} seconds to run.\n")
                
            print(f"The compute_correspondences_sd_dino function took {end_time - start_time} seconds to run.")

            print('kps_1_batch.shape', kps_1_batch.shape)
            print('corres_target_kps.shape', corres_target_kps.shape)
            print('cycle_consi_corres_kps.shape', cycle_consi_corres_kps.shape)
            
            # LOSS
            loss = nn_functional.l1_loss(kps_1_batch, corres_target_kps, reduction='mean')
            # draw.text((50, 50), f"L1 Loss:{loss}", fill='orange', font = font)
            
            rendered_image_with_kps = draw_correspondences_1_image(kps_1_batch, rendered_image_PIL, index = 0) #, color='yellow')              #[-6:]
            # Set the background color to grey
            plt.gcf().set_facecolor('grey')
            plt.text(80, 0.95, f'Rendered Img ; Loss: {loss}', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
            rendered_image_with_kps.savefig(f'{self.path_to_save_images}/{index}_rendered_image_with_kps.png', bbox_inches='tight') 
            
            # FIX IT
            # plt.text(80, 0.95, f'Target Img', verticalalignment='top', horizontalalignment='left', color = 'black', fontsize ='11')
            # Set the background color to grey
            plt.gcf().set_facecolor('grey')
            target_image_with_kps.savefig(f'{self.path_to_save_images}/{index}_target.png', bbox_inches='tight')
                    
            rendered_image_with_kps_list.append(rendered_image_with_kps)
            target_image_with_kps_list.append(target_image_with_kps)
            plt.close()

            # LOSS CALCULATED AFTER CYCLE-CONSISTENCY CHECK
            loss = nn_functional.l1_loss(kps_1_batch, cycle_consi_corres_kps, reduction='mean')
            # draw.text((50, 50), f"L1 Loss:{loss}", fill='orange', font = font)
            # plt.text(80, 0.95, f' Loss: {loss}', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
            # Set the background color to grey
            plt.gcf().set_facecolor('grey')
            plt.text(80, 0.95, f'Cycle Consistency', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
            cycle_consi_image_with_kps.savefig(f'{self.path_to_save_images}/{index}_cycle.png', bbox_inches='tight')
            
            cycle_consi_image_with_kps_list.append(cycle_consi_image_with_kps)
            cycle_consi_kps_tensor_stack = torch.cat((cycle_consi_kps_tensor_stack, cycle_consi_corres_kps.unsqueeze(0)), dim=0)
            
            # REMOVING POINTS FOLLOWING CYCLE CONSISTENCY CHECK
            # Calculate the difference
            difference = torch.abs(kps_1_batch - cycle_consi_corres_kps)
            # Find the points where the difference is less than or equal to 2
            mask = torch.all(difference <= 15, dim=1)
            # Apply the mask to kps_img_resolu
            kps_img_resolu = kps_1_batch[mask]
            
            # Update the Target kps after CYCLE CONSISTENCY CHECK
            corres_target_kps = corres_target_kps[mask]
            
            # corres_target_kps_tensor_stack = torch.cat((corres_target_kps_tensor_stack, corres_target_kps.unsqueeze(0)), dim=0)
            
            rendered_image_with_kps_cyc_check = draw_correspondences_1_image(kps_img_resolu, rendered_image_PIL, index = 0) #, color='yellow')              #[-6:]
            
            # Set the background color to grey
            plt.gcf().set_facecolor('grey')
            plt.text(30, 0.95, f'Final Rendered Img after Eroded Mask & Cycle Consi Check', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
            plt.text(80, 40, f'Loss: {loss}', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
            rendered_image_with_kps_cyc_check.savefig(f'{self.path_to_save_images}/{index}_rendered_image_with_kps_after_cyclic_check.png', bbox_inches='tight') 
            
            # Set the background color to grey
            plt.gcf().set_facecolor('grey')    
            target_image_with_kps_cyc_check = draw_correspondences_1_image(corres_target_kps, target_image_PIL, index = 0) #, color='yellow')              #[-6:]
            target_image_with_kps_cyc_check.savefig(f'{self.path_to_save_images}/{index}_target_image_with_kps_after_cyclic_check.png', bbox_inches='tight') 
            plt.close()    
            
            rendered_image_with_kps_list_after_cyc_check.append(rendered_image_with_kps_cyc_check)
            target_image_with_kps_list_after_cyc_check.append(target_image_with_kps_cyc_check)
            corres_target_kps_list.append(corres_target_kps)
            kps_img_resolu_list.append(kps_img_resolu)
            
        
        # Following cycle consistency check some of the points got removed, in order to make the lenght same, it has been padded with zeros.
        # Find the maximum length
        max_length = max(len(item) for item in kps_img_resolu_list if hasattr(item, '__len__'))

        # # Find the maximum length (max value of n)
        # max_length = max(tensor.shape[0] for tensor in tensor_list)

        # Function to pad a tensor to the max_length
        def pad_tensor(tensor, max_length, device):
            n = tensor.shape[0]
            if n < max_length:
                padding = max_length - n
                # Ensuring the padding tensor is on the same device as the input tensor
                padding_tensor = torch.zeros(padding, 2, device=device)
                padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
                return padded_tensor
            return tensor

        # Pad tensors in both lists
        padded_kps_img_resolu_list = [pad_tensor(tensor.to(device), max_length, device) for tensor in kps_img_resolu_list]
        padded_corres_target_kps_list = [pad_tensor(tensor.to(device), max_length, device) for tensor in corres_target_kps_list]
        
        # Print padded tensors (for verification)
        for tensor1, tensor2 in zip(padded_kps_img_resolu_list, padded_corres_target_kps_list):
            print(tensor1)
            print(tensor2)
            print("---")  # Separator for clarity

        
        output_dict = {
        "rendered_kps": torch.stack(padded_kps_img_resolu_list),  #[-6:]
        "rendered_image_with_kps": rendered_image_with_kps_list,
        "target_image_with_kps": target_image_with_kps_list,
        "target_corres_kps": torch.stack(padded_corres_target_kps_list),     # corres_target_kps_tensor_stack,
        "cycle_consi_image_with_kps": cycle_consi_image_with_kps_list,
        "rendered_image_with_kps_list_after_cyc_check": rendered_image_with_kps_list_after_cyc_check,
        "target_image_with_kps_list_after_cyc_check": target_image_with_kps_list_after_cyc_check
        }
        
        # output_dict.update(target_dict)
        
        return output_dict

    
    def forward(self, batch, sd_model, sd_aug):
        
        batch_size = batch["image"].shape[0]
        if self.shape_template is not None:
            mesh = self.shape_template.extend(batch_size)
        else:
            mesh = batch["mesh"]  # rest pose
        
        # estimate bones
        # bones_predictor_outputs is dictionary with keys - ['bones_pred', 'skinnig_weights', 'kinematic_chain', 'aux'])
        start_time = time.time()
        bones_predictor_outputs = self.bones_predictor(mesh.v_pos)

        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'bones_predictor' took {end_time - start_time} seconds to run.\n")
                
        print(f"The bones_predictor function took {end_time - start_time} seconds to run.")
        
        batch_size, num_bones = bones_predictor_outputs["bones_pred"].shape[:2]
        
        # BONE ROTATIONS
        start_time = time.time()
        bones_rotations = self.articulation_predictor(batch, num_bones)
        
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'articulation_predictor' took {end_time - start_time} seconds to run.\n")
        print(f"The articulation_predictor function took {end_time - start_time} seconds to run.")
                
        # NO BONE ROTATIONS
        # bones_rotations = torch.zeros(batch_size, num_bones, 3, device=mesh.v_pos.device)
        
        # DUMMY BONE ROTATIONS - pertrub the bones rotations (only to test the implementation)
        # bones_rotations = bones_rotations + torch.randn_like(bones_rotations) * 0.1
    
        start_time = time.time()  # Record the start time
        
        # apply articulation to mesh
        articulated_mesh, aux = mesh_skinning(
            mesh,
            bones_predictor_outputs["bones_pred"],
            bones_predictor_outputs["kinematic_chain"],
            bones_rotations,
            bones_predictor_outputs["skinnig_weights"],
            output_posed_bones=True,
        )

        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'mesh_skinning' took {end_time - start_time} seconds to run.\n")
        print(f"The mesh_skinning function took {end_time - start_time} seconds to run.")
        
        #articulated_bones_predictor_outputs = self.bones_predictor(articulated_mesh.v_pos)
        
        # render mesh
        if "texture_features" in batch:
            texture_features = batch["texture_features"]
        else:
            texture_features = None

        if self.enable_texture_predictor:
            
            start_time = time.time()
            material = self.texture_predictor
            end_time = time.time()

            with open('log.txt', 'a') as file:
                file.write(f"The function took {end_time - start_time} seconds to run.\n")
                
        else:
            # if texture predictor is not enabled, use the loaded material from the mesh
            material = mesh.material

        # if pose not provided, compute it from the camera matrix
        if "pose" not in batch:
            
            # COMMENT out for 3D/Multiview
            # # pose shape is [1,12]
            # pose = geometry_utils.blender_camera_matrix_to_magicpony_pose(
            #     batch["camera_matrix"]
            # )
            
            # For Multi View
            # # pose shape is [10, 12]
            pose, pose_directions = utils.rand_poses(self.num_pose, device=torch.device('cuda:0'))
        else:
            pose=batch["pose"]
        
        if "background" in batch:
            background = batch["background"]
        else:
            background = None

        start_time = time.time()
        
        renderer_outputs = self.renderer(
            articulated_mesh,
            material=material,
            pose=pose,
            im_features=texture_features,
            # CHANGED IT
            # background=background,
        )
        
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'renderer' took {end_time - start_time} seconds to run.\n")
        print(f"The renderer function took {end_time - start_time} seconds to run.")
        
        # target_img_tensor_list = []
        
        # Creates an empty tensor to hold the final result
        # all_generated_target_img shape is [num_pose, 3, 256, 256]
        all_generated_target_img = torch.empty(renderer_outputs["image_pred"].shape[0:])
        
        for i in range(pose.shape[0]):
            
            # renderer_outputs = self.renderer(
            # articulated_mesh,
            # material=material,
            # pose=pose,
            # im_features=texture_features,
            # # CHANGED IT
            # # background=background,
            # )
            
            rendered_image_PIL = F.to_pil_image(renderer_outputs["image_pred"][i])
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            dir_path = f'{self.path_to_save_images}/all_iteration_Train/batch_size_0/diff_pose/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            rendered_image_PIL.save(f'{dir_path}{i}_diff_pose_rendered_image.png', bbox_inches='tight')
            

            # target_img_fr_sd = self.sd_Text_to_Target_Img(
            # cache_dir="/work/oishideb/cache/huggingface_hub",
            # output_dir='output-new',
            # init_image_path='/users/oishideb/laam/dos/examples/data/cow.png',
            # vis_name='cow-sds_latent-l2_image-600-lr1e-1.jpg',
            # prompts=['a running cow'],
            # negative_prompts=[''],
            # mode="sds_latent-l2_image",
            # lr=0.1,
            # lr_l2=1e4,
            # seed=2,
            # num_inference_steps= 8,
            # guidance_scale= 100,
            # device=torch.device('cuda:0'),
            # optimizer_class=torch.optim.SGD,
            # torch_dtype=torch.float16,
            # input_image = renderer_outputs["image_pred"][i],
            # image_fr_path = False
            # )
            
            target_img_rgb, target_img_decoded = self.sd_Text_to_Target_Img.run_experiment(
            input_image = renderer_outputs["image_pred"][i],
            image_fr_path = False
            )
            
            #target_img_rgb, target_img_decoded = target_img_fr_sd.run_experiment()
            
            target_image_PIL = F.to_pil_image(target_img_rgb[0])
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            
            dir_path = f'{self.path_to_save_images}/all_iteration_Train/batch_size_0/diff_pose/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            target_image_PIL.save(f'{dir_path}{i}_diff_pose_target_image.png', bbox_inches='tight')

            print("target_img_rgb.shape", target_img_rgb.shape)
            
            # Inserts the new image into the final tensor
            all_generated_target_img[i] = target_img_rgb
            
            
        print('batch["image"].shape', batch["image"].shape)
        
        start_time = time.time()
        # compute_correspondences for keypoint loss
        correspondences_dict = self.compute_correspondences(
            articulated_mesh,
            pose,                                      # batch["pose"].shape is torch.Size([Batch size, 12])
            self.renderer,
            aux["posed_bones"],                        # predicted articulated bones
            #bones_predictor_outputs["bones_pred"],    # this is a rest pose    # bones_predictor_outputs["bones_pred"].shape is torch.Size([4, 20, 2, 3]), 4 is batch size, 20 is number of bones, 2 are the two endpoints of the bones and 3 means the 3D point defining one of the end points of the line segment in 3D that defines the bone 
            renderer_outputs["mask_pred"],
            renderer_outputs["image_pred"],            # renderer_outputs["image_pred"].shape is torch.Size([4, 3, 256, 256]), 4 is batch size, 3 is RGB channels, 256 is image resolution
            all_generated_target_img,                  # CHANGED replaced the target image with sd generated, BEFORE batch["image"],
            # batch["image"],                          # static Target Images
            sd_model, 
            sd_aug
        )

        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'compute_correspondences' took {end_time - start_time} seconds to run.\n")
        
        print(f"The compute_correspondences took {end_time - start_time} seconds to run.")
        
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
        model_outputs["rendered_kps"] = model_outputs["rendered_kps"].to(device)
        model_outputs["target_corres_kps"] = model_outputs["target_corres_kps"].to(device)

        loss = nn_functional.l1_loss(model_outputs["rendered_kps"], model_outputs["target_corres_kps"], reduction='mean')
        
        # print('model_outputs["target_corres_kps"] shape', model_outputs["target_corres_kps"].shape)
        # print('model_outputs["rendered_kps"] Shape', model_outputs["rendered_kps"].shape)
        # print('Loss from inside get_loss_dict function is: ', loss)
        
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
