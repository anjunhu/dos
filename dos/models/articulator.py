import random
import os
# # Set TORCH_HOME to a custom directory
# os.environ['TORCH_HOME'] = '/work/oishideb/cache/torch_hub'
import time
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_functional
import torchvision.transforms.functional as F
import torchvision
from PIL import Image, ImageDraw

from dos.components.fuse.compute_correspond import \
    ComputeCorrespond  # compute_correspondences_sd_dino
from dos.components.fuse.extractor_sd import load_model
from dos.utils.correspondence import (draw_correspondences_1_image,
                                      padding_tensor, resize,
                                      tensor_to_matplotlib_figure)

from ..components.diffusion_model_text_to_image.diffusion_sds import \
    DiffusionForTargetImg
from ..components.skinning.bones_estimation import BonesEstimator
from ..components.skinning.skinning import mesh_skinning
from ..modules.renderer import Renderer
from ..nvdiffrec.render.mesh import load_mesh
from ..predictors.articulation_predictor import ArticulationPredictor
from ..predictors.texture import TexturePredictor
from ..utils import geometry as geometry_utils
from ..utils import mesh as mesh_utils
from ..utils import multi_view, utils
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
        path_to_save_images,
        num_pose_for_optim,
        num_pose_for_visual,
        num_sample_bone_line,
        num_sample_farthest_points = 100,
        mode_kps_selection = "kps_fr_sample_on_bone_line",
        enable_texture_predictor=True,
        texture_predictor=None,
        bones_predictor=None,
        articulation_predictor=None,
        renderer=None,
        shape_template_path=None,
        view_option = "multi_view_azimu",
        fit_shape_template_inside_unit_cube=False,
        diffusion_Text_to_Target_Img=None,
        device = "cuda",
        correspond = None,
        # TODO: Create a view sampler class to encapsulate the view sampling logic and settings
        random_camera_radius= 2.5,  #[2.5, 2.5],
        phi_range_for_optim = [90,90],
        phi_range_for_visual = [0, 360],
        cyc_check_img_save = False,
        bones_rotations = "bones_rotations",
        debug_mode = False,
        using_pil_object = False,
        cyc_consi_check_switch = True,
        cyc_consi_check_dist_threshold = 15,
        seed = 50,
        target_image_fixed = False,
        save_individual_img = False,
        two_side_views_only = False, #Two side view poses along the azimuth
    ):
        super().__init__()
        self.path_to_save_images = path_to_save_images
        self.num_pose_for_optim = num_pose_for_optim
        self.num_pose_for_visual = num_pose_for_visual
        self.num_sample_bone_line = num_sample_bone_line
        self.num_sample_farthest_points = num_sample_farthest_points
        self.mode_kps_selection = mode_kps_selection
        self.enable_texture_predictor = enable_texture_predictor
        self.texture_predictor = (
            texture_predictor if texture_predictor is not None else TexturePredictor()
        )
        self.bones_predictor = (
            bones_predictor if bones_predictor is not None else BonesEstimator()
        )
        # Articulation predictor
        self.articulation_predictor = (articulation_predictor if articulation_predictor else ArticulationPredictor())
        self.renderer = renderer if renderer is not None else Renderer()
        self.view_option = view_option
        
        if shape_template_path is not None:
            self.shape_template = self._load_shape_template(shape_template_path, fit_inside_unit_cube=fit_shape_template_inside_unit_cube)    # False if self.view_option == "single_view" else True
        else:
            self.shape_template = None
        
        self.diffusion_Text_to_Target_Img = diffusion_Text_to_Target_Img if diffusion_Text_to_Target_Img is not None else DiffusionForTargetImg()
        self.device = device
        self.correspond = (correspond if correspond else ComputeCorrespond())
        self.random_camera_radius = random_camera_radius    # 1 if self.view_option == "single_view" else 2.5
        self.phi_range_for_optim = phi_range_for_optim
        self.phi_range_for_visual = phi_range_for_visual
        self.cyc_check_img_save = cyc_check_img_save
        self.bones_rotations = bones_rotations
        self.debug_mode = debug_mode
        self.using_pil_object = using_pil_object
        self.cyc_consi_check_switch = cyc_consi_check_switch
        self.cyc_consi_check_dist_threshold = cyc_consi_check_dist_threshold
        self.seed = random.seed(seed)
        self.target_image_fixed = target_image_fixed
        self.save_individual_img = save_individual_img
        self.two_side_views_only = two_side_views_only
        
        if debug_mode == False:
            # LOADING ODISE MODEL
            start_time = time.time()
            # 'diffusion_ver' options are v1-5, v1-3, v1-4, v1-5, v2-1-base
            # 'image_size' is for the sd input for the Fuse model i.e 960
            # 'timestep' for diffusion should be in the range [0, 1000], 0 for no noise added
            # 'block_indices' is selecting different layers from the UNet decoder for extracting sd features, only the first three are used by default.
            self.sd_model, self.sd_aug = load_model(
                config_path='Panoptic/odise_label_coco_50e.py',
                diffusion_ver='v1-5',
                image_size=960,
                num_timesteps=100,
                block_indices=(2, 5, 8, 11)
            )
            end_time = time.time()  # Record the end time
            print(f"The Fuse model loading took {end_time - start_time} seconds to run.\n")


    def _load_shape_template(self, shape_template_path, fit_inside_unit_cube=False):
        mesh = load_mesh(shape_template_path)
        # position the mesh inside the unit cube
        if fit_inside_unit_cube:
            mesh = mesh_utils.fit_inside_unit_cube(mesh)
        return mesh

    def compute_correspondences(
        self, articulated_mesh, pose, renderer, bones, rendered_mask, rendered_image, target_image
    ):
        # 1. Sample keypoints from the rendered image
        #    - find the closest visible point on the articulated_mesh in 3D (the visibility is done in 2D)
        #    - select the keypoints from the eroded mask
        #    - sample keypoints along the bone lines
        # 2. Find corresponding target keypoints using Fuse method. (TODO: some additional tricks e.g. optimal transport etc.)
        # 3. Compute cycle consistency check
        
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
        
        eroded_mask = self.mask_erode_tensor(rendered_mask)
            
        # SELECT KEYPOINT SAMPLING OPTION
        if self.mode_kps_selection == "kps_fr_sample_on_bone_line":
            kps_img_resolu = self.kps_fr_sample_on_bone_line(bones, mvp, articulated_mesh, visible_vertices, self.num_sample_bone_line, eroded_mask)
        elif self.mode_kps_selection == "kps_fr_sample_farthest_points":
            kps_img_resolu = self.kps_fr_sample_farthest_points(rendered_image, mvp, visible_vertices, articulated_mesh, eroded_mask, self.num_sample_farthest_points)
        
        output_dict = {}
        
        target_image_with_kps_list_after_cyc_check = []
        rendered_image_with_kps_list_after_cyc_check =[]
        cycle_consi_image_with_kps_list = []
        
        # rendered_image = nn_functional.interpolate(rendered_image, size=(840, 840), mode='bilinear', align_corners=False)
        # target_image = nn_functional.interpolate(target_image, size=(840, 840), mode='bilinear', align_corners=False)
        
        if self.cyc_consi_check_switch:
            start_time = time.time()
            rendered_image_with_kps_list, rendered_image_NO_kps_list, target_image_with_kps_list, target_image_NO_kps_list, corres_target_kps_tensor_stack, cycle_consi_image_with_kps_list, cycle_consi_kps_tensor_stack, rendered_target_image_with_wo_kps_list = self.correspond.compute_correspondences_sd_dino(img1_tensor=rendered_image, img1_kps=kps_img_resolu, img2_tensor=target_image, model=self.sd_model, aug=self.sd_aug, using_pil_object=self.using_pil_object)
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'compute_correspondences_sd_dino' took {end_time - start_time} seconds to run.\n")    
            print(f"The compute_correspondences_sd_dino function took {end_time - start_time} seconds to run.")
        else:
            start_time = time.time()
            rendered_image_with_kps_list, rendered_image_NO_kps_list, target_image_with_kps_list, target_image_NO_kps_list, corres_target_kps_tensor_stack, rendered_target_image_with_wo_kps_list = self.correspond.compute_correspondences_sd_dino(img1_tensor=rendered_image, img1_kps=kps_img_resolu, img2_tensor=target_image, model=self.sd_model, aug=self.sd_aug, using_pil_object=self.using_pil_object)
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'compute_correspondences_sd_dino' took {end_time - start_time} seconds to run.\n")    
            print(f"The compute_correspondences_sd_dino function took {end_time - start_time} seconds to run.")      
        
            
        # IF TRUE, REMOVE POINTS FOLLOWING CYCLE CONSISTENCY CHECK
        if self.cyc_consi_check_switch:
            # Calculate the squared difference along the coordinate dimension (dim=2)
            squared_diff = torch.pow(kps_img_resolu - cycle_consi_kps_tensor_stack, 2)

            # Sum the squared differences along the coordinate dimension to get the squared Euclidean distance
            squared_distances = torch.sum(squared_diff, dim=2)

            # Take the square root to get the Euclidean distance
            euclidean_distances = torch.sqrt(squared_distances)

            # Create a mask where the Euclidean distance is less than or equal to cyc_consi_check_dist_threshold
            mask = euclidean_distances <= self.cyc_consi_check_dist_threshold

            # Expanding mask to make it compatible for broadcasting
            mask_expanded = mask.unsqueeze(-1) 

            # Use the mask to zero out points
            # We multiply the mask with the keypoints. However, since the mask is Boolean, we first need to convert it to the same dtype as kps_img_resolu
            kps_img_resolu_filtered = kps_img_resolu * mask_expanded.to(kps_img_resolu.dtype)

            # Update the Target kps after CYCLE CONSISTENCY CHECK
            corres_target_kps_filtered = corres_target_kps_tensor_stack * mask_expanded.to(corres_target_kps_tensor_stack.dtype)

            kps_img_resolu = kps_img_resolu_filtered
            corres_target_kps_tensor_stack = corres_target_kps_filtered
        
            # IF TRUE THEN SAVE CYCLE CONSISTENCY IMAGES
            if self.cyc_check_img_save:
                for index in range(kps_img_resolu.shape[0]):
                    # rendered_image_PIL.save(f'{self.path_to_save_images}/{index}_rendered_image_PIL.png', bbox_inches='tight')
                    # target_image_PIL.save(f'{self.path_to_save_images}/{index}_target_image_PIL.png', bbox_inches='tight')
                    rendered_image_PIL = torchvision.transforms.functional.to_pil_image(rendered_image[index])
                    rendered_image_PIL = resize(rendered_image_PIL, 840, resize=True, to_pil=True)
                    target_image_PIL = torchvision.transforms.functional.to_pil_image(target_image[index])
                    target_image_PIL = resize(target_image_PIL, 840, resize=True, to_pil=True)
                    
                    rendered_image_with_kps_cyc_check = draw_correspondences_1_image(kps_img_resolu_filtered[index], rendered_image_PIL) #, color='yellow')              #[-6:]
                    target_image_with_kps_cyc_check = draw_correspondences_1_image(corres_target_kps_filtered[index], target_image_PIL) #, color='yellow')              #[-6:]

                    rendered_image_with_kps_list_after_cyc_check.append(rendered_image_with_kps_cyc_check)
                    target_image_with_kps_list_after_cyc_check.append(target_image_with_kps_cyc_check)

                    # SAVE CYCLE CONSISTENCY IMAGES
                    self.save_cyc_consi_check_images(cycle_consi_image_with_kps_list[index], rendered_image_with_kps_cyc_check, target_image_with_kps_cyc_check, index)
        
            
        output_dict = {
        "rendered_kps": kps_img_resolu,                     
        "target_corres_kps": corres_target_kps_tensor_stack,         
        "rendered_image_with_kps": rendered_image_with_kps_list,
        "target_image_with_kps": target_image_with_kps_list,
        "target_img_NO_kps": target_image_NO_kps_list,
        "rendered_img_NO_kps": rendered_image_NO_kps_list,
        "cycle_consi_image_with_kps": cycle_consi_image_with_kps_list,
        "rendered_image_with_kps_list_after_cyc_check": rendered_image_with_kps_list_after_cyc_check,
        "target_image_with_kps_list_after_cyc_check": target_image_with_kps_list_after_cyc_check,
        "rendered_target_image_with_wo_kps_list": rendered_target_image_with_wo_kps_list,
        }        
        
        ## Saving multiple random poses with and without keypoints visualisation
        self.save_multiple_random_poses(output_dict, self.path_to_save_images)
        
        return output_dict

    
    def forward(self, batch):
        
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
        
        if self.bones_rotations == "bones_rotations":
            start_time = time.time()
            # BONE ROTATIONS
            bones_rotations = self.articulation_predictor(batch, num_bones)
            end_time = time.time()  # Record the end time
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'articulation_predictor' took {end_time - start_time} seconds to run.\n")
            print(f"The articulation_predictor function took {end_time - start_time} seconds to run.")
            
        elif self.bones_rotations == "NO_bones_rotations":
            # NO BONE ROTATIONS
            bones_rotations = torch.zeros(batch_size, num_bones, 3, device=mesh.v_pos.device)
        
        elif self.bones_rotations == "DUMMY_bones_rotations":
            # DUMMY BONE ROTATIONS - pertrub the bones rotations (only to test the implementation)
            bones_rotations = bones_rotations + torch.randn_like(bones_rotations) * 0.1
    
    
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
            print(f"The function took {end_time - start_time} seconds to run.\n")
        else:
            # if texture predictor is not enabled, use the loaded material from the mesh
            material = mesh.material

        # if pose not provided, compute it from the camera matrix
        if "pose" not in batch:
            if self.view_option == "single_view":
                # For Single View
                # pose shape is [1,12]
                pose = geometry_utils.blender_camera_matrix_to_magicpony_pose(
                    batch["camera_matrix"]
                )
            elif self.view_option == "multi_view_rand":
                # For Multi View
                # pose shape is [num_pose, 12]
                pose, _ = multi_view.rand_poses(self.num_pose_for_optim, self.device, radius_range=self.random_camera_radius)
                
            elif self.view_option == "multi_view_azimu":
                pose, _ = multi_view.poses_along_azimuth(self.num_pose_for_optim, self.device, radius=self.random_camera_radius, phi_range=self.phi_range_for_optim, two_side_views_only=self.two_side_views_only)
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
            im_features=texture_features
            # CHANGED IT
            # background=background,
        )

        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'renderer' took {end_time - start_time} seconds to run.\n")
        print(f"The renderer function took {end_time - start_time} seconds to run.")
        
        # Creates an empty tensor to hold the final result
        # all_generated_target_img shape is [num_pose, 3, 256, 256]
        
        # all_generated_target_img = {}
        
        # For Debugging purpose, save all the poses before optimisation
        self.save_all_poses_before_optimisation(pose, renderer_outputs, self.path_to_save_images)
        # GENERATING TARGET IMAGES USING DIFFUSION (SD or DF)
        target_img_rgb = self.diffusion_Text_to_Target_Img.run_experiment(
            input_image=renderer_outputs["image_pred"],
            image_fr_path=False,
        )
        
        # # Inserts the new image into a dictionary
        # # all_generated_target_img["target_img_NO_kps"].shape is [1, 3, 256, 256]
        # all_generated_target_img["target_img_NO_kps"] = target_img_rgb
        
        # Inserts the new image into the final tensor
        # resizes the image to the target resolution
        target_img_rgb = torch.nn.functional.interpolate(target_img_rgb, size=renderer_outputs["image_pred"].shape[2:], mode='bilinear', align_corners=False)

        for i in range(pose.shape[0]):
            # target_img_rgb.shape is [1, 3, 256, 256]
            target_image_PIL = F.to_pil_image(target_img_rgb[0])
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            
            dir_path = f'{self.path_to_save_images}/diff_pose/target_img/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            target_image_PIL.save(f'{dir_path}{i}_diff_pose_target_image.png', bbox_inches='tight')
        
        print('target_img_rgb.shape', target_img_rgb.shape)
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
            target_image = batch["image"] if self.target_image_fixed else target_img_rgb,                  # batch["image"] is fixed Target images (generated from SD), target_img_rgb randomly generated per iteration
        )

        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'compute_correspondences' took {end_time - start_time} seconds to run.\n")
        print(f"The compute_correspondences took {end_time - start_time} seconds to run.")
        
        outputs = {}
        # TODO: probaly rename the ouputs of the renderer
        
        # outputs.update(target_img_rgb)
        outputs.update(renderer_outputs)        # renderer_outputs keys are dict_keys(['image_pred', 'mask_pred', 'albedo', 'shading'])
        outputs.update(correspondences_dict)

        ## Saving poses along the azimuth
        self.save_pose_along_azimuth(articulated_mesh, material, self.path_to_save_images)      
        
        return outputs

    def get_metrics_dict(self, model_outputs, batch):
        return {}

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        
        # Keypoint loss
        # Computes the loss between the source and target keypoints
        print('Calculating l2 loss')
        # loss = nn_functional.mse_loss(rendered_keypoints, target_keypoints, reduction='mean')
        model_outputs["rendered_kps"] = model_outputs["rendered_kps"].to(self.device)
        model_outputs["target_corres_kps"] = model_outputs["target_corres_kps"].to(self.device)

        loss = nn_functional.mse_loss(model_outputs["rendered_kps"], model_outputs["target_corres_kps"], reduction='mean')
       
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
    
    ## Saving all poses
    def save_multiple_random_poses(self, model_outputs, path_to_save_images):
        
        for index, item in enumerate(model_outputs["rendered_image_with_kps"]):
        
            if self.save_individual_img:
                # With KPs visualisation - these images are Matplotlib Object
                if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_img_with_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_rendered_img_with_KPs')
                plt.gcf().set_facecolor('grey')
                model_outputs["rendered_image_with_kps"][index].savefig(f'{path_to_save_images}/all_poses_rendered_img_with_KPs/{index}_poses_rendered_img_with_KPs.png', bbox_inches='tight')
                if not os.path.exists(f'{path_to_save_images}/all_poses_target_img_with_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_target_img_with_KPs')
                plt.gcf().set_facecolor('grey')
                model_outputs["target_image_with_kps"][index].savefig(f'{path_to_save_images}/all_poses_target_img_with_KPs/{index}_poses_target_img_with_KPs.png', bbox_inches='tight')
                
                # Without KPs visualisation - these images are PIL Object
                if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs')
                model_outputs["rendered_img_NO_kps"][index].save(f'{path_to_save_images}/all_poses_rendered_img_NO_KPs/{index}_poses_rendered_img_NO_KPs.png', bbox_inches='tight')
                
                if not os.path.exists(f'{path_to_save_images}/all_poses_target_img_NO_KPs'):
                    os.makedirs(f'{path_to_save_images}/all_poses_target_img_NO_KPs')
                model_outputs["target_img_NO_kps"][index].save(f'{path_to_save_images}/all_poses_target_img_NO_KPs/{index}_poses_target_img_NO_KPs.png', bbox_inches='tight')
                
            if not os.path.exists(f'{path_to_save_images}/all_poses_rendered_target_combined'):
                os.makedirs(f'{path_to_save_images}/all_poses_rendered_target_combined')
            model_outputs["rendered_target_image_with_wo_kps_list"][index].save(f'{path_to_save_images}/all_poses_rendered_target_combined/{index}_all_poses_rendered_target_combined.png', bbox_inches='tight')
            

    ## Saving poses along the azimuth
    def save_pose_along_azimuth(self, articulated_mesh, material, path_to_save_images):
        
        if self.view_option == "single_view":
            # Added for debugging purpose
            pose, _ = multi_view.poses_along_azimuth_single_view(self.num_pose_for_visual, device=self.device)
        else:
            pose, _ = multi_view.poses_along_azimuth(self.num_pose_for_visual, device=self.device, radius=self.random_camera_radius, phi_range=self.phi_range_for_visual)
        
        renderer_outputs = self.renderer(
            articulated_mesh,
            material= material,
            pose=pose,
            im_features= None
        )
        
        for i in range(pose.shape[0]):
            rendered_image_PIL = F.to_pil_image(renderer_outputs["image_pred"][i])
            rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            dir_path = f'{path_to_save_images}/azimuth_pose/rendered_img/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            rendered_image_PIL.save(f'{dir_path}{i}_azimuth_pose_rendered_image.png', bbox_inches='tight')
            
            
    # Saving Rendered Image at every iteration with keypoints visualisation
    def save_img_each_iteration(self, model_outputs, iteration, index_of_image, path_to_save_img_per_iteration):
        
        start_time = time.time()
        
        for index, item in enumerate(model_outputs["rendered_image_with_kps"]):
            
            if self.save_individual_img:  
                # This image is a Matplotlib Object
                dir_path = f'{path_to_save_img_per_iteration}/rendered_img_with_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                plt.gcf().set_facecolor('grey')
                model_outputs["rendered_image_with_kps"][index].savefig(f'{dir_path}/{iteration}_rendered_img_with_kps.png', bbox_inches='tight')

                # This image is a PIL Object
                dir_path = f'{path_to_save_img_per_iteration}/rendered_img_NO_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["rendered_img_NO_kps"][index].save(f'{dir_path}/{iteration}_rendered_img_NO_kps.png', bbox_inches='tight')

                # This image is a Matplotlib Object
                plt.gcf().set_facecolor('grey')
                dir_path = f'{path_to_save_img_per_iteration}/target_img_with_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["target_image_with_kps"][index].savefig(f'{dir_path}/{iteration}_target_img_with_kps.png', bbox_inches='tight')
                
                # This image is a PIL Object
                dir_path = f'{path_to_save_img_per_iteration}/target_img_NO_kps/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                model_outputs["target_img_NO_kps"][index].save(f'{dir_path}/{iteration}_target_img_NO_kps.png', bbox_inches='tight')
                
            dir_path = f'{path_to_save_img_per_iteration}/rendered_target_image_with_wo_kps_list/{index}_pose'
            os.makedirs(dir_path, exist_ok=True)
            model_outputs["rendered_target_image_with_wo_kps_list"][index].save(f'{dir_path}/{iteration}_rendered_target_image_with_wo_kps_list.png', bbox_inches='tight')
            
            
            if (self.cyc_consi_check_switch & self.cyc_check_img_save):
                
                # This image is a Matplotlib Object
                dir_path = f'{path_to_save_img_per_iteration}/cycle_consi/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                plt.gcf().set_facecolor('grey')
                model_outputs["cycle_consi_image_with_kps"][index].savefig(f'{dir_path}/{iteration}_cycle_consi.png', bbox_inches='tight')

                # This image is a Matplotlib Object
                dir_path = f'{path_to_save_img_per_iteration}/rendered_image_with_kps_list_after_cyc_check/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                plt.gcf().set_facecolor('grey')
                model_outputs["rendered_image_with_kps_list_after_cyc_check"][index].savefig(f'{dir_path}/{iteration}_rendered_image_with_kps_list_after_cyc_check.png', bbox_inches='tight')

                # This image is a Matplotlib Object
                dir_path = f'{path_to_save_img_per_iteration}/target_image_with_kps_list_after_cyc_check/{index}_pose'
                os.makedirs(dir_path, exist_ok=True)
                plt.gcf().set_facecolor('grey')
                model_outputs["target_image_with_kps_list_after_cyc_check"][index].savefig(f'{dir_path}/{iteration}_target_image_with_kps_list_after_cyc_check.png', bbox_inches='tight')

            end_time = time.time()  # Record the end time
            print(f"The 'Saving img for every iterations' took {end_time - start_time} seconds to run.")
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'Saving img for every iterations' took {end_time - start_time} seconds to run.\n")
    
    
    # For Debugging purpose, save all the poses before optimisation
    def save_all_poses_before_optimisation(self, pose, renderer_outputs, path_to_save_images):    
        for i in range(pose.shape[0]):
            rendered_image_PIL = F.to_pil_image(renderer_outputs["image_pred"][i])
            #rendered_image_PIL = resize(rendered_image_PIL, target_res = 840, resize=True, to_pil=True)
            dir_path = f'{self.path_to_save_images}/diff_pose/rendered_img/'
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            # Save the image
            rendered_image_PIL.save(f'{dir_path}{i}_diff_pose_rendered_image.png', bbox_inches='tight')
            
    # Keypoint selection using sampling points on the bone line.
    def kps_fr_sample_on_bone_line(self, bones, mvp, articulated_mesh, visible_vertices, num_sample_bone_line, eroded_mask):
            
        bone_end_pt_1_3D = bones[:, :, 0, :]  # one end of the bone in 3D
        bone_end_pt_2_3D = bones[:, :, 1, :]  # other end of the bone in 3D
        
        bones_in_3D_all_kp40 = torch.cat((bone_end_pt_1_3D, bone_end_pt_2_3D), dim=1)
        bones_2D_proj_all_kp40 = geometry_utils.project_points(bones_in_3D_all_kp40, mvp)
        
        bone_end_pt_1_projected_in_2D = geometry_utils.project_points(bone_end_pt_1_3D, mvp)
        bone_end_pt_2_projected_in_2D = geometry_utils.project_points(bone_end_pt_2_3D, mvp)
        
        bones_midpts_in_3D = (bones[:, :, 0, :] + bones[:, :, 1, :]) / 2.0        # This is in 3D the shape is torch.Size([2, 20, 3])
        
        # SAMPLE POINTS ON BONE LINE
        bones_midpts_in_3D = self.sample_points_on_line(bone_end_pt_1_3D, bone_end_pt_2_3D, num_sample_bone_line)
        
        bones_midpts_projected_in_2D = geometry_utils.project_points(bones_midpts_in_3D, mvp)
    
        start_time = time.time()
        closest_midpts = self.closest_visible_points(bones_midpts_in_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
        
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
        
        start_time = time.time()
        ## get_vertices_inside_mask
        vertices_inside_mask = self.get_vertices_inside_mask(pixel_projected_visible_v_in_2D, eroded_mask)
        
        end_time = time.time()  # Record the end time
        # with open('log.txt', 'a') as file:
        #     file.write(f"The 'get_vertices_inside_mask' took {end_time - start_time} seconds to run.\n")
        
        print(f"The get_vertices_inside_mask function took {end_time - start_time} seconds to run.")
    
        kps_img_resolu = (vertices_inside_mask/256) * 840
        
        bone_end_pt_1 = self.closest_visible_points(bone_end_pt_1_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
        bone_end_pt_2 = self.closest_visible_points(bone_end_pt_2_3D, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
        
        # bone_end_pt_1_in_2D_cls = geometry_utils.project_points(bone_end_pt_1, mvp)
        # bone_end_pt_2_in_2D_cls = geometry_utils.project_points(bone_end_pt_2, mvp)
        
        # bones_mid_pt_in_2D = (bone_end_pt_1_in_2D_cls + bone_end_pt_2_in_2D_cls) / 2.0
        
        bones_all = torch.cat((bone_end_pt_1, bone_end_pt_2), dim=1)
        
        bones_all = self.closest_visible_points(bones_all, articulated_mesh.v_pos, visible_vertices) # , eroded_mask)
        # bones_closest_pts_2D_proj_all_kp40 = geometry_utils.project_points(bones_all, mvp)
        
        return kps_img_resolu
        
    
    def kps_fr_sample_farthest_points(self, rendered_image, mvp, visible_vertices, articulated_mesh, eroded_mask, num_samples):
            
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
        
        # num_samples = 100
        visible_v_position = geometry_utils.sample_farthest_points(visible_v_position, num_samples)
        visible_v_position = visible_v_position.permute(0,2,1)
        
        projected_visible_v_in_2D = geometry_utils.project_points(visible_v_position, mvp)  
    
        # Convert to Pixel Coordinates of the mask
        pixel_projected_visible_v_in_2D = (projected_visible_v_in_2D + 1) * eroded_mask.size(1)/2
        vertices_inside_mask = self.get_vertices_inside_mask(pixel_projected_visible_v_in_2D, eroded_mask)
        kps_img_resolu = (vertices_inside_mask/256) * 840
        
        # project vertices/keypoints example
        # mesh.v_pos shape is torch.Size([2, 31070, 3])
        # mvp.shape is torch.Size([Batch size, 4, 4])
        # projected_vertices.shape is torch.Size([4, 31070, 2]), # mvp is model-view-projection
        projected_vertices = geometry_utils.project_points(articulated_mesh.v_pos, mvp)  
        
        projected_vertices = projected_vertices[:, :num_samples, :]
        kps_img = projected_vertices[:,:,:][0] * rendered_image.shape[2]
        
        return kps_img_resolu
    

    def closest_visible_points(self, bones_midpts, mesh_v_pos, visible_vertices):
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

    
    def mask_erode_tensor(self, batch_of_masks):

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

    def get_vertices_inside_mask(self, projected_visible_v_in_2D, eroded_mask):
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

    def sample_points_on_line(self, pt1, pt2, num_samples):
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


    def save_cyc_consi_check_images(self, cycle_consi_image_with_kps, rendered_image_with_kps_cyc_check, target_image_with_kps_cyc_check, index):
        # # Set the background color to grey
        # plt.gcf().set_facecolor('grey')
        # plt.text(80, 0.95, f'Cycle Consistency', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
        cycle_consi_image_with_kps.savefig(f'{self.path_to_save_images}/{index}_cycle.png', bbox_inches='tight')
        plt.text(30, 0.95, f'Final Img after Eroded Mask & Cycle Consi Check', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
        ## For now commented Loss printout
        ## plt.text(80, 40, f'Loss: {loss}', verticalalignment='top', horizontalalignment='left', color = 'orange', fontsize ='11')
        rendered_image_with_kps_cyc_check.savefig(f'{self.path_to_save_images}/{index}_rendered_image_with_kps_after_cyclic_check.png', bbox_inches='tight')
        # # Set the background color to grey
        # plt.gcf().set_facecolor('grey')    
        target_image_with_kps_cyc_check.savefig(f'{self.path_to_save_images}/{index}_target_image_with_kps_after_cyclic_check.png', bbox_inches='tight')
        plt.close()