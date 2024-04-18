# Code mainly taken from - https://github.com/bytedance/MVDream-threestudio/blob/main/threestudio/models/guidance/multiview_diffusion_guidance.py
import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from dos.utils.utils import C

from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
# from dos.components.diffusion_model_text_to_image.base import PromptProcessor

class MultiviewDiffusionGuidance(nn.Module):
    def __init__(self, device, cache_dir=None, hf_key=None, torch_dtype=torch.float16):
        super().__init__()
        
        self.device = device
        self.model_name = "sd-v2.1-base-4view" # check mvdream.model_zoo.PRETRAINED_MODELS
        self.ckpt_path = None # path to local checkpoint (None for loading from url)
        # self.guidance_scale = 50.0
        self.grad_clip = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        self.half_precision_weights = True
        self.min_step_percent = 0.02
        self.max_step_percent = 0.98
        self.camera_condition_type = "rotation"
        self.view_dependent_prompting = False
        self.n_view = 1 # 4
        self.image_size = 256
        self.recon_loss = True
        self.recon_std_rescale = 0.5

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.num_train_timesteps = 1000
        min_step_percent = C(self.min_step_percent, 0, 0)
        max_step_percent = C(self.max_step_percent, 0, 0)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)
        
        # Instantiate PromptProcessor Class
        # self.prompt_processor = PromptProcessor()

        # # call the configure of the PromptProcessor Class
        # self.prompt_processor.configure()

        # Use the __call__ method to get a PromptProcessorOutput instance
        # self.prompt_utils = self.prompt_processor()
        
        print(f"Loaded Multiview Stable Diffusion - MVDream!")

    def get_camera_cond(self, 
            camera: Float[Tensor, "B 4 4"],
            fovy = None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.camera_condition_type == "rotation": # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.camera_condition_type}")
        return camera

    def encode_imgs(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image
    

    def train_step(
        self,
        rgb,
        text_embeddings,
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy = None,
        fixed_step=None,
        input_is_latent=True,
        guidance_scale=50,
        return_aux=False,
        use_nfsd=False,
        # **kwargs,
    ):
        
        batch_size = rgb.shape[0]
        camera = c2w

        # rgb_BCHW = rgb.permute(0, 3, 1, 2) # 
        rgb_BCHW = rgb 
        
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(rgb_BCHW, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if fixed_step is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert fixed_step >= 0 and fixed_step < self.num_train_timesteps
            t = torch.full([1], fixed_step, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2,1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.n_view}
            else:
                context = {"context": text_embeddings}
            
            # import ipdb; ipdb.set_trace()
            # latent_model_input.shape [2, 4, 64, 64]
            # t_input shape [2]
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t])
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.min_step_percent, epoch, global_step)
        max_step_percent = C(self.max_step_percent, epoch, global_step)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        