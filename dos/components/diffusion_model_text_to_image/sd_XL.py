"""
Based on https://github.com/fudan-zvg/PGC-3D/blob/main/guidance/sdxl.py
"""

import os
from typing import List, Tuple, Union

from diffusers import (AutoencoderKL, DDIMScheduler, IFPipeline, PNDMScheduler,
                       StableDiffusionPipeline, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, logging

# suppress partial model loading warning
logging.set_verbosity_error()
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import custom_bwd, custom_fwd

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (gt_grad,) = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusionXL(nn.Module):
    def __init__(self, device, cache_dir=None, hf_key=None, torch_dtype=torch.float16):
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.model_key: str = "stabilityai/stable-diffusion-xl-base-1.0"  #
        self.enable_channels_last_format = True

        print(f"Loading Stable Diffusion XL...")

        # The default VAE model is not compatible with fp16, so we need to load a custom one
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            # "pretrained/SDXL/vae_fp16",
            # local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float16,
        )

        # Create model
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_key,
            vae=vae,
            variant="fp16",
            torch_dtype=torch.float16,  #
            cache_dir=cache_dir,
        ).to(self.device)

        if self.enable_channels_last_format:
            self.pipeline.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipeline.unet

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipeline.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        # added for DDS loss
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(
            self.device, dtype=self.torch_dtype
        )
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.t_min = 50
        self.t_max = 950
        self.prediction_type = self.pipeline.scheduler.prediction_type
        print(f"Loaded Stable Diffusion XL!")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        """
        prompt, negative_prompt: [str]

        Based on https://github.com/fudan-zvg/PGC-3D/blob/main/guidance/sdxl.py
        """

        # Define tokenizers and text encoders
        tokenizers = (
            [self.pipeline.tokenizer, self.pipeline.tokenizer_2]
            if self.pipeline.tokenizer is not None
            else [self.pipeline.tokenizer_2]
        )
        text_encoders = (
            [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
            if self.pipeline.text_encoder is not None
            else [self.pipeline.text_encoder_2]
        )

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                prompt_embeds = text_encoder(
                    text_inputs.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Do the same for unconditional embeddings
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    # equivalent to UNET_AttentionBlock code in 'pytorch-stable-diffusion' diffusion.py file
    # FIXME: loss_weight not implemented,
    def train_step(
        self,
        text_embeddings,
        pred_rgb=None,
        latents=None,
        guidance_scale=100,
        loss_weight=1.0,
        min_step_pct=0.02,
        max_step_pct=0.98,
        return_aux=False,
        fixed_step=None,
        noise_random_seed=None,
    ):
        """
        Based on https://github.com/fudan-zvg/PGC-3D/blob/main/guidance/sdxl.py
        """

        text_embeddings = [x.to(self.torch_dtype) for x in text_embeddings]
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        if latents is None:
            b = pred_rgb.size(0)
            pred_rgb_1024 = F.interpolate(
                pred_rgb, (1024, 1024), mode="bilinear", align_corners=False
            )
            latents = self.encode_imgs(pred_rgb_1024)
        else:
            b = latents.size(0)

        add_text_embeds = pooled_prompt_embeds
        res = 1024  # if self.opt.latent else self.opt.res_fine
        add_time_ids = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds.dtype
        ).repeat_interleave(b, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if fixed_step is None:
            min_step = int(self.num_train_timesteps * min_step_pct)
            max_step = int(self.num_train_timesteps * max_step_pct)
            t = torch.randint(
                min_step, max_step + 1, [b], dtype=torch.long, device=self.device
            )
        else:
            t = torch.zeros([b], dtype=torch.long, device=self.device) + fixed_step

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if noise_random_seed is not None:
                torch.manual_seed(noise_random_seed)
                torch.cuda.manual_seed(noise_random_seed)

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise - latent_model_input shape is torch.Size([2, 256, 64, 64]) this is wrong should be [B, 3, 64, 64]
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            t_input = torch.cat([t, t])

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                latent_model_input,
                t_input,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # perform guidance (high scale from paper!)
        # THIS DOES THE CLASSIFIER-FREE GUIDANCE
        # THE OUTPUT IS SPLITTED IN TWO PARTS, ONE FOR CONDITIONED-ON-TEXT AND ANOTHER ONE FOR UNCONDITIONED-ON-TEXT outputs.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1).to(self.torch_dtype)

        grad_unweighted = noise_pred - noise

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / b

        if return_aux:
            aux = {
                "grad": grad,
                "grad_unweighted": grad_unweighted,
                "t": t,
                "w": w,
                "latents": latents,
            }
            return loss, aux
        else:
            return loss

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.pipeline.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents
        self.pipeline.vae.config.scaling_factor
        with torch.no_grad():
            imgs = self.pipeline.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
