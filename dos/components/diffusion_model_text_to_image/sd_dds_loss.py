import os
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers import IFPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline
from typing import Union, List, Tuple
from transformers import T5EncoderModel
# suppress partial model loading warning
logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd 
from typing import Tuple, Union, Optional, List
import numpy as np
from PIL import Image

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
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusionDDSLoss(nn.Module):
    def __init__(self, device, cache_dir=None, hf_key=None, torch_dtype=torch.float16):
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.model_key: str = "runwayml/stable-diffusion-v1-5"  ## "stabilityai/stable-diffusion-xl-base-1.0"
        self.enable_channels_last_format = True
        
        print(f"Loading Stable Diffusion ...")
        # Create model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_key,
            variant="fp16",
            torch_dtype= torch.float16,  #
            cache_dir=cache_dir
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
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(self.device, dtype=self.torch_dtype)  
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.t_min = 50
        self.t_max = 950
        self.prediction_type = self.pipeline.scheduler.prediction_type
        print(f'Loaded Stable Diffusion!')

    
    @torch.no_grad()
    def get_text_embeds(self, prompt_source, negative_prompt, prompt_target):
        
        tokens = self.pipeline.tokenizer([prompt_source], padding="max_length", max_length=77, truncation=True,
                                    return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        
        cond_text_embeddings = self.pipeline.text_encoder(tokens).last_hidden_state.detach()
        
        tokens = self.pipeline.tokenizer([negative_prompt], padding="max_length", max_length=77, truncation=True,
                                    return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        
        uncond_text_embeddings = self.pipeline.text_encoder(tokens).last_hidden_state.detach()
        
        tokens = self.pipeline.tokenizer([prompt_target], padding="max_length", max_length=77, truncation=True,
                                    return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        
        target_text_embeddings = self.pipeline.text_encoder(tokens).last_hidden_state.detach()
        
        text_embedding_source = torch.stack([uncond_text_embeddings, cond_text_embeddings], dim=1)
        text_embedding_target = torch.stack([uncond_text_embeddings, target_text_embeddings], dim=1)
        
        return text_embedding_source, text_embedding_target

    
    # equivalent to UNET_AttentionBlock code in 'pytorch-stable-diffusion' diffusion.py file
    def train_step(
            self, text_embeddings, pred_rgb=None, latents=None, guidance_scale=100, loss_weight=1.0, min_step_pct=0.02, 
            max_step_pct=0.98, return_aux=False, fixed_step=None, noise_random_seed=None):
        
        # text_embeddings shape torch.Size([2, 77, 4096])
        text_embeddings = text_embeddings.to(self.torch_dtype)

        if latents is None:
            pred_rgb = pred_rgb.to(self.torch_dtype)
            b = pred_rgb.shape[0]
            # interp to 512x512 to be fed into vae.
            # _t = time.time()
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')
            # encode image into latents with vae, requires grad!
            # _t = time.time()
            latents = self.encode_imgs(pred_rgb_512)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')
        else:
            b = latents.shape[0]
        
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if fixed_step is None:
            min_step = int(self.num_train_timesteps * min_step_pct)
            max_step = int(self.num_train_timesteps * max_step_pct)
            t = torch.randint(min_step, max_step + 1, [b], dtype=torch.long, device=self.device)
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
            
            noise_pred = self.unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings
            )  # (2B, 6, 64, 64)
        
        # perform guidance (high scale from paper!)
        # THIS DOES THE CLASSIFIER-FREE GUIDANCE
        # THE OUTPUT IS SPLITTED IN TWO PARTS, ONE FOR CONDITIONED-ON-TEXT AND ANOTHER ONE FOR UNCONDITIONED-ON-TEXT outputs.        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        
        grad_unweighted = (noise_pred - noise)
        
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
            aux = {'grad': grad, 'grad_unweighted': grad_unweighted, 't': t, 'w': w, 'latents': latents}
            return loss, aux
        else:
            return loss
        
        
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.pipeline.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
          
    @torch.no_grad()
    def denormalize(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image[0]
    
    @torch.no_grad()
    def decode_latents(self, latent: T, im_cat: TN = None):
        image = self.pipeline.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
        image = self.denormalize(image)
        if im_cat is not None:
            image = np.concatenate((im_cat, image), axis=1)
        return Image.fromarray(image)


    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, get_raw=False,
                           guidance_scale=7.5):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0


    def get_dds_loss(self, z_source, z_target, text_emb_source, text_emb_target,
                            eps=None, reduction='mean', symmetric = False, calibration_grad=None, timestep = None,
                      guidance_scale=7.5, raw_log=False):
        
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(z_source, eps, timestep)
            z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
            eps_pred, _ = self.get_eps_prediction(torch.cat((z_t_source, z_t_target)),
                                                  torch.cat((timestep, timestep)),
                                                  torch.cat((text_emb_source, text_emb_target)),
                                                  torch.cat((alpha_t, alpha_t)),
                                                  torch.cat((sigma_t, sigma_t)),
                                                  guidance_scale=guidance_scale)
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source)
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timestep - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()
        loss = z_target * grad.clone()
        if symmetric:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss
    
    
    def load_512(self, image, image_path=None, left=0, right=0, top=0, bottom=0):
        # If loading from image path
        # image = np.array(Image.open(image_path))[:, :, :3]    
        
        # Convert tensor to numpy array and transpose from [C, H, W] to [H, W, C]
        image = image.permute(1, 2, 0).cpu().detach().numpy()
        
        # Ensure image is in [0, 255] range if it's normalized between [0, 1]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        
        # Crop the image based on the provided margins
        image = image[top:h-bottom, left:w-right]
        
        h, w, c = image.shape
        
        # Center crop to square if necessary
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w, :]
            
        image = np.array(Image.fromarray(image).resize((512, 512)))
        return image