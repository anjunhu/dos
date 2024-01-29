##------- CODE partly taken from  https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/deep_floyd_guidance.py

import os
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers import IFPipeline
from typing import Union, List, Tuple
from transformers import T5EncoderModel

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

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


class DeepFloyd(nn.Module):
    def __init__(self, device, cache_dir=None, hf_key=None, torch_dtype=torch.float16):
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
        # self.weights_dtype = torch.float16
       
        self.enable_channels_last_format = True
        
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto" # this makes it memory efficient, helps to avoid getting CUDA out-of-memory error.
        ) 
        
        print(f"Loading Deep Floyd ...")
        # Create model
        self.pipe = IFPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            variant="fp16",
            torch_dtype= torch.float16  #
        ).to(self.device)


        if self.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None
        
        print(f'Loaded Deep Floyd!')

    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.torch_dtype),
            t.to(self.torch_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.torch_dtype),
        ).sample.to(input_dtype)


    # Added
    def get_text_embeds(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ):  # -> Tuple[Float[Tensor, "B 77 4096"], Float[Tensor, "B 77 4096"]]
        text_embeddings, uncond_text_embeddings = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device=self.device
        )
        # return text_embeddings, uncond_text_embeddings
        return torch.cat([uncond_text_embeddings, text_embeddings])
    

    # equivalent to UNET_AttentionBlock code in 'pytorch-stable-diffusion' diffusion.py file
    def train_step(
            self, text_embeddings, pred_rgb=None, latents=None, guidance_scale=100, loss_weight=1.0, min_step_pct=0.02, 
            max_step_pct=0.98, return_aux=False, fixed_step=None, noise_random_seed=None):
        
        # text_embeddings shape torch.Size([2, 77, 4096])
        text_embeddings = text_embeddings.to(self.torch_dtype)

        rgb_BCHW = pred_rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )
        
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
            
            noise_pred = self.forward_unet(
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
        
