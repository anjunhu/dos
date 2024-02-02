##------- CODE taken from https://github.com/tomasjakab/laam/blob/sds-investigation/dos/video3d/diffusion/sd.py. 

import os
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

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

class StableDiffusion(nn.Module):
    def __init__(self, device, cache_dir=None, sd_version='1.5', hf_key=None, torch_dtype=torch.float32):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.torch_dtype = torch_dtype

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        
        print(f'[INFO] loading stable diffusion {model_key}')

        # Create model
        # cache_dir="/scratch/local/ssd/tomj/cache/huggingface_hub"
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch_dtype, cache_dir=cache_dir).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", cache_dir=cache_dir).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch_dtype, cache_dir=cache_dir).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", cache_dir=cache_dir)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        # added for DDS loss
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(self.device, dtype=self.torch_dtype)  
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.t_min = 50
        self.t_max = 950

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            # Shape of "text_input.input_ids" is [1, 77]
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # equivalent to UNET_AttentionBlock code in 'pytorch-stable-diffusion' diffusion.py file
    def train_step(
            self, text_embeddings, pred_rgb=None, latents=None, guidance_scale=100, loss_weight=1.0, min_step_pct=0.02, 
            max_step_pct=0.98, return_aux=False, fixed_step=None, noise_random_seed=None):
        text_embeddings = text_embeddings.to(self.torch_dtype)

        #
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
            print('fixed_step is None')
            min_step = int(self.num_train_timesteps * min_step_pct)
            max_step = int(self.num_train_timesteps * max_step_pct)
            t = torch.randint(min_step, max_step + 1, [b], dtype=torch.long, device=self.device)
        else:
            print('fixed_step is not None')
            t = torch.zeros([b], dtype=torch.long, device=self.device) + fixed_step


        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if noise_random_seed is not None:
                torch.manual_seed(noise_random_seed)
                torch.cuda.manual_seed(noise_random_seed)
            
            # noise shape [1, 4, 64, 64]
            noise = torch.randn_like(latents)
            
            # latents_noisy shape [1, 4, 64, 64]
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            t_input = torch.cat([t, t])
            
            # text_embeddings shape [2, 77, 768]
            # t_input shape [2]
            # latent_model_input shape [2, 4, 64, 64]
            # noise_pred shape [2, 4, 64, 64]
            noise_pred = self.unet(latent_model_input, t_input, encoder_hidden_states=text_embeddings).sample
        
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')
            
        # perform guidance (high scale from paper!)
        # THIS DOES THE CLASSIFIER-FREE GUIDANCE
        # THE OUTPUT IS SPLITTED IN TWO PARTS, ONE FOR CONDITIONED-ON-TEXT AND ANOTHER ONE FOR UNCONDITIONED-ON-TEXT outputs.        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        
        # w(t), sigma_t^2
        # w is used for scaling the gradient later.
        # alphas is variance schedules or noise levels.
        # t is time_step
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad_unweighted = (noise_pred - noise)
        
        # The unweighted gradient is scaled by loss_weight and the weight w (broadcasted to match dimensions). 
        # This scales the gradient according to the model's current state and the importance of the loss.
        grad = loss_weight * w[:, None, None, None] * grad_unweighted

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        
        # replaces NaNs (not a number) in the gradient tensor with numerical values (zeros by default), ensuring the stability of the training process.
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # _t = time.time()
        loss = SpecifyGradient.apply(latents, grad)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
        
        # Added for DDS Loss
        # loss, _ = self.get_sds_loss(z = latents, text_embeddings = text_embeddings, eps = None, mask=None, t=None,
        #          timestep = None, guidance_scale=7.5)

        if return_aux:
            aux = {'grad': grad, 'grad_unweighted': grad_unweighted, 't': t, 'w': w, 'latents': latents}
            return loss, aux
        else:
            return loss 


    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        # 'self.scheduler' is equivalent to 'sampler' in pytorch-stable-diffusion -> pipeline.py code
        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
            
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
    
    def noise_input(self, z, eps=None, timestep = None):
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
    
    def get_eps_prediction(self, z_t, timestep, text_embeddings, alpha_t, sigma_t, get_raw=False,
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
    
    def get_sds_loss(self, z, text_embeddings, eps = None, mask=None, t=None,
                 timestep = None, guidance_scale=7.5):
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, _ = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        del grad_z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss


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


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()








