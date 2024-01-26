##------- CODE taken from https://github.com/tomasjakab/laam/blob/sds-investigation/dos/video3d/diffusion/sd.py. 

import os
from transformers import CLIPTextModel, CLIPTokenizer, logging, AutoTokenizer
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
            model_key = "DeepFloyd/IF-I-XL-v1.0"
        elif self.sd_version == '2.0':
            model_key = "DeepFloyd/IF-I-XL-v1.0"
        elif self.sd_version == '1.5':
            model_key = "DeepFloyd/IF-I-XL-v1.0"
        else:
            raise ValueError(f'Deep-Floyd version {self.sd_version} not supported.')
        
        print(f'[INFO] loading Deep-Floyd {model_key}')

        # Create model
        # cache_dir="/scratch/local/ssd/tomj/cache/huggingface_hub"
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch_dtype, cache_dir=cache_dir).to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", cache_dir=cache_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="tokenizer", cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", cache_dir=cache_dir).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch_dtype, cache_dir=cache_dir).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", cache_dir=cache_dir)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded Deep-Floyd!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        import ipdb; ipdb.set_trace()
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            # below line giving this error: The size of tensor a (512) must match the size of tensor b (77) at non-singleton dimension 1.
            # Shape of "text_input.input_ids.to(self.device)" is [1, 512].
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    # def get_text_embeds(self, prompt, negative_prompt):
    #     # Tokenize text and get embeddings
    #     text_input = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
    #     with torch.no_grad():
    #         prompt_embeds = self.text_encoder(text_input.input_ids)[0]

    #     if negative_prompt:
    #         neg_input = self.tokenizer(negative_prompt, return_tensors='pt').to(self.device)
    #         with torch.no_grad():
    #             negative_embeds = self.text_encoder(neg_input.input_ids)[0]
    #     else:
    #         negative_embeds = None

    #     # Cat for final embeddings
    #     text_embeddings = torch.cat([negative_embeds, prompt_embeds])
    #     print('text_embeddings shape', text_embeddings.shape)
    #     return text_embeddings


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
            import ipdb; ipdb.set_trace()
            
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
        print('shape of latents', latents)
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
            print('shape of imgs', imgs)
            
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




