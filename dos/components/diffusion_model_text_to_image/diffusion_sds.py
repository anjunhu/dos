##------- CODE partly taken from https://github.com/tomasjakab/laam/blob/sds-investigation/dos/examples/diffusion_sds_example.py

import os
# add dos to path
import sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
from einops import rearrange
from PIL import Image
from tqdm import tqdm
sys.path.append('../../dos')
import argparse
import torch.optim
from omegaconf import OmegaConf
from dos.components.diffusion_model_text_to_image.deep_floyd import DeepFloyd
from dos.components.diffusion_model_text_to_image.sd import (StableDiffusion,
                                                             seed_everything)
from dos.components.diffusion_model_text_to_image.sd_dds_loss import StableDiffusionDDSLoss
from dos.components.diffusion_model_text_to_image.sd_XL import StableDiffusionXL
from dos.utils.framework import read_configs_and_instantiate

schedule = np.array([600] * 50).astype('int32')
device=torch.device('cuda:0')
optimizer_class=torch.optim.SGD
torch_dtype=torch.float16

class DiffusionForTargetImg:
    
    def __init__(
        self,
        cache_dir=None,
        init_image_path=None,
        output_dir = "sd_sds_output",
        vis_name = "cow-sds_latent-l2_image-600-lr1e-1.jpg", 
        prompts_source = ["a cow with front leg raised"], 
        negative_prompts = [''],
        prompts_target = [],
        mode = "sds_latent-l2_image", 
        lr=0.1,
        lr_l2=1e4, 
        seed=2, 
        num_inference_steps=20, 
        guidance_scale=100, 
        schedule=schedule,
        optimizer_class=optimizer_class,
        torch_dtype=torch_dtype,
        image_fr_path = False,
        select_diffusion_option = 'sd_dds_loss',
        dds = False,
    ):
        
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.init_image_path = init_image_path
        self.vis_name = vis_name
        self.prompts_source = prompts_source
        self.negative_prompts = negative_prompts
        self.prompts_target = prompts_target
        self.mode = mode
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.lr_l2 = lr_l2
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.schedule = schedule
        self.torch_dtype = torch_dtype
        self.select_diffusion_option = select_diffusion_option
        
        if self.select_diffusion_option=='df':
            self.df = DeepFloyd(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option=='sd':
            self.sd = StableDiffusion(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option=='sd_XL':
            self.sd_XL = StableDiffusionXL(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option=='sd_dds_loss':
            self.sd_dds_loss = StableDiffusionDDSLoss(device, cache_dir, torch_dtype=torch_dtype)
            
        self.image_fr_path = image_fr_path
        self.dds = dds
        
        seed_everything(self.seed)

    def run_experiment(self, input_image, image_fr_path=False, index=0):
        
        if self.select_diffusion_option=='df':
            text_embeddings = self.df.get_text_embeds(self.prompts_target, self.negative_prompts)
        elif self.select_diffusion_option=='sd':
            # Uses pre-trained CLIP Embeddings; # Prompts -> text embeds
            # SHAPE OF text_embeddings [2, 77, 768]
            text_embeddings = self.sd.get_text_embeds(self.prompts_target, self.negative_prompts)
        elif self.select_diffusion_option=='sd_XL':
            text_embeddings = self.sd_XL.get_text_embeds(self.prompts_target, self.negative_prompts)
        elif self.select_diffusion_option=='sd_dds_loss':
            text_embedding_source, text_embeddings = self.sd_dds_loss.get_text_embeds(self.prompts_source, self.negative_prompts, self.prompts_target)
        
        # init img
        height, width = 256, 256
        
        if self.image_fr_path == True:
            if self.init_image_path is not None:
                # load image -- source img
                img = Image.open(self.init_image_path).convert('RGB')
                img = img.resize((width, height), Image.LANCZOS)
                img = torchvision.transforms.ToTensor()(img)              # shape is torch.Size([3, 256, 256])
                # img[None]: This operation adds an additional dimension to the tensor, effectively reshaping it from [C,H,W] to [1,C,H,W]. 
                # In Python, None is used to add a new axis.
                # The 1, 1, 1 arguments indicate that the image should not be repeated along the channel, height, or width dimensions.
                # text_embeddings.shape[0] // 2: This divides the size of the first dimension by 2, using integer division.
                
                # repeat(...): The repeat function is used to replicate the tensor along specified dimensions. 
                # The arguments inside the repeat function indicate how many times to repeat the tensor along each dimension.
                
                # .repeat(text_embeddings.shape[0] // 2, 1, 1, 1) takes this single-item batch and repeats it. 
                # The image is not repeated across the color channels, height, or width dimensions (1, 1, 1) but is repeated text_embeddings.shape[0] // 2 times along the batch dimension. 
                # This creates a batch of images where each image in the batch is identical.
                
                img = img[None].repeat(text_embeddings.shape[0] // 2, 1, 1, 1)  # shape is torch.Size([1, 3, 256, 256])
                pred_rgb = img
            else:
                pred_rgb = torch.zeros((text_embeddings.shape[0] // 2, 3, height, width))
        
        else:
            
            if self.dds:
                pred_rgb = self.sd_dds_loss.load_512(input_image)
                pred_rgb = torch.from_numpy(pred_rgb).float().permute(2, 0, 1) / 127.5 - 1
                pred_rgb = pred_rgb.unsqueeze(0).to(device)
            else:
                img = input_image
                img = img[None].repeat(text_embeddings.shape[0] // 2, 1, 1, 1)
                pred_rgb = img   
            
        
        pred_rgb = pred_rgb.to(device).detach().clone().requires_grad_(True)


        def image_to_latents(pred_rgb):
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            pred_rgb_512 = pred_rgb_512.to(self.torch_dtype)
            
            if self.select_diffusion_option=='sd':
                latents = self.sd.encode_imgs(pred_rgb_512)
            if self.select_diffusion_option=='sd_XL':
                latents = self.sd_XL.encode_imgs(pred_rgb_512)
            elif self.select_diffusion_option=='sd_dds_loss':
                latents = self.sd_dds_loss.encode_imgs(pred_rgb_512)
            return latents

        if self.mode == "sds_image":
            param = pred_rgb
        elif self.mode in ["sds_latent", "sds_latent-l2_image"]:
            # # random init latents with normal distribution (same size as latents)
            # latents = torch.randn_like(latents)
            # latents shape torch.Size([1, 4, 64, 64])
            latents = image_to_latents(pred_rgb)
            latents = latents.detach().clone().requires_grad_(True)
            param = latents
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        optimizer = self.optimizer_class([param], lr=self.lr)

        # 
        if self.mode == "sds_latent-l2_image":
            optimizer_l2 = self.optimizer_class([pred_rgb], lr=self.lr_l2)

        all_imgs = []
        # all_imgs.append(pred_rgb.clone().detach())
        all_decoded_imgs = []


        # optimize
        for i in tqdm(range(self.num_inference_steps)):
            # optimizer.zero_grad()

            if self.mode == "sds_image-l2_image":
                # replace latents tensor value with current encoded image (do not create new var)
                # latents.data.shape: torch.Size([1, 4, 64, 64])
                latents.data = image_to_latents(pred_rgb).data
            
            if self.mode == "sds_image":
                # 'train_step_fn' - training steps are set differently based on the mode.
                # partial function creates a new func by passing the function and the arguments we want to pre-fill to partial.
                # train_step_fn is a new function
                if self.select_diffusion_option=='df':
                    train_step_fn = partial(self.df.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option in ['sd']:
                    train_step_fn = partial(self.sd.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option in ['sd_XL']:
                    train_step_fn = partial(self.sd_XL.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option=='sd_dds_loss':
                    train_step_fn = partial(self.sd_dds_loss.train_step, pred_rgb=pred_rgb)
                    
            elif self.mode in ["sds_latent", "sds_latent-l2_image"]:
                if self.select_diffusion_option=='df':
                    train_step_fn = partial(self.df.train_step, latents=latents)
                elif self.select_diffusion_option in ['sd']:
                    train_step_fn = partial(self.sd.train_step, latents=latents)
                elif self.select_diffusion_option in ['sd_XL']:
                    train_step_fn = partial(self.sd_XL.train_step, latents=latents)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if self.select_diffusion_option=='sd_dds_loss':
                # For sd_DDS Loss
                img_target = latents.clone()
                loss, log_loss = self.sd_dds_loss.get_dds_loss(latents, img_target, text_embedding_source, text_embeddings)
                optimizer.zero_grad()
                (2000 * loss).backward()
                optimizer.step()
                
                if i % 2 == 0:
                    rgb_decoded = self.sd_dds_loss.decode_latents(img_target, im_cat=None)
                    # rgb_decoded = rgb_decoded.resize((256, 256))

                    rgb_decoded.save(f'{self.output_dir}/{i}_dds_loss_rgb_decoded.jpg')
                
            else:
                # For SD, SD_XL and DeepFloyd sds Loss
                loss, aux = train_step_fn(text_embeddings, guidance_scale=self.guidance_scale, fixed_step=self.schedule[i], return_aux=True)
                latents = aux['latents']
                latents.retain_grad()
                loss.backward()

                # print min and max of latents, latents grad, and rgb_decoded and pred_rgb
                print(f"latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
                print(f"latents.grad: min={latents.grad.min().item():.4f}, max={latents.grad.max().item():.4f}")
                print(f"pred_rgb: min={pred_rgb.min().item():.4f}, max={pred_rgb.max().item():.4f}")

                # Decoding the Latent to image space for Stable Diffusion
                rgb_decoded = self.sd.decode_latents(latents)
                print(f"rgb_decoded: min={rgb_decoded.min().item():.4f}, max={rgb_decoded.max().item():.4f}")
                
                optimizer.step()
                latents.grad = None

                # optimize pred_rgb to be close to rgb_decoded
                if self.mode == "sds_latent-l2_image":
                    optimizer_l2.zero_grad()

                    rgb_decoded_ = F.interpolate(rgb_decoded.detach().to(pred_rgb.dtype), (height, width), mode='bilinear', align_corners=False)
                    loss_l2 = F.mse_loss(pred_rgb, rgb_decoded_)
                    # print loss_l2
                    print(f"loss_l2: {loss_l2.item():.4f}")
                    loss_l2.backward()
                    # print min and max of pred_rgb grad in scientific notation
                    print(f"pred_rgb.grad: min={pred_rgb.grad.min().item():.4e}, max={pred_rgb.grad.max().item():.4e}")
                    optimizer_l2.step()

            if i % 2 == 0:
                all_imgs.append(pred_rgb.clone().detach())
                
                if self.select_diffusion_option in ['sd', 'sd_XL']:
                    all_decoded_imgs.append(rgb_decoded.clone().detach())

        # %%
        # save all images
        n_images = len(all_imgs)
        all_imgs = rearrange(torch.stack(all_imgs), 't b c h w -> (b t) c h w')
        all_imgs = torchvision.utils.make_grid(all_imgs, nrow=n_images, pad_value=1)
        
        if self.select_diffusion_option in ['sd', 'sd_XL']:
            all_decoded_imgs = rearrange(torch.stack(all_decoded_imgs), 't b c h w -> (b t) c h w')
            all_decoded_imgs = torchvision.utils.make_grid(all_decoded_imgs, nrow=n_images, pad_value=1)

            # add below
            # resize all_imgs to be the same size as all_decoded_imgs
            all_imgs = torch.nn.functional.interpolate(all_imgs[None], size=all_decoded_imgs.shape[-2:])[0]

            if self.mode == "sds_latent":
                all_imgs = all_decoded_imgs
            else:
                all_imgs = torch.cat([all_imgs, all_decoded_imgs], dim=1)

        all_imgs = all_imgs.detach().cpu().permute(1, 2, 0).numpy()
        # clip to [0, 1]
        all_imgs_save = all_imgs.copy()
        all_imgs_save = all_imgs_save.clip(0, 1)
        all_imgs_save = (all_imgs_save * 255).round().astype('uint8')
        file_name = f'{index}_cow-sds_latent-l2_image-600-lr1e-1.jpg'
        out_path = Path(self.output_dir) / file_name
        out_path.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(all_imgs_save).save(out_path)
        
        # pred_rgb size is 256x256 
        pred_rgb_PIL = torchvision_F.to_pil_image(pred_rgb[0])
        pred_rgb_PIL.save(f'{self.output_dir}/{index}_pred_rgb.jpg')
        
        if self.select_diffusion_option in ['sd', 'sd_XL']:
            # rgb_decoded size is 512x512
            rgb_decoded_PIL = torchvision_F.to_pil_image(rgb_decoded[0])
            rgb_decoded_PIL.save(f'{self.output_dir}/{index}_rgb_decoded.jpg')
        
        return pred_rgb
        

if __name__ == '__main__':    
    # Use the configuration
    sd_text_to_target_img, _ = read_configs_and_instantiate()

    # Call the fn run_experiment
    sd_text_to_target_img.run_experiment(None)
