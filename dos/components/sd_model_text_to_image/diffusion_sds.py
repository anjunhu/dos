# %%
import os
import torch
import torchvision
import numpy as np

from PIL import Image
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from functools import partial
from pathlib import Path
import torchvision.transforms.functional as torchvision_F


# add dos to path
import sys
sys.path.append('../../dos')

# %%
# from video3d.diffusion.sd import StableDiffusion
# from video3d.diffusion.sd import seed_everything

# UNCOMMENT IT LATER
from dos.components.sd_model_text_to_image.sd import StableDiffusion
from dos.components.sd_model_text_to_image.sd import seed_everything


schedule = np.array([600] * 50).astype('int32')

class Stable_Diffusion_Text_to_Target_Img:
    def __init__(self, device, torch_dtype, cache_dir, output_dir, init_image_path, vis_name, prompts, negative_prompts, mode, optimizer_class, lr, lr_l2, seed, num_inference_steps, guidance_scale, input_image, image_fr_path = False, schedule=schedule):
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.init_image_path = init_image_path
        self.vis_name = vis_name
        self.prompts = prompts
        self.negative_prompts = negative_prompts
        self.mode = mode
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.lr_l2 = lr_l2
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.input_image = input_image
        self.image_fr_path = image_fr_path
        self.schedule = schedule
        self.sd = StableDiffusion(device, torch_dtype=torch_dtype, cache_dir=cache_dir)
        seed_everything(self.seed)

        
    def run_experiment(self):
        
        # Uses pre-trained CLIP Embeddings
        # Prompts -> text embeds
        # SHAPE OF text_embeddings [2, 77, 768]
        text_embeddings = self.sd.get_text_embeds(self.prompts, self.negative_prompts) 

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
            # Newly Added
            print('self.input_image.shape', self.input_image.shape)
            img = self.input_image
            img = img[None].repeat(text_embeddings.shape[0] // 2, 1, 1, 1)
            print('img.shape', img.shape)
            pred_rgb = img   
         
        pred_rgb = pred_rgb.to(self.sd.device).detach().clone().requires_grad_(True)


        def image_to_latents(pred_rgb):
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            pred_rgb_512 = pred_rgb_512.to(self.sd.torch_dtype)
            latents = self.sd.encode_imgs(pred_rgb_512)
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
                train_step_fn = partial(self.sd.train_step, pred_rgb=pred_rgb)
            elif self.mode in ["sds_latent", "sds_latent-l2_image"]:
                train_step_fn = partial(self.sd.train_step, latents=latents)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            loss, aux = train_step_fn(text_embeddings, guidance_scale=self.guidance_scale, fixed_step=self.schedule[i], return_aux=True)
            latents = aux['latents']
            latents.retain_grad()
            loss.backward()

            # Decoding the Latent to image space
            rgb_decoded = self.sd.decode_latents(latents)

            # print min and max of latents, latents grad, and rgb_decoded and pred_rgb
            print(f"latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
            print(f"latents.grad: min={latents.grad.min().item():.4f}, max={latents.grad.max().item():.4f}")
            print(f"rgb_decoded: min={rgb_decoded.min().item():.4f}, max={rgb_decoded.max().item():.4f}")
            print(f"pred_rgb: min={pred_rgb.min().item():.4f}, max={pred_rgb.max().item():.4f}")
            
            optimizer.step()
            latents.grad = None

            # optimize pred_rgb to be close to rgb_decoded
            if self.mode == "sds_latent-l2_image":
                optimizer_l2.zero_grad()
                # match size of rgb_decoded to pred_rgb
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
                all_decoded_imgs.append(rgb_decoded.clone().detach())

        # %%
        # save all images
        n_images = len(all_imgs)
        all_imgs = rearrange(torch.stack(all_imgs), 't b c h w -> (b t) c h w')
        all_imgs = torchvision.utils.make_grid(all_imgs, nrow=n_images, pad_value=1)
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
        out_path = Path(self.output_dir) / self.vis_name
        out_path.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(all_imgs_save).save(out_path)
        
        # pred_rgb size is 256x256 
        pred_rgb_PIL = torchvision_F.to_pil_image(pred_rgb[0])
        pred_rgb_PIL.save(f'{self.output_dir}/pred_rgb.jpg')
        
        # rgb_decoded size is 512x512
        rgb_decoded_PIL = torchvision_F.to_pil_image(rgb_decoded[0])
        rgb_decoded_PIL.save(f'{self.output_dir}/rgb_decoded.jpg')
        
        return pred_rgb, rgb_decoded
        

if __name__ == '__main__':
    
    from sd import StableDiffusion
    from sd import seed_everything

    # Usage:
    # creating an object for the class 'Stable_Diffusion_Text_to_Target_Img'
    sd_text_to_target_img = Stable_Diffusion_Text_to_Target_Img(
        device=torch.device('cuda:0'),
        torch_dtype=torch.float16,
        cache_dir="/work/oishideb/cache/huggingface_hub",
        output_dir='output-new',
        init_image_path='/users/oishideb/laam/dos/examples/data/cow.png',
        vis_name='cow-sds_latent-l2_image-600-lr1e-1.png',
        prompts=['a running cow'],
        negative_prompts=[''],
        mode="sds_latent-l2_image",
        optimizer_class=torch.optim.SGD,
        lr=0.1,
        lr_l2=1e4,
        seed=2,
        num_inference_steps=8,
        guidance_scale=100,
        image_fr_path = True,
        input_image = None,
        schedule = np.array([600] * 50).astype('int32')
    )

    # Call the fn run_experiment
    sd_text_to_target_img.run_experiment()



# # %%
# device = torch.device('cuda:0')
# # cache_dir="/scratch/local/ssd/tomj/cache/huggingface_hub"
# # cache_dir="/scratch/shared/beegfs/tomj/cache/huggingface_hub"
# cache_dir = "/work/oishideb/cache/huggingface_hub"
# sd = StableDiffusion(device, torch_dtype=torch.float16, cache_dir=cache_dir)

# # %%
# # settings
# init_image_path = '/users/oishideb/laam/dos/examples/data/cow.png'

# output_dir = 'output-new'
# vis_name = 'cow-sds_latent-l2_image-600-lr1e-1.jpg'

# # prompts = ['a cow']
# prompts = ['a standing cow']
# prompts = ['a running cow']
# # prompts = ['a sitting cow']
# negative_prompts = ['']

# # mode = "sds_image"
# # mode = "sds_latent"
# mode = "sds_latent-l2_image"

# # optimizer_class = torch.optim.Adam
# optimizer_class = torch.optim.SGD
# lr = 0.1
# lr_l2 = 1e4
# seed = 2

# num_inference_steps = 50
# guidance_scale = 100

# # schedule = np.linspace(999, 0, num_inference_steps).astype('int32')
# schedule = np.array([600] * num_inference_steps).astype('int32')

# # %%
# seed_everything(seed)
