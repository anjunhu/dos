##------- CODE partly taken from https://github.com/tomasjakab/laam/blob/sds-investigation/dos/examples/diffusion_sds_example.py

import glob
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

from dos.datasets import ImageLoader

sys.path.append("../../dos")

import torch.optim

from dos.components.diffusion_model_text_to_image.deep_floyd import DeepFloyd
from dos.components.diffusion_model_text_to_image.sd import (StableDiffusion,
                                                             seed_everything)
from dos.components.diffusion_model_text_to_image.sd_dds_loss import \
    StableDiffusionDDSLoss
from dos.components.diffusion_model_text_to_image.sd_XL import \
    StableDiffusionXL
from dos.utils.framework import read_configs_and_instantiate


def load_images(image_paths, size, device):
    """
    creates a batch of images from a list of image paths
    """
    images = []
    for path in image_paths:
        img = ImageLoader(size)(path)
        images.append(img)
    return torch.stack(images).to(device)


# TODO: move this inside the class
device = torch.device("cuda:0")
optimizer_class = torch.optim.SGD
torch_dtype = torch.float16


class DiffusionForTargetImg:

    def __init__(
        self,
        cache_dir=None,
        init_image_path=None,
        output_dir=None,
        vis_name="cow-sds_latent-l2_image-600-lr1e-1.jpg",
        prompts=["a cow with front leg raised"],
        negative_prompts=[""],
        prompts_source=[],
        mode="sds_latent-l2_image",
        lr=0.1,
        momentum=0.0,
        lr_l2=1e4,
        seed=2,
        num_inference_steps=20,
        l2_image_period=1,
        guidance_scale=100,
        schedule="[600] * 50",
        optimizer_class=optimizer_class,
        torch_dtype=torch_dtype,
        image_fr_path=False,
        select_diffusion_option="sd",
        use_nfsd=False,
        dds=False,
        save_visuals_every_n_iter=2,
    ):

        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.vis_name = vis_name
        self.prompts_source = prompts_source
        self.negative_prompts = negative_prompts
        self.prompts = prompts
        self.mode = mode
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.momentum = momentum
        self.lr_l2 = lr_l2
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.l2_image_period = l2_image_period
        self.guidance_scale = guidance_scale

        if isinstance(schedule, str):
            schedule = np.array(eval(schedule)).astype("int32")
        self.schedule = schedule

        self.torch_dtype = torch_dtype
        self.select_diffusion_option = select_diffusion_option
        self.use_nfsd = use_nfsd
        self.save_visuals_every_n_iter = save_visuals_every_n_iter

        if init_image_path is not None:
            if isinstance(init_image_path, str):
                init_image_path = glob.glob(init_image_path)
        self.init_image_path = init_image_path

        if self.select_diffusion_option == "df":
            self.df = DeepFloyd(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option == "sd":
            self.sd = StableDiffusion(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option == "sd_XL":
            self.sd_XL = StableDiffusionXL(device, cache_dir, torch_dtype=torch_dtype)
        elif self.select_diffusion_option == "sd_dds_loss":
            self.sd_dds_loss = StableDiffusionDDSLoss(
                device, cache_dir, torch_dtype=torch_dtype
            )
        else:
            raise ValueError(
                f"Unknown diffusion option: {self.select_diffusion_option}"
            )

        self.image_fr_path = image_fr_path
        self.dds = dds

        seed_everything(self.seed)

    def run_experiment(self, input_image, image_fr_path=False, index=0):

        if self.select_diffusion_option == "df":
            text_embeddings = self.df.get_text_embeds(
                self.prompts, self.negative_prompts
            )
        elif self.select_diffusion_option == "sd":
            # Uses pre-trained CLIP Embeddings; # Prompts -> text embeds
            # SHAPE OF text_embeddings [2, 77, 768]
            text_embeddings = self.sd.get_text_embeds(
                self.prompts, self.negative_prompts, use_nfsd=self.use_nfsd
            )
        elif self.select_diffusion_option == "sd_XL":
            text_embeddings = self.sd_XL.get_text_embeds(
                self.prompts, self.negative_prompts
            )
        elif self.select_diffusion_option == "sd_dds_loss":
            text_embedding_source, text_embeddings = self.sd_dds_loss.get_text_embeds(
                self.prompts_source, self.negative_prompts, self.prompts
            )

        # FIXME: this is 64 for deepfloyd
        encoder_image_size = 1024 if self.select_diffusion_option == "sd_XL" else 512

        if self.image_fr_path == True:
            if self.init_image_path is not None:
                img = load_images(self.init_image_path, encoder_image_size, device)
                # prompts can be a list of string or a single string
                n_prompts = 1 if isinstance(self.prompts, str) else len(self.prompts)
                img = img.repeat(n_prompts, 1, 1, 1)
                pred_rgb = img
            else:
                pred_rgb = torch.zeros(
                    (len(self.prompts), 3, encoder_image_size, encoder_image_size)
                )
        else:
            if self.dds:
                pred_rgb = self.sd_dds_loss.load_512(input_image)
                pred_rgb = (
                    torch.from_numpy(pred_rgb).float().permute(2, 0, 1) / 127.5 - 1
                )
                pred_rgb = pred_rgb.unsqueeze(0).to(device)
            else:
                img = input_image
                img = img[None].repeat(text_embeddings.shape[0] // 2, 1, 1, 1)
                pred_rgb = img

        pred_rgb = pred_rgb.to(device).detach().clone().requires_grad_(True)

        def image_to_latents(pred_rgb):
            pred_rgb_512 = F.interpolate(
                pred_rgb,
                (encoder_image_size, encoder_image_size),
                mode="bilinear",
                align_corners=False,
            )
            pred_rgb_512 = pred_rgb_512.to(self.torch_dtype)

            if self.select_diffusion_option == "sd":
                latents = self.sd.encode_imgs(pred_rgb_512)
            if self.select_diffusion_option == "sd_XL":
                latents = self.sd_XL.encode_imgs(pred_rgb_512)
            elif self.select_diffusion_option == "sd_dds_loss":
                latents = self.sd_dds_loss.encode_imgs(pred_rgb_512)
            return latents

        if self.mode == "sds_image":
            param = pred_rgb
        elif self.mode in ["sds_latent", "sds_latent_decenc", "sds_latent-l2_image"]:
            # # random init latents with normal distribution (same size as latents)
            # latents = torch.randn_like(latents)
            # latents shape torch.Size([1, 4, 64, 64])
            latents = image_to_latents(pred_rgb)
            latents = latents.detach().clone().requires_grad_(True)
            param = latents
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        optimizer = self.optimizer_class([param], lr=self.lr, momentum=self.momentum)

        #
        if self.mode == "sds_latent-l2_image":
            optimizer_l2 = self.optimizer_class([pred_rgb], lr=self.lr_l2)

        all_imgs = []
        # all_imgs.append(pred_rgb.clone().detach())
        all_decoded_imgs = []

        # optimize
        for i in tqdm(range(self.num_inference_steps)):
            # optimizer.zero_grad()

            if self.mode == "sds_image":
                # 'train_step_fn' - training steps are set differently based on the mode.
                # partial function creates a new func by passing the function and the arguments we want to pre-fill to partial.
                # train_step_fn is a new function
                if self.select_diffusion_option == "df":
                    train_step_fn = partial(self.df.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option in ["sd"]:
                    train_step_fn = partial(self.sd.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option in ["sd_XL"]:
                    train_step_fn = partial(self.sd_XL.train_step, pred_rgb=pred_rgb)
                elif self.select_diffusion_option == "sd_dds_loss":
                    train_step_fn = partial(
                        self.sd_dds_loss.train_step, pred_rgb=pred_rgb
                    )

            elif self.mode in [
                "sds_latent",
                "sds_latent_decenc",
                "sds_latent-l2_image",
            ]:
                if self.select_diffusion_option == "df":
                    train_step_fn = partial(self.df.train_step, latents=latents)
                elif self.select_diffusion_option in ["sd"]:
                    train_step_fn = partial(self.sd.train_step, latents=latents)
                elif self.select_diffusion_option in ["sd_XL"]:
                    train_step_fn = partial(self.sd_XL.train_step, latents=latents)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if self.select_diffusion_option == "sd_dds_loss":
                # For sd_DDS Loss
                img_target = latents.clone()
                loss, log_loss = self.sd_dds_loss.get_dds_loss(
                    latents, img_target, text_embedding_source, text_embeddings
                )
                optimizer.zero_grad()
                (2000 * loss).backward()
                optimizer.step()

                if i % 2 == 0:
                    rgb_decoded = self.sd_dds_loss.decode_latents(
                        img_target, im_cat=None
                    )
                    # rgb_decoded = rgb_decoded.resize((256, 256))

                    rgb_decoded.save(f"{self.output_dir}/{i}_dds_loss_rgb_decoded.jpg")

            else:
                # For SD, SD_XL and DeepFloyd sds Loss
                loss, aux = train_step_fn(
                    text_embeddings,
                    guidance_scale=self.guidance_scale,
                    fixed_step=self.schedule[i],
                    return_aux=True,
                    use_nfsd=self.use_nfsd,
                )
                if self.mode == "sds_image":
                    latents = aux["latents"]
                latents.retain_grad()
                loss.backward()

                # print min and max of latents, latents grad, and rgb_decoded and pred_rgb
                print(
                    f"latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}"
                )
                print(
                    f"latents.grad: min={latents.grad.min().item():.4f}, max={latents.grad.max().item():.4f}"
                )
                print(
                    f"pred_rgb: min={pred_rgb.min().item():.4f}, max={pred_rgb.max().item():.4f}"
                )

                # Decoding the Latent to image space for Stable Diffusion
                # TODO: use the same variable name for all the models
                sd = self.sd if self.select_diffusion_option == "sd" else self.sd_XL
                # TODO: add option to decode only the last image if the mode is sds_latent - get speed up
                rgb_decoded = sd.decode_latents(latents)
                print(
                    f"rgb_decoded: min={rgb_decoded.min().item():.4f}, max={rgb_decoded.max().item():.4f}"
                )
                optimizer.step()
                latents.grad = None

                if self.mode == "sds_latent_decenc":
                    latents.data = image_to_latents(rgb_decoded).data

                # optimize pred_rgb to be close to rgb_decoded
                if self.mode == "sds_latent-l2_image" and i % self.l2_image_period == 0:
                    optimizer_l2.zero_grad()

                    rgb_decoded_ = F.interpolate(
                        rgb_decoded.detach().to(pred_rgb.dtype),
                        (encoder_image_size, encoder_image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss_l2 = F.mse_loss(pred_rgb, rgb_decoded_)
                    # print loss_l2
                    print(f"loss_l2: {loss_l2.item():.4f}")
                    loss_l2.backward()
                    # print min and max of pred_rgb grad in scientific notation
                    print(
                        f"pred_rgb.grad: min={pred_rgb.grad.min().item():.4e}, max={pred_rgb.grad.max().item():.4e}"
                    )
                    optimizer_l2.step()

                    # replace latents tensor value with current encoded image (do not create new var)
                    # latents.data.shape: torch.Size([1, 4, 64, 64])
                    latents.data = image_to_latents(pred_rgb).data

            if i % self.save_visuals_every_n_iter == 0:
                all_imgs.append(pred_rgb.clone().detach())

                if self.select_diffusion_option in ["sd", "sd_XL"]:
                    all_decoded_imgs.append(rgb_decoded.clone().detach())

        if self.mode in ["sds_latent", "sds_latent_decenc"]:
            pred_rgb = sd.decode_latents(latents)
            if input_image is not None:
                # resize pred_rgb to be the same size as input_image
                pred_rgb = torch.nn.functional.interpolate(
                    pred_rgb, size=input_image.shape[-2:]
                )

        # %%
        # save all images
        if self.output_dir is not None:
            n_images = len(all_imgs)
            all_imgs = rearrange(torch.stack(all_imgs), "t b c h w -> (b t) c h w")
            all_imgs = torchvision.utils.make_grid(all_imgs, nrow=n_images, pad_value=1)

            if self.select_diffusion_option in ["sd", "sd_XL"]:
                all_decoded_imgs = rearrange(
                    torch.stack(all_decoded_imgs), "t b c h w -> (b t) c h w"
                )
                all_decoded_imgs = torchvision.utils.make_grid(
                    all_decoded_imgs, nrow=n_images, pad_value=1
                )

                # add below
                # resize all_imgs to be the same size as all_decoded_imgs
                all_imgs = torch.nn.functional.interpolate(
                    all_imgs[None], size=all_decoded_imgs.shape[-2:]
                )[0]

                if self.mode in "sds_latent":
                    all_imgs = all_decoded_imgs
                else:
                    all_imgs = torch.cat([all_imgs, all_decoded_imgs], dim=1)

            all_imgs = all_imgs.detach().cpu().permute(1, 2, 0).numpy()
            # clip to [0, 1]
            all_imgs_save = all_imgs.copy()
            all_imgs_save = all_imgs_save.clip(0, 1)
            all_imgs_save = (all_imgs_save * 255).round().astype("uint8")
            file_name = f"{index}-{self.vis_name}"
            out_path = Path(self.output_dir) / file_name
            out_path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(all_imgs_save).save(out_path)

            # pred_rgb size is 256x256
            pred_rgb_PIL = torchvision_F.to_pil_image(pred_rgb[0])
            pred_rgb_PIL.save(f"{self.output_dir}/{index}_pred_rgb.jpg")

            if self.select_diffusion_option in ["sd", "sd_XL"]:
                # rgb_decoded size is 512x512
                rgb_decoded_PIL = torchvision_F.to_pil_image(rgb_decoded[0])
                rgb_decoded_PIL.save(f"{self.output_dir}/{index}_rgb_decoded.jpg")

        return pred_rgb


if __name__ == "__main__":
    # Use the configuration
    sd_text_to_target_img, _ = read_configs_and_instantiate()

    # Call the fn run_experiment
    sd_text_to_target_img.run_experiment(None)
