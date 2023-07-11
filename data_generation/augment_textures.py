import cv2
import os
from albumentations import (
    Compose,
    HueSaturationValue,
    RGBShift,
    CLAHE,
    InvertImg,
    ToGray,
    RandomBrightnessContrast,
    RandomGamma,
    Blur,
    GaussNoise,
    JpegCompression,
)
import glob
import random


class BlendImages:
    def __init__(
        self, image_list, augmentation, alpha_range=(0, 1), always_apply=False, p=0.5
    ):
        self.image_list = image_list
        self.augmentation = augmentation
        self.alpha_range = alpha_range
        self.always_apply = always_apply
        self.p = p

    def __call__(self, image, **kwargs):
        # Randomly select two images
        img1 = random.choice(self.image_list)
        img2 = random.choice(self.image_list)

        # Ensure the images have the same size as the input image
        img1 = cv2.resize(img1, (image.shape[1], image.shape[0]))
        img2 = cv2.resize(img2, (image.shape[1], image.shape[0]))

        # Apply augmentations
        img1 = self.augmentation(image=img1)["image"]
        img2 = self.augmentation(image=img2)["image"]

        # Perform alpha blending
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
        blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

        # Return the result
        return {"image": blended}


num_augmented_per_texture = 20

# List of your texture files
texture_path = "/scratch/local/hdd/tomj/datasets/synth_animals/data/DOC/maps/frankensteinDiffuses_v001/diffuse_horse_*.jpg"
texture_files = glob.glob(texture_path)

# Output directory for augmented images
output_dir = (
    "/scratch/local/hdd/tomj/datasets/synth_animals/textures/generated/augmented/horse"
)

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load all texture images into memory
texture_images = [
    cv2.cvtColor(cv2.imread(texture_file), cv2.COLOR_BGR2RGB)
    for texture_file in texture_files
]

# Define the augmentation pipeline
image_augmentation = Compose(
    [
        HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), always_apply=False, p=0.5),
        InvertImg(p=0.2),
        ToGray(p=0.2),
        RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5
        ),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(p=0.3),
        JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    ],
    p=1,
)  # Apply all augmentations

# Define the full augmentation pipeline, including blending
augmentation = Compose(
    [BlendImages(texture_images, image_augmentation, p=0.5)],
    p=1,
)  # Apply all augmentations

i = 0
for img in texture_images:
    for _ in range(num_augmented_per_texture):
        # Convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = augmentation(image=img)
        augmented_img = augmented["image"]

        # Convert the image back from RGB to BGR
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

        # Save augmented image
        filename = os.path.join(output_dir, f"{i:05d}.jpg")
        cv2.imwrite(filename, augmented_img)

        i += 1
