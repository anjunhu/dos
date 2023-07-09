import torch
import torch.nn as nn
import torchvision.transforms as transforms

from third_party.dino_vit_extractor.extractor import ViTExtractor


# TODO: disenangle the original ViT encoder and newely added learnable layers 'final_layer_type'
class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_type="dino_vits8",
        stride=8,
        facet="key",
        layer=11,
        image_size=256,
        pad=0,
        device=None,
    ):
        super().__init__()
        self.extractor = ViTExtractor(
            model_type=model_type, stride=stride, device=device
        )
        self.model_type = model_type
        self.facet = facet
        self.layer = layer
        self.image_size = image_size
        self.pad = pad

    def preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses an image before extraction.
        :param image_tensor: tensor of shape BxCxHxW.
        :return: a tensor containing the preprocessed image to insert the model of shape BxCxHxW.
        """
        prep = transforms.Compose(
            [transforms.Normalize(mean=self.extractor.mean, std=self.extractor.std)]
        )
        prep_img = prep(image_tensor)
        return prep_img

    def forward(self, x):
        with torch.no_grad():
            # resizes the image to the correct size
            if x.shape[-1] != self.image_size:
                assert x.shape[-1] == x.shape[-2]
                x = nn.functional.interpolate(
                    x, size=(self.image_size, self.image_size), mode="bilinear"
                )
            x = self.preprocess_tensor(x)
            x = torch.nn.functional.pad(x, 4 * [self.pad], mode="reflect")
            features = self.extractor.extract_descriptors(
                x, facet=self.facet, layer=self.layer, bin=False
            )
            features = features.reshape(
                features.shape[0], *self.extractor.num_patches, features.shape[-1]
            )
            features = features.permute(0, 3, 1, 2)
        return features
