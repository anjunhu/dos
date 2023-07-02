from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    dataset attributes: [{name: ..., root_dir: ..., suffix: ...}, ...]

    suffix is the end of the file name '*{suffix}'
    """

    def __init__(self, root_dir, attributes, image_size=256):
        """ """
        self.root_dir = root_dir
        self.attributes = attributes

        # init the loaders and transforms
        # TODO: add support for other attributes
        self.attribute_loaders = {
            "image": torchvision.datasets.folder.default_loader,
            "mask": torchvision.datasets.folder.default_loader,
            "camera_matrix": lambda x: torch.from_numpy(np.loadtxt(x)).float(),
        }
        self.attribute_transforms = {
            "image": transforms.Compose(
                [transforms.Resize(image_size), transforms.ToTensor()]
            ),
            "mask": transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ]
            ),
        }

        # add root_dir to each attribute if not already present
        for attribute in self.attributes:
            if "root_dir" not in attribute:
                attribute["root_dir"] = self.root_dir

        # recursively find the sample names based on the first attribute inside the root_dir
        # sample name = file name without suffix
        root_dir = self.attributes[0]["root_dir"]
        suffix = self.attributes[0]["suffix"]
        self.sample_names = self._find_sample_names(root_dir, suffix)

        # check that dataset is not empty
        assert (
            len(self.sample_names) > 0
        ), f"Dataset is empty. Tried to search for files matching {root_dir}/**/*{suffix} "

        # sort the sample names
        self.sample_names = sorted(self.sample_names)

        print(f"Found {len(self.sample_names)} samples in {root_dir}")

    def _find_sample_names(self, root_dir, suffix):
        """
        recursively find the sample names inside the root_dir matching the suffix

        suffix is the end of the file name: '*{suffix}'
        """
        root_dir = Path(root_dir)
        # Glob the directory with the given suffix
        files = root_dir.glob(f"**/*{suffix}")

        # Extract stem (file name without suffix) for each file
        sample_names = [str(f)[: -len(suffix)] for f in files]

        return sample_names

    def _load_attribute(self, sample_name, attribute):
        """ """
        # Construct the file path
        file_path = Path(attribute["root_dir"]) / f"{sample_name}{attribute['suffix']}"

        # Load the file
        data = self.attribute_loaders[attribute["name"]](file_path)

        # Apply transforms
        if attribute["name"] in self.attribute_transforms:
            data = self.attribute_transforms[attribute["name"]](data)

        return data

    def __getitem__(self, index):
        """ """
        sample_name = self.sample_names[index]
        sample_data = {}
        for attribute in self.attributes:
            sample_data[attribute["name"]] = self._load_attribute(
                sample_name, attribute
            )
        return sample_data

    def __len__(self):
        """ """
        return len(self.sample_names)
