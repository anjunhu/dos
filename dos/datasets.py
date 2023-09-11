from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .nvdiffrec.render.mesh import Mesh, concat_meshes, load_mesh


class BaseLoader(object):
    def __call__(self, x):
        x = self.loader(x)
        if hasattr(self, "transform"):
            x = self.transform(x)
        return x


class ImageLoader(BaseLoader):
    def __init__(self, image_size=256):
        self.loader = torchvision.datasets.folder.default_loader
        self.transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )


class MaskLoader(BaseLoader):
    def __init__(self, image_size=256, mask_threshold=0.1):
        self.loader = torchvision.datasets.folder.default_loader
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
                lambda x: (x > mask_threshold).float(),
            ]
        )


class TxtLoader(BaseLoader):
    def __init__(self):
        self.loader = lambda x: torch.from_numpy(np.loadtxt(x)).float()


class MeshLoader(BaseLoader):
    def __init__(self):
        self.loader = lambda x: load_mesh(x, load_materials=False)


class ImageDataset(Dataset):
    """
    dataset attributes: [{name: ..., root_dir: ..., suffix: ...}, ...]

    suffix is the end of the file name '*{suffix}'
    """

    def __init__(self, root_dir, attributes, image_size=256, mask_threshold=0.1):
        """ """
        self.root_dir = root_dir
        self.attributes = attributes

        # init the loaders and transforms
        self.attribute_loaders = {
            "image": ImageLoader(image_size=image_size),
            "background": ImageLoader(image_size=image_size),
            "mask": MaskLoader(image_size=image_size, mask_threshold=mask_threshold),
            "camera_matrix": TxtLoader(),
            "pose": TxtLoader(),
            "texture_features": torch.load,
            "mesh": MeshLoader(),
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
        # Remove root_dir from the sample names
        sample_names = [str(f)[len(str(root_dir)) + 1 :] for f in sample_names]
        return sample_names

    def _load_attribute(self, sample_name, attribute):
        """ """
        # Construct the file path
        file_path = Path(attribute["root_dir"]) / f"{sample_name}{attribute['suffix']}"

        # Load the file
        data = self.attribute_loaders[attribute["name"]](file_path)

        return data

    def __getitem__(self, index):
        """ """
        sample_name = self.sample_names[index]
        sample_data = {}
        for attribute in self.attributes:
            sample_data[attribute["name"]] = self._load_attribute(
                sample_name, attribute
            )
        sample_data["name"] = sample_name
        return sample_data

    def __len__(self):
        """ """
        return len(self.sample_names)

    @staticmethod
    def collate_fn(batch):
        mesh_items = {}
        non_mesh_items = {}

        # Separate mesh items from non-mesh items
        for key, value in batch[0].items():
            if isinstance(value, Mesh):
                mesh_items[key] = concat_meshes([item[key] for item in batch])
            else:
                non_mesh_items[key] = default_collate([item[key] for item in batch])

        # Combine mesh items and non-mesh items
        return {**mesh_items, **non_mesh_items}
