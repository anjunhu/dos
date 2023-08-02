import numpy as np
import torch
from torch import nn

from ..networks.mlp import MLPTextureSimple


class TexturePredictor(nn.Module):
    def __init__(
        self,
        grid_scale=7,  # FIXME: this is a shared parameter with other components of the model, should be set with one parameter for the whole model
        embedder_freq_tex=10,
        hidden_size=256,
        num_layers_tex=8,
        latent_dim=256,
        sym_texture=True,
        uniform_texture=False,
        kd_min=[0.0, 0.0, 0.0, 0.0],
        kd_max=[1.0, 1.0, 1.0, 1.0],
        ks_min=[0.0, 0.0, 0.0],
        ks_max=[0.0, 0.0, 0.0],
        nrm_min=[-1.0, -1.0, 0.0],
        nrm_max=[1.0, 1.0, 1.0],
    ):
        super().__init__()
        kd_min = torch.FloatTensor(kd_min)
        kd_max = torch.FloatTensor(kd_max)
        ks_min = torch.FloatTensor(ks_min)
        ks_max = torch.FloatTensor(ks_max)
        nrm_min = torch.FloatTensor(nrm_min)
        nrm_max = torch.FloatTensor(nrm_max)

        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        min_max = torch.stack((mlp_min, mlp_max), dim=0)
        out_chn = 9
        embedder_scaler = (
            2 * np.pi / grid_scale * 0.9
        )  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
        self.perturb_normal = False
        self.net_texture = MLPTextureSimple(
            3,  # x, y, z coordinates
            out_chn,
            num_layers_tex,
            nf=hidden_size,
            dropout=0,
            activation="sigmoid",
            min_max=min_max,
            n_harmonic_functions=embedder_freq_tex,
            omega0=embedder_scaler,
            extra_dim=latent_dim,
            embed_concat_pts=True,
            perturb_normal=self.perturb_normal,
            symmetrize=sym_texture,
            uniform=uniform_texture,
        )

    def forward(self, *args, **kwargs):
        return self.net_texture(*args, **kwargs)

    # TODO: should use forward instead of sample
    def sample(self, *args, **kwargs):
        return self.net_texture.sample(*args, **kwargs)
