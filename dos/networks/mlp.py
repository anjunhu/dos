import torch
from torch import nn

from .misc import get_activation


class MLP(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        super().__init__()
        assert num_layers >= 1
        if num_layers == 1:
            network = [nn.Linear(cin, cout, bias=False)]
        else:
            network = [nn.Linear(cin, nf, bias=False)]
            for _ in range(num_layers - 2):
                network += [nn.ReLU(inplace=True), nn.Linear(nf, nf, bias=False)]
                if dropout:
                    network += [nn.Dropout(dropout)]
            network += [nn.ReLU(inplace=True), nn.Linear(nf, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `omega0` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = omega0 * (2.0 ** torch.arange(n_harmonic_functions))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class MLPTextureSimple(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        num_layers,
        nf=256,
        dropout=0,
        activation=None,
        min_max=None,
        n_harmonic_functions=10,
        omega0=1,
        extra_dim=0,
        embed_concat_pts=True,
        perturb_normal=False,
        symmetrize=False,
        uniform=False,
    ):
        super().__init__()
        self.extra_dim = extra_dim
        self.uniform = uniform

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, omega0)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin

        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + extra_dim, cout, num_layers, nf, dropout, activation)
        self.perturb_normal = perturb_normal
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer("min_max", min_max)
        else:
            self.min_max = None
        self.bsdf = None

    def sample(self, x, feat=None):
        assert (feat is None and self.extra_dim == 0) or (
            feat.shape[-1] == self.extra_dim
        )
        b, h, w, c = x.shape

        if self.uniform:
            x = x * 0

        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x

        x = x.view(-1, c)
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x

        x_in = self.in_layer(x_in)
        if feat is not None:
            feat = feat[:, None, None].expand(b, h, w, -1).reshape(b * h * w, -1)
            x_in = torch.concat([x_in, feat], dim=-1)
        out = self.mlp(self.relu(x_in))
        if self.min_max is not None:
            out = (
                out * (self.min_max[1][None, :] - self.min_max[0][None, :])
                + self.min_max[0][None, :]
            )
        return out.view(b, h, w, -1)
