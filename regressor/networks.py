import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from typing import Union, List, Tuple


EPS = 1e-7


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError


class Encoder32(nn.Module):
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(
                cin, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 32x32 -> 16x16
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 16x16 -> 8x8
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 8x8 -> 4x4
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, cout, kernel_size=4, stride=1, padding=0, bias=False
            ),  # 4x4 -> 1x1
        ]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


# TODO: disenangle the original ViT encoder and newely added learnable layers 'final_layer_type'
class ViTEncoder(nn.Module):
    def __init__(
        self,
        cout,
        which_vit="dino_vits8",
        pretrained=False,
        frozen=False,
        in_size=256,
        final_layer_type="none",
    ):
        super().__init__()
        self.ViT = torch.hub.load(
            "facebookresearch/dino:main", which_vit, pretrained=pretrained
        )
        if frozen:
            for p in self.ViT.parameters():
                p.requires_grad = False
        if which_vit == "dino_vits8":
            self.vit_feat_dim = 384
            self.patch_size = 8
        elif which_vit == "dino_vitb8":
            self.vit_feat_dim = 768
            self.patch_size = 8

        self._feats = []
        self.hook_handlers = []

        if final_layer_type == "none":
            pass
        elif final_layer_type == "conv":
            self.final_layer_patch_out = Encoder32(
                self.vit_feat_dim, cout, nf=256, activation=None
            )
            self.final_layer_patch_key = Encoder32(
                self.vit_feat_dim, cout, nf=256, activation=None
            )
        else:
            raise NotImplementedError
        self.final_layer_type = final_layer_type

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.ViT.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def forward(self, x, return_patches=False):
        b, c, h, w = x.shape
        self._feats = []
        self._register_hooks([11], "key")

        x = self.ViT.prepare_tokens(x)
        for blk in self.ViT.blocks:
            x = blk(x)
        out = self.ViT.norm(x)
        self._unregister_hooks()

        ph, pw = h // self.patch_size, w // self.patch_size
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(b, ph, pw, self.vit_feat_dim).permute(0, 3, 1, 2)

        patch_key = self._feats[0][:, :, 1:, :]  # B, num_heads, num_patches, dim
        patch_key_in = patch_key.permute(0, 1, 3, 2).reshape(
            b, self.vit_feat_dim, ph, pw
        )
        patch_key_dino = (
            patch_key.permute(0, 2, 3, 1)
            .flatten(start_dim=-2, end_dim=-1)
            .unsqueeze(dim=1)
        )  # B, 1, num_patches, (dimxhead)

        if self.final_layer_type == "none":
            global_feat_out = out[:, 0].reshape(b, -1)  # first is class token
            global_feat_key = self._feats[0][:, :, 0].reshape(
                b, -1
            )  # first is class token
        elif self.final_layer_type == "conv":
            global_feat_out = self.final_layer_patch_out(patch_out).view(b, -1)
            global_feat_key = self.final_layer_patch_key(patch_key_in).view(b, -1)
        else:
            raise NotImplementedError
        if not return_patches:
            patch_out = patch_key = None
        return global_feat_out, global_feat_key, patch_out, patch_key_in, patch_key_dino
