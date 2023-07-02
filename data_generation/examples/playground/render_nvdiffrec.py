

import sys
nvdiffrec_path = '/users/tomj/nvdiffrec' 
if nvdiffrec_path not in sys.path:
    sys.path.insert(0, nvdiffrec_path)
from nvdiffrec.render import mesh, util
from nvdiffrec.dataset.dataset_mesh import DatasetMesh

import nvdiffrast.torch as dr

import argparse
import os
import torch
import json
import numpy as np


parser = argparse.ArgumentParser(description='nvdiffrec')
parser.add_argument('--config', type=str, default=None, help='Config file')
parser.add_argument('-i', '--iter', type=int, default=5000)
parser.add_argument('-b', '--batch', type=int, default=1)
parser.add_argument('-s', '--spp', type=int, default=1)
parser.add_argument('-l', '--layers', type=int, default=1)
parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
parser.add_argument('-dr', '--display-res', type=int, default=None)
parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
parser.add_argument('-di', '--display-interval', type=int, default=0)
parser.add_argument('-si', '--save-interval', type=int, default=1000)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
parser.add_argument('-o', '--out-dir', type=str, default=None)
parser.add_argument('-rm', '--ref_mesh', type=str)
parser.add_argument('-bm', '--base-mesh', type=str, default=None)
parser.add_argument('--validate', type=bool, default=True)

FLAGS = parser.parse_args()

FLAGS.mtl_override        = None                     # Override material of model
FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
FLAGS.envmap              = None                     # HDR environment probe
FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
FLAGS.lock_light          = False                    # Disable light optimization in the second pass
FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
FLAGS.cam_near_far        = [0.1, 1000.0]
FLAGS.learn_light         = True

FLAGS.local_rank = 0
FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
if FLAGS.multi_gpu:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = 'localhost'
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = '23456'

    FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(FLAGS.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

if FLAGS.config is not None:
    data = json.load(open(FLAGS.config, 'r'))
    for key in data:
        FLAGS.__dict__[key] = data[key]

if FLAGS.display_res is None:
    FLAGS.display_res = FLAGS.train_res

if FLAGS.local_rank == 0:
    print("Config / Flags:")
    print("---------")
    for key in FLAGS.__dict__.keys():
        print(key, FLAGS.__dict__[key])
    print("---------")


class FixedDirectionLight(torch.nn.Module):
    def __init__(self, direction, amb, diff):
        super(FixedDirectionLight, self).__init__()
        self.light_dir = direction
        self.amb = amb
        self.diff = diff
        self.is_hacking = not (isinstance(self.amb, float) or isinstance(self.amb, int))

    def forward(self, feat):
        batch_size = feat.shape[0]
        if self.is_hacking:
            return torch.concat([self.light_dir, self.amb, self.diff], -1)
        else:
            return torch.concat([self.light_dir, torch.FloatTensor([self.amb, self.diff]).to(self.light_dir.device)], -1).expand(batch_size, -1)

    def shade(self, feat, kd, normal):
        light_params = self.forward(feat)
        light_dir = light_params[..., :3][:, None, None, :]
        int_amb = light_params[..., 3:4][:, None, None, :]
        int_diff = light_params[..., 4:5][:, None, None, :]
        shading = (int_amb + int_diff * torch.clamp(util.dot(light_dir, normal), min=0.0))
        shaded = shading * kd
        return shaded, shading


glctx = dr.RasterizeGLContext()

ref_mesh = '/users/tomj/nvdiffrec/data/bob/bob_tri.obj'

ref_mesh = mesh.load_mesh(ref_mesh)


glctx = dr.RasterizeGLContext()


RADIUS = 3.0

device = 'cuda'

gray_light = FixedDirectionLight(direction=torch.FloatTensor([0, 0, 1]).to(device), amb=0.2, diff=0.7)

dataset_train = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)

sample = dataset_train[0]
img = sample['img']
# save the image
from PIL import Image
img = Image.fromarray(img.detach().cpu().numpy()[0, ..., :3].astype(np.uint8))
img.save('test.png')

print()
