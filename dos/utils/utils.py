import functools
import math
import random
from dataclasses import dataclass

#import kaolin as kal
import kornia
import numpy as np
import torch
import torchvision
from PIL import Image


def tensor_to_image(tensor, chw=True):
    """
    Convert a tensor to a PIL image
    """
    if len(tensor.shape) == 4:
        if not chw:
            tensor = tensor.permute(0, 3, 1, 2)
            chw = True
        tensor = torchvision.utils.make_grid(
            tensor, nrow=int(math.sqrt(tensor.shape[0]))
        )
    if chw:
        tensor = tensor.permute(1, 2, 0)
    return Image.fromarray((tensor * 255).detach().cpu().numpy().astype(np.uint8))


def dino_features_to_image(
    patch_key, dino_pca_mat, h=256, w=256, dino_feature_recon_dim=3
):
    """
    Convert DINO features to an image
    """
    dino_feat_im = patch_key.reshape(-1, patch_key.shape[-1]).cpu().numpy()
    dims = dino_feat_im.shape[:-1]
    dino_feat_im = dino_feat_im / np.linalg.norm(dino_feat_im, axis=1, keepdims=True)
    dino_feat_im = (
        torch.from_numpy(dino_pca_mat.apply_py(dino_feat_im))
        .to(patch_key.device)
        .reshape(*dims, -1)
    )
    dino_feat_im = (
        dino_feat_im.reshape(-1, 32, 32, dino_feat_im.shape[-1])
        .permute(0, 3, 1, 2)
        .clip(-1, 1)
        * 0.5
        + 0.5
    )
    # TODO: is it needed?
    dino_feat_im = torch.nn.functional.interpolate(
        dino_feat_im, size=[h, w], mode="bilinear"
    )[:, :dino_feature_recon_dim]
    return dino_feat_im


def blender_to_opengl(matrix):
    """
    Convert the camera matrix from Blender to OpenGL coordinate system
    """
    device = matrix.device
    # fmt: off
    conversion_mat = torch.Tensor([
                                [1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]]).float().to(device)
    # fmt: on
    return conversion_mat @ matrix


def matrix_to_rotation_translation(matrix):
    """
    matrix is a 4x4 matrix in the OpenGL coordinate system

    retruns rotation and translation, rotation is represented by a quaternion, translation is a 3D vector
    """
    rotation, translation = kornia.geometry.conversions.matrix4x4_to_Rt(
        matrix.contiguous()
    )
    rotation = kornia.geometry.conversions.rotation_matrix_to_quaternion(
        rotation.contiguous()
    )
    return rotation, translation.squeeze(2)


def rotation_translation_to_matrix(rotation, translation):
    """
    rotation is a quaternion, translation is a 3D vector

    returns matrix, which is a 4x4 matrix in the OpenGL coordinate system

    supports batched inputs
    """
    rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation)
    return kornia.geometry.conversions.Rt_to_matrix4x4(
        rotation, translation.unsqueeze(2)
    )


class MeshRenderer(object):
    def __init__(self, obj_path=None, device="cuda"):
        self.device = device
        if obj_path is not None:
            self.mesh = self.load_obj(obj_path)

    def load_obj(self, obj_path):
        mesh = kal.io.obj.import_mesh(obj_path, with_materials=True)
        vertices = mesh.vertices.to(self.device).unsqueeze(0)
        faces = mesh.faces.to(self.device)
        uvs = mesh.uvs.to(self.device).unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx.to(self.device)
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        texture_map = (
            mesh.materials[0]["map_Kd"].float().to(self.device).permute(2, 0, 1)[None]
            / 255
        )
        return {
            "vertices": vertices,
            "faces": faces,
            "face_uvs": face_uvs,
            "texture_map": texture_map,
        }

    def render(self, view_matrix, fov=50, obj_path=None):
        """
        Render the mesh using the camera matrix
        view_matrix: 4x4 matrix in the OpenGL coordinate system
        """
        if obj_path is not None:
            mesh = self.load_obj(obj_path)
        else:
            mesh = self.mesh
        vertices, faces, face_uvs, texture_map = (
            mesh[k] for k in ["vertices", "faces", "face_uvs", "texture_map"]
        )

        batch_size = view_matrix.shape[0]

        cam_proj = kal.render.camera.generate_perspective_projection(
            math.radians(fov), ratio=1.0, dtype=torch.float32
        ).to(self.device)

        # opengl is column major, but kaolin is row major
        view_matrix = view_matrix[:, :3].permute(0, 2, 1)

        (
            face_vertices_camera,
            face_vertices_image,
            face_normals,
        ) = kal.render.mesh.prepare_vertices(
            vertices.repeat(batch_size, 1, 1),
            faces,
            cam_proj,
            camera_transform=view_matrix,
        )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        nb_faces = faces.shape[0]
        face_attributes = [
            face_uvs.repeat(batch_size, 1, 1, 1),
            torch.ones((batch_size, nb_faces, 3, 1), device=self.device),
        ]

        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            256,
            256,
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_attributes,
            face_normals[:, :, -1],
            rast_backend="cuda",
        )

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        # texture_coords = image_features
        # image = image_features
        image = kal.render.mesh.texture_mapping(
            texture_coords, texture_map.repeat(batch_size, 1, 1, 1), mode="bilinear"
        )
        image = torch.clamp(image * mask, 0.0, 1.0)

        return image


@dataclass
class RandomMaskOccluder(object):
    num_occluders_range = (1, 6)
    min_size = 0.1
    max_size = 0.3

    def __call__(self, masks):
        # Get the input tensor shape
        batch_size, _, height, width = masks.shape
        masks = masks.clone()

        # Iterate over images in the batch
        # TODO: vectorize this
        for i in range(batch_size):
            num_occlusions = random.randint(*self.num_occluders_range)
            # Create multiple occlusions per image
            min_size = int(self.min_size * min(height, width))
            max_size = int(self.max_size * min(height, width))
            for _ in range(num_occlusions):
                # Define occlusion size
                occlusion_size_x = random.randint(min_size, max_size)
                occlusion_size_y = random.randint(min_size, max_size)

                # Define occlusion position
                occlusion_x = random.randint(0, width - occlusion_size_x)
                occlusion_y = random.randint(0, height - occlusion_size_y)

                # Create occlusion on all channels
                masks[
                    i,
                    :,
                    occlusion_y : occlusion_y + occlusion_size_y,
                    occlusion_x : occlusion_x + occlusion_size_x,
                ] = 0

        return masks


def rgetattr(obj, attr, *args):
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def safe_batch_to_device(batch, *args, **kwargs):
    out_batch = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out_batch[k] = v.to(*args, **kwargs)
        else:
            out_batch[k] = v
    return out_batch


# ADDED FOR MULTI-VIEW/3D

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front, phi_offset=0):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [360 - front / 2, front / 2)
    # side (left) = 1   [front / 2, 180 - front / 2)
    # back = 2          [180 - front / 2, 180 + front / 2)
    # side (right) = 3  [180 + front / 2, 360 - front / 2)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)

    # first determine by phis
    phi_offset = np.deg2rad(phi_offset)
    phis = phis + phi_offset
    phis = phis % (2 * np.pi)
    half_front = front / 2
    
    res[(phis >= (2*np.pi - half_front)) | (phis < half_front)] = 0
    res[(phis >= half_front) & (phis < (np.pi - half_front))] = 1
    res[(phis >= (np.pi - half_front)) & (phis < (np.pi + half_front))] = 2
    res[(phis >= (np.pi + half_front)) & (phis < (2*np.pi - half_front))] = 3

    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def view_direction_id_to_text(view_direction_id):
    dir_texts = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
    return [dir_texts[i] for i in view_direction_id]


# original
# rand_poses(size, device, radius_range=[1, 1], theta_range=[0, 120], phi_range=[0, 360], cam_z_offset=10, return_dirs=True, angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, uniform_sphere_rate=0.5):
    
def rand_poses(size, device, radius_range=[1, 1], theta_range=[0, 180], phi_range=[0, 0], cam_z_offset=10, return_dirs=True, angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_range: [min, max]
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    if random.random() < uniform_sphere_rate:
        # based on http://corysimon.github.io/articles/uniformdistn-on-sphere/
        # acos takes in [-1, 1], first convert theta range to fit in [-1, 1] 
        theta_range = torch.from_numpy(np.array(theta_range)).to(device)
        theta_amplitude_range = torch.cos(theta_range)
        # sample uniformly in amplitude space range
        thetas_amplitude = torch.rand(size, device=device) * (theta_amplitude_range[1] - theta_amplitude_range[0]) + theta_amplitude_range[0]
        # convert back
        thetas = torch.acos(thetas_amplitude)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]

    centers = -torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1) + up_noise)

    poses = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
    radius = radius[..., None] - cam_z_offset
    translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), radius], dim=-1)
    poses = torch.cat([poses.view(-1, 9), translations], dim=-1)

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
        dirs = view_direction_id_to_text(dirs)
    else:
        dirs = None
    
    return poses, dirs