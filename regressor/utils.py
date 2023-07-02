import torch


from PIL import Image
import torch
import numpy as np

import kaolin as kal
import math
import torch
import kornia


def tensor_to_image(tensor, no_permute=False):
    """
    Convert a tensor to a PIL image
    """
    if not no_permute:
        tensor = tensor.permute(1, 2, 0)
    return Image.fromarray((tensor * 255).detach().cpu().numpy().astype(np.uint8))


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

        batch_size = 1

        cam_proj = kal.render.camera.generate_perspective_projection(
            math.radians(fov), ratio=1.0, dtype=torch.float32
        ).to(self.device)

        # opengl is column major, but kaolin is row major
        view_matrix = view_matrix[:3].T

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
