import json
import os
import glob
import time

from PIL import Image
import torch
import numpy as np

import kaolin as kal
import math

# use makeit3d conda env


def render(obj_path, camera_matrix_path, out_img_path):
    mesh = kal.io.obj.import_mesh(obj_path, with_materials=True)
    # the sphere is usually too small (this is fine-tuned for the clock)
    vertices = mesh.vertices.cuda().unsqueeze(0)
    faces = mesh.faces.cuda()
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()

    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()

    # texture_res = 128
    # texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
    #                          requires_grad=True)
    
    texture_map = mesh.materials[0]['map_Kd'].float().to('cuda').permute(2, 0, 1)[None] / 255

    ### Prepare mesh data with projection regarding to camera ###
    # vertices_batch = recenter_vertices(vertices, vertice_shift)
    vertices_batch = vertices
    batch_size = 1

    cam_proj = kal.render.camera.generate_perspective_projection(math.radians(40), ratio=1.0, dtype=torch.float32).to('cuda')

    camera_matrix = np.loadtxt(camera_matrix_path)
    camera_matrix = torch.from_numpy(camera_matrix).float().to('cuda')
    # camera_matrix.inverse()

    camera_position = torch.tensor([[2.732, 0, 0]], dtype=torch.float32, device='cuda')
    look_at = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
    camera_up_direction = torch.tensor([[0, 1, 0]], dtype=torch.float32, device='cuda')
    camera_transform = kal.render.camera.generate_transformation_matrix(camera_position, look_at, camera_up_direction)

    # Convert the camera matrix from Blender to OpenGL coordinate system
    conversion_mat = torch.Tensor([[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]]).float().to('cuda')
    camera_matrix = conversion_mat @ camera_matrix

    # Convert the camera matrix to view matrix
    view_matrix = camera_matrix.inverse()[:3]

    # # Column major
    # padded_vertices = torch.nn.functional.pad(
    #     vertices, (0, 1), mode='constant', value=1.
    # )
    # Project the vertices on the camera image plan
    # padded_vertices_T = padded_vertices.permute(0, 2, 1)
    # vertices_camera = view_matrix @ padded_vertices_T
    # vertices_camera = vertices_camera.permute(0, 2, 1)

    # # Project the vertices on the camera image plan
    # vertices_image = kal.render.camera.perspective_camera(vertices_camera, cam_proj)
    # face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    # face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
    # face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

    # Row major
    # Blender is column major, but kaolin is row major
    view_matrix = view_matrix.T

    face_vertices_camera, face_vertices_image, face_normals = \
    kal.render.mesh.prepare_vertices(
        vertices_batch.repeat(batch_size, 1, 1),
        faces, cam_proj, camera_transform=view_matrix
    )

    ### Perform Rasterization ###
    # Construct attributes that DIB-R rasterizer will interpolate.
    # the first is the UVS associated to each face
    # the second will make a hard segmentation mask
    nb_faces = faces.shape[0]
    face_attributes = [
        face_uvs.repeat(batch_size, 1, 1, 1),
        torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
    ]
    # face_attributes = torch.ones((batch_size, nb_faces, 3, 3), device='cuda')

    # If you have nvdiffrast installed you can change rast_backend to
    # nvdiffrast or nvdiffrast_fwd
    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        256, 256, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1],
        rast_backend='cuda')

    # image_features is a tuple in composed of the interpolated attributes of face_attributes
    texture_coords, mask = image_features
    # texture_coords = image_features
    # image = image_features
    image = kal.render.mesh.texture_mapping(texture_coords,
                                            texture_map.repeat(batch_size, 1, 1, 1), 
                                            mode='bilinear')
    image = torch.clamp(image * mask, 0., 1.)
    # image = torch.clamp(image, 0., 1.)

    # save the image
    from PIL import Image
    image = Image.fromarray((image * 255).detach().cpu().numpy()[0, ..., :3].astype(np.uint8))
    image.save(out_img_path)


if __name__ == '__main__':
    # single image
    obj_path = 'horse_009_arabian_galgoPosesV1.obj'
    camera_matrix_path = 'camera_matrix.txt'
    out_img_path = 'kaolin.png'
    render(obj_path, camera_matrix_path, out_img_path)

    # dataset
    dataset_dir = '/scratch/local/hdd/tomj/datasets/synth_animals/renders/v1-debug'
    out_dir = '/scratch/local/hdd/tomj/synth_animals/v1-debug'
   
    for i in range(10):
        matrix = '{}/camera_{:06d}.txt'.format(dataset_dir, i)
        render(obj_path, matrix, '{}/render_{:06d}.png'.format(out_dir, i))
