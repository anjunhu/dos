import torch
import torchvision
import numpy as np
from pathlib import Path
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex, PointLights
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)


dataset_dir = '/scratch/local/hdd/tomj/datasets/synth_animals/renders/v1-debug'
obj_path = 'horse_009_arabian_galgoPosesV1.obj'

out_dir = '/scratch/local/hdd/tomj/synth_animals/v1-debug'

# Load the 3D model
verts, faces, aux = load_obj("horse_009_arabian_galgoPosesV1.obj")
# Create a brown texture
# Create a brown color
brown_color = torch.tensor([0.65, 0.16, 0.16], device=verts.device)
textures_color = brown_color[None, None].repeat(1, verts.shape[0], 1)
# Create a brown texture
textures = TexturesVertex(verts_features=textures_color)

mesh = Meshes(
    verts=[verts],   
    faces=[faces.verts_idx], 
    textures=textures
)
mesh = mesh.cuda()

device = 'cuda'

# mesh = load_objs_as_meshes(['data/cow_mesh/cow.obj'], device=device)


# Load camera matrices
# eg. camera_000000.txt
matrices = []
for i in range(10):
    matrix = np.loadtxt('{}/camera_{:06d}.txt'.format(dataset_dir, i))
    matrix = torch.Tensor(matrix).cuda()
    matrices.append(matrix)

# Define the conversion matrix
conversion_mat = torch.Tensor([[-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]]).float().cuda()

# Set renderer settings
raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)

R, T = look_at_view_transform(4, 0, 0) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Set up renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

Path(out_dir).mkdir(parents=True, exist_ok=True)    

for i, matrix in enumerate(matrices):
    # Convert the camera matrix from Blender to pytorch3d coordinate system

    c2w = matrix
    swap_y_z = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])
    swap_y_z = torch.from_numpy(swap_y_z).float().to(device)

    deg180 = np.deg2rad(180)
    rot_z = np.array([[np.cos(deg180), -np.sin(deg180), 0],
            [np.sin(deg180), np.cos(deg180), 0],
            [0, 0, 1]])
    rot_z = torch.from_numpy(rot_z).float().to(device)

    c2w = swap_y_z @ c2w

    T = c2w[:3,-1]  # Extract translation of the camera
    R = c2w[:3, :3] @ rot_z # Extract rotation matrix of the camera

    T = T @ R # Make rotation local
    R = R[None]  # Add batch dimension
    T = T[None]  # Add batch dimension

    # matrix = conversion_mat @ matrix
    
    # # Extract camera parameters
    # R = matrix[:3, :3][None]
    # T = matrix[:3, 3][None]

    # T = R.inverse() @ T.T
    # T = T.squeeze(2)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Set up renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    # Render the mesh
    # images = renderer(mesh, R=R[None], T=T[None])
    images = renderer(mesh)

    # Save the image
    # Path(out_dir) / 'render_{i:06d}.png'
    images = images.permute(0, 3, 1, 2)
    torchvision.utils.save_image(images[0], '{}/render_{:06d}.png'.format(out_dir, i))
