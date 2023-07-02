import torch
import numpy as np
import torchvision


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


# Load the saved camera matrix
camera_matrix = np.loadtxt('camera_matrix.txt')
camera_matrix = torch.from_numpy(camera_matrix).float()

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

# Convert Blender camera matrix to PyTorch3D format
R = camera_matrix[:3, :3][None]  # rotation
T = camera_matrix[:3, 3]  # translation
T = T[None]  # Add batch dimension


print(R.round())
print(T)

# Define the conversion matrix
conversion_mat = torch.Tensor([[-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]]).float()

# Convert the camera matrix from Blender to Pytorch coordinate system
matrix = conversion_mat @ camera_matrix

# Get a view matrix from the camera matrix
view_matrix = camera_matrix.inverse()

# Pytorch is row major, Blender is column major
view_matrix = view_matrix.T

# Extract camera parameters
R = view_matrix[:3, :3][None]
T = view_matrix[3, :3][None]

# Rotate the translation vector
# T = R.inverse() @ T.T
# T = T.squeeze(2)
# print(R.round())
# print(T)

R, T = look_at_view_transform(5, 0, 0)
import ipdb; ipdb.set_trace()
# print(R.round())
# print(T)

# Set renderer settings
raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)

# Create a FoV camera from the camera matrix
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
# cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
cameras.get_world_to_view_transform()

import ipdb; ipdb.set_trace()

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

# Render the mesh
images = renderer(mesh)
images = images.permute(0, 3, 1, 2)
torchvision.utils.save_image(images[0], 'pytorch-render.png')
