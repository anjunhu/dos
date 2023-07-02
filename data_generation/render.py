import bpy
import os
import random
import numpy as np
import math
import mathutils
import glob

# Constants
RADIUS = 5  # Radius of the sphere
UP = (0, 0, 1)  # Up direction


file_path = "/scratch/local/hdd/tomj/datasets/synth_animals/data/DOC/3dModels/horse/02_released/horse_009_arabian_galgoPosesV1.glb"
texture_path = "/scratch/local/hdd/tomj/datasets/synth_animals/data/DOC/maps/frankensteinDiffuses_v001/diffuse_horse_*.jpg"
out_dir = "/scratch/local/hdd/tomj/datasets/synth_animals/renders/v1-debug"
n_renders = 10
random_frame = False
fov = 50


def clean_up_scene():
    # Clear all Meshes
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    # Clear all armature objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="ARMATURE")
    bpy.ops.object.delete()

    # Clear all Cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()


def setup_renderer():
    # Set render engine to CYCLES
    bpy.context.scene.render.engine = "CYCLES"

    # Enable GPU
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or 'OPENCL' or 'NONE'

    # Get list of all the available devices (GPUs in this case)
    devices = bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Enable all GPU devices
    for device in devices:
        for subdevice in device:
            subdevice.use = True

    # Set device to GPU
    bpy.context.scene.cycles.device = "GPU"

    # Set render settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256


clean_up_scene()
setup_renderer()

# Find texture files matching the pattern
texture_files = glob.glob(texture_path)

# Import GLB model
bpy.ops.import_scene.gltf(filepath=file_path)
object = bpy.context.object

# Traverse the object hierarchy to find the mesh object
mesh_obj = None
for obj in object.children:
    if obj.type == "MESH":
        mesh_obj = obj
        break

# Get the material from the slot
material = mesh_obj.data.materials[0]

# Enable 'Use Nodes'
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]

# Specular to 0
bsdf.inputs[5].default_value = 0

# Add a texture node and set the texture image
texImage = material.node_tree.nodes.new("ShaderNodeTexImage")

# Set the second UV map as the active one for texture mapping
uv_map_name = mesh_obj.data.uv_layers[1].active_render = True

# Link texture node to BSDF node
material.node_tree.links.new(bsdf.inputs["Base Color"], texImage.outputs["Color"])

bpy.data.lights["Light"].energy = 1200

# Add a camera
bpy.ops.object.camera_add(location=(0, -3, 0))
camera = bpy.context.object

# Set the active camera
bpy.context.scene.camera = camera

random.seed(0)

# Render from n different random views
for i in range(n_renders):
    # Random light

    # Clear all lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Add a new point light
    bpy.ops.object.light_add(type="POINT")
    point_light = bpy.context.object

    # Set initial light location
    point_light.location = (0, 0, 0)

    # Sample random position for point light
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    z = random.uniform(0, 10)

    # Change the light location
    point_light.location = (x, y, z)

    # Sample random energy for the light
    min_energy = 500  # Minimum energy value
    max_energy = 2000  # Maximum energy value
    random_energy = random.uniform(min_energy, max_energy)

    point_light.data.energy = random_energy

    # Choose a random frame between 0 and 100
    if random_frame:
        frame = random.randint(0, 100)
        bpy.context.scene.frame_set(frame)

    # Random texture
    texImage.image = bpy.data.images.load(random.choice(texture_files))

    # Randomly sample spherical coordinates in top half sphere and convert to cartesian
    theta = 2 * math.pi * random.random()  # Random angle around z-axis
    phi = math.acos(
        1 - random.random()
    )  # Random angle from positive z-axis (top half sphere)

    x = RADIUS * math.sin(phi) * math.cos(theta)
    y = RADIUS * math.sin(phi) * math.sin(theta)
    z = RADIUS * math.cos(phi)

    # Get the bounding box center (object's local coordinates)
    bbox_center_local = (
        sum((mathutils.Vector(b) for b in object.bound_box), mathutils.Vector()) / 8
    )
    # Convert to global coordinates
    bbox_center_global = object.matrix_world @ bbox_center_local

    # Compute direction vector from camera to origin
    print(bbox_center_global)
    direction = bbox_center_global - mathutils.Vector((x, y, z))

    # Compute the rotation matrix to align the -Z axis with the direction vector
    rot_matrix = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()

    # Set the camera rotation
    camera.matrix_world = rot_matrix

    # Set the camera position
    camera.matrix_world.translation = mathutils.Vector((x, y, z))

    # Set the FOV
    camera.data.angle = math.radians(fov)

    # Set the output file path
    bpy.context.scene.render.filepath = os.path.join(out_dir, f"render_{i:06d}.png")

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Get the camera matrix
    camera_matrix = camera.matrix_world

    # Convert the camera matrix to a numpy array
    camera_matrix_array = np.array(camera_matrix)

    # Define the file path for saving the camera matrix
    file_path = os.path.join(out_dir, f"camera_{i:06d}.txt")

    # Save the camera matrix as a text file
    np.savetxt(file_path, camera_matrix_array, fmt="%f", delimiter=" ")
