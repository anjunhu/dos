import bpy
import numpy as np
import math
import mathutils

from blender_utils import clean_up_scene, setup_renderer

clean_up_scene()
setup_renderer()

# Load the OBJ file
obj_file = 'horse_009_arabian_galgoPosesV1.obj'
imported_object = bpy.ops.import_scene.obj(filepath=obj_file)

obj_object = bpy.context.selected_objects[0]

# Add a new camera
bpy.ops.object.camera_add(location=(0, 0, 0))
cam = bpy.context.object

# Convert distance, elevation and azimuth to cartesian coordinates
distance = 5
elevation = math.radians(0)
azimuth = math.radians(0)

x = distance * math.cos(elevation) * math.cos(azimuth)
y = distance * math.cos(elevation) * math.sin(azimuth)
z = distance * math.sin(elevation)

cam_location = mathutils.Vector((x, y, z))

# Look at the object
direction = obj_object.location - cam_location
rot_matrix = direction.to_track_quat('-Z', 'Y').to_matrix().to_4x4()

# Set the camera rotation
cam.matrix_world = rot_matrix
cam.matrix_world.translation = cam_location
# cam.rotation_euler = rot_quat.to_euler()

# Set the FOV
fov = 50
cam.data.angle = math.radians(fov)

# Set the camera as the active camera
bpy.context.scene.camera = cam

# Render the scene
bpy.context.scene.render.filepath = 'blender-render.png'
bpy.ops.render.render(write_still=True)

# Save the camera matrix
camera_matrix = cam.matrix_world
np.savetxt('camera_matrix.txt', camera_matrix)
