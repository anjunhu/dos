import bpy


def clean_up_scene():
    # Clear all Meshes
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Clear all armature objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='ARMATURE')
    bpy.ops.object.delete()

    # Clear all Cameras
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()


def setup_renderer():
    # Set render engine to CYCLES
    bpy.context.scene.render.engine = 'CYCLES'

    # Enable GPU
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA' # or 'OPENCL' or 'NONE' 

    # Get list of all the available devices (GPUs in this case)
    devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Enable all GPU devices
    for device in devices:
        for subdevice in device:
            subdevice.use = True

    # Set device to GPU
    bpy.context.scene.cycles.device = 'GPU'

    # Set render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256


clean_up_scene()
setup_renderer()
