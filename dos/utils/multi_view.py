import random

import numpy as np
import torch


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

def get_2_alternating_phi(device, last_phi=[None]):
    # Use a mutable default argument as a static variable to store the last state
    if last_phi[0] is None or last_phi[0] == np.deg2rad(-90):
        last_phi[0] = np.deg2rad(90)
    else:
        last_phi[0] = np.deg2rad(-90)
    return torch.tensor([last_phi[0]], dtype=torch.float, device=device)

def get_4_alternating_phi(device, last_phi=[-1]):
    # the sequence of phi values to cycle through
    phi_sequence = [np.deg2rad(90), np.deg2rad(180), np.deg2rad(-90), np.deg2rad(360)]
    
    # Use a static variable (last_phi) to store the index of the last phi value in the sequence
    # Increment the index to move to the next value in the sequence
    last_phi[0] = (last_phi[0] + 1) % len(phi_sequence)
    
    # Select the current phi value based on the updated index
    current_phi = phi_sequence[last_phi[0]]
    
    # Return a tensor containing the current phi value
    return torch.tensor([current_phi], dtype=torch.float, device=device)


def poses_helper_func(size, device, phis, thetas, radius_range=[2.5, 2.5], angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, cam_z_offset=0, return_dirs=True):
    
    angle_overhead = np.deg2rad(angle_overhead)
    
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    targets = torch.zeros(size, 3, device=device)
    
    centers = -torch.stack([
        torch.sin(thetas) * torch.sin(phis),
        torch.cos(thetas),
        torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]
    
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
    
    translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), -radius], dim=-1) 
    
    poses = torch.cat([poses.view(-1, 9), translations], dim=-1)
    
    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
        dirs = view_direction_id_to_text(dirs)
    else:
        dirs = None
    
    return poses, dirs

# Global variable to store the state
phis_state = {
    "last_phis": None,
    "last_update_iteration": -1,
    "long_update_interval": 15,  # Interval for side views
    "short_update_interval": 2,  # Interval for all other views
    "update_interval": 15
}

def initialize_phis_state(device):
    """ Initialize the phis state with a side view angle of 90 degrees. """
    global phis_state
    phis_state["last_phis"] = torch.tensor([np.deg2rad(90)], dtype=torch.float, device=device)
    phis_state["last_update_iteration"] = 0  # Assume starting at iteration 0

    
def is_side_view(phis, tolerance=np.deg2rad(1), device='cpu'):
    ninety_degrees = torch.tensor(np.deg2rad(90), dtype=torch.float, device=device)
    minus_ninety_degrees = torch.tensor(np.deg2rad(-90), dtype=torch.float, device=device)
    return torch.any(torch.isclose(phis, ninety_degrees, atol=tolerance)) or torch.any(torch.isclose(phis, minus_ninety_degrees, atol=tolerance))


def poses_along_azimuth(size, device, batch_number=0, iteration=0,  radius=2.5, theta=90, phi_range=[0, 360], multi_view_option='multiple_random_phi_in_batch', **kwargs):
    ''' generate random poses from an orbit camera along uniformly distributed azimuth and fixed elevation
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        theta: is a constant                          
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    # global phis_state  # Reference the global variable
    print('poses_along_azimuth', kwargs)
    
    if iteration == 0:
        initialize_phis_state(device)
    
    theta = np.deg2rad(theta)
    
    if multi_view_option == '2_side_views_only_in_batch':
        # Side view 1 at 90 degrees and side view 2 at -90 degrees
        phis = torch.tensor([np.deg2rad(90), np.deg2rad(-90)], dtype=torch.float, device=device)
        
    elif multi_view_option == 'guidance_and_rand_views_in_batch':
        # phis selects two values, a fixed side view (90 degrees) and a random view each iteration.
                                                # Here, 1 represents Size i.e 1 random view
        phis = torch.tensor([np.deg2rad(90), torch.rand(1, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
                            ], dtype=torch.float, device=device)  
        
    elif multi_view_option == 'rand_phi_each_step_along_azi_for_one_fixed_iter':
        # Check if it's time to update the phis value
        if iteration > 0 and iteration % phis_state["update_interval"] == 0:
            # Generate new phis and update the state
            phis = torch.rand(size, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            # Use the last stored phis value
            phis = phis_state["last_phis"]  
        print(phis, 'UPDATING PHIS at iteration', iteration, phis_state)
    
    elif multi_view_option == 'rand_phi_each_step_along_azi_long_short_update_intervals':
        
        # Usage within the update logic
        if phis_state["last_phis"] is not None:
            # Determine if the last phis value is a side view
            if is_side_view(phis_state["last_phis"], device=device):
                current_update_interval = phis_state["long_update_interval"]
            else:
                current_update_interval = phis_state["short_update_interval"]
        else:
            # No last_phis means start with a long interval assuming first is a side view
            current_update_interval = phis_state["long_update_interval"]

        # Check if it's time to update the phis value
        if iteration > 0 and iteration % current_update_interval == 0:
            # Generate new phis and update the state
            phis = torch.rand(size, device=device) * (np.deg2rad(phi_range[1]) - np.deg2rad(phi_range[0])) + np.deg2rad(phi_range[0])
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            # Use the last stored phis value
            phis = phis_state["last_phis"]
    
    elif multi_view_option == 'alternate_2_side_views_each_step_along_azimuth':
        # phis = get_2_alternating_phi(device)
        
        if iteration > 0 and iteration % phis_state["update_interval"] == 0:
            # Generate new phis and update the state
            phis = get_2_alternating_phi(device)
            
            phis_state["last_phis"] = phis
            phis_state["last_update_iteration"] = iteration
        else:
            # Use the last stored phis value
            phis = phis_state["last_phis"]
        
    elif multi_view_option == 'alternate_4_side_views_each_step_along_azimuth':
        phis = get_4_alternating_phi(device)
        
    elif multi_view_option == 'multiple_random_phi_in_batch':                       
        phi_range = np.deg2rad(phi_range)
        # For azimuth rotation (phi), will create a sequence of values within the specified range
        # Do not include endpoint (as in np.linspace) to avoid duplicate values
        phis = torch.linspace(phi_range[0], phi_range[1], steps=size+1, device=device)[:size]
        
    
    # Keeping theta (elevation angle) constant
    thetas = torch.full((size,), theta, device=device)
    
    # targets = torch.zeros(size, 3, device=device)
    
    poses, dirs = poses_helper_func(size, device, phis, thetas, radius_range=[radius, radius])
    
    return poses, dirs

def rand_poses(size, device, theta_range=[0, 120], phi_range=[0, 360], uniform_sphere_rate=0.5, radius_range=[2.5, 2.5]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    
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
    
    poses, dirs = poses_helper_func(size, device, phis, thetas, radius_range=radius_range)
    
    return poses, dirs




# Added for debugging purpose
def poses_helper_func_single_view(size, device, phis, thetas, radius_range=[1, 1], angle_overhead=30, angle_front=60, phi_offset=0, jitter=False, cam_z_offset=12, return_dirs=True):

    angle_overhead = np.deg2rad(angle_overhead)

    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    targets = torch.zeros(size, 3, device=device)

    centers = -torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]
    
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

    translations = torch.cat([torch.zeros_like(radius), torch.zeros_like(radius), radius], dim=-1) # Original

    poses = torch.cat([poses.view(-1, 9), translations], dim=-1)

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_offset=phi_offset)
        dirs = view_direction_id_to_text(dirs)
    else:
        dirs = None

    return poses, dirs

def poses_along_azimuth_single_view(size, device, theta=90, phi_range=[0, 360]):
    ''' generate random poses from an orbit camera along uniformly distributed azimuth and fixed elevation
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        theta: is a constant                          
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    theta = np.deg2rad(theta)                   
    phi_range = np.deg2rad(phi_range)
    
    # For azimuth rotation (phi), we will create a sequence of values within the specified range
    phis = torch.linspace(phi_range[0], phi_range[1], steps=size, device=device)
    
    # Keeping theta (elevation angle) constant
    thetas = torch.full((size,), theta, device=device)

    # targets = torch.zeros(size, 3, device=device)

    poses, dirs = poses_helper_func_single_view(size, device, phis, thetas)

    return poses, dirs
