import kornia
import numpy as np
import torch
from einops import repeat

from ..nvdiffrec.render import renderutils
from ..nvdiffrec.render import util as render_util
from ..utils import utils


def sample_farthest_points(pts, k, return_index=False):
    b, c, n = pts.shape
    farthest_pts = torch.zeros((b, 3, k), device=pts.device, dtype=pts.dtype) 
    indexes = torch.zeros((b, k), device=pts.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=pts.device)

    gather_index = repeat(index, "b -> b c 1", c=c)
    farthest_pts[:, :, 0] = torch.gather(pts, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    distances = torch.norm(farthest_pts[:, :, 0][:, :, None] - pts, dim=1)

    for i in range(1, k):
        _, index = torch.max(distances, dim=1)
        gather_index = repeat(index, "b -> b c 1", c=c)
        farthest_pts[:, :, i] = torch.gather(pts, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        distances = torch.min(
            distances, torch.norm(farthest_pts[:, :, i][:, :, None] - pts, dim=1)
        )

    if return_index:
        return farthest_pts, indexes
    else:
        return farthest_pts


def line_segment_distance(a, b, points, sqrt=True):
    """
    compute the distance between a point and a line segment defined by a and b
    a, b: ... x D
    points: ... x D
    """

    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    a, b = a[..., None, :], b[..., None, :]

    t_min = sumprod(points - a, b - a) / torch.max(
        sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device)
    )

    t_line = torch.clamp(t_min, 0.0, 1.0)

    # closest points on the line to every point
    s = a + t_line * (b - a)

    distance = sumprod(s - points, s - points, keepdim=False)

    if sqrt:
        distance = torch.sqrt(distance + 1e-6)

    return distance


def angle_axis_to_rotation_matrix(axis, angle):
    """Converts an axis-angle rotation B to a Bx4x4 rotation matrix.
    axis = 'x', 'y', 'z'
    angle = angle in radians tesnor of shape B
    """
    # angle_axis of shape (B, 3)
    b = angle.shape[0]
    angle_axis = torch.zeros((b, 3)).to(angle.device)
    if axis == "x":
        angle_axis[:, 0] = angle
    elif axis == "y":
        angle_axis[:, 1] = angle
    elif axis == "z":
        angle_axis[:, 2] = angle
    # rot_mat of shape (B, 3, 3)
    rot_mat = kornia.geometry.conversions.angle_axis_to_rotation_matrix(angle_axis)
    # rot_mat of shape (B, 4, 4)
    rot_mat = torch.cat([rot_mat, torch.zeros((b, 3, 1)).to(angle.device)], dim=2)
    rot_mat = torch.cat([rot_mat, torch.zeros((b, 1, 4)).to(angle.device)], dim=1)
    rot_mat[:, 3, 3] = 1
    return rot_mat


# TODO: clean the camera functions
# TODO: identify the best locations for these functions
# TODO: identify better names
def get_camera_extrinsics_from_pose(pose, cam_pos_z_offset=0):
    N = len(pose)
    cam_pos_offset = torch.FloatTensor([0, 0, -cam_pos_z_offset]).to(pose.device)
    pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)
    pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
    pose_T = pose_T.view(N, 3, 1)
    pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
    w2c = torch.cat(
        [pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)],
        axis=1,
    )  # Nx4x4
    campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
    return w2c, campos


def get_mvp_from_w2c(w2c, crop_fov_approx, znear=0.1, zfar=1000.0):
    device = w2c.device
    # We assume the images are perfect square.
    if isinstance(crop_fov_approx, float) or isinstance(crop_fov_approx, int):
        proj = render_util.perspective(crop_fov_approx / 180 * np.pi, 1, znear, zfar)[
            None
        ].to(device)
    elif isinstance(crop_fov_approx, torch.Tensor):
        proj = render_util.batched_perspective(
            crop_fov_approx / 180 * np.pi, 1, znear, zfar
        ).to(device)
    else:
        raise ValueError("crop_fov_approx must be a float or a tensor")
    mvp = torch.matmul(proj, w2c)
    return mvp


def get_camera_extrinsics_and_mvp_from_pose(
    pose, crop_fov_approx, znear=0.1, zfar=1000.0, cam_pos_z_offset=0
):
    w2c, campos = get_camera_extrinsics_from_pose(
        pose, cam_pos_z_offset=cam_pos_z_offset
    )
    mvp = get_mvp_from_w2c(w2c, crop_fov_approx, znear, zfar)

    return mvp, w2c, campos


def project_points(v_pos, mvp):
    v_pos_clip4 = renderutils.xfm_points(v_pos, mvp)
    v_pos_uv = v_pos_clip4[..., :2] / v_pos_clip4[..., 3:]
    return v_pos_uv


def blender_camera_matrix_to_magicpony_pose(camera_matrix):
    # convert the camera matrix from Blender to OpenGL coordinate system
    camera_matrix = utils.blender_to_opengl(camera_matrix)
    # convert the camera matrix to view matrix
    view_matrix = camera_matrix.inverse()
    # magicpony pose is the transpose of the view matrix
    pose = view_matrix.transpose(2, 1)
    pose = pose[:, :, :3].reshape(camera_matrix.shape[0], -1)
    return pose
