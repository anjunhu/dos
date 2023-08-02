"""
Code for skinning

TODO: standarize the input format 
    - still needs disenanglement between bones estimation and skinning,
        _estimate_bone_rotation should be part of the bones estimation
"""
import torch
import torch.nn as nn

from dos.nvdiffrec.render.mesh import make_mesh


def _invert_transform_mtx(mtx):
    inv_mtx = torch.eye(4)[None].repeat(len(mtx), 1, 1).to(mtx.device)
    rotation = mtx[:, :3, :3]
    translation = mtx[:, :3, 3]
    inv_mtx[:, :3, :3] = rotation.transpose(1, 2)
    inv_mtx[:, :3, 3] = -torch.bmm(
        rotation.transpose(1, 2), translation.unsqueeze(-1)
    ).squeeze(-1)
    return inv_mtx


def _estimate_bone_rotation(forward):
    """
    (0, 0, 1) = matmul(b, R^(-1))

    assumes y, z is a symmetry plane

    returns R
    """
    forward = nn.functional.normalize(forward, p=2, dim=-1)

    right = torch.FloatTensor([[1, 0, 0]]).to(forward.device)
    right = right.expand_as(forward)
    up = torch.cross(forward, right, dim=-1)
    up = nn.functional.normalize(up, p=2, dim=-1)
    right = torch.cross(up, forward, dim=-1)
    up = nn.functional.normalize(up, p=2, dim=-1)

    R = torch.stack([right, up, forward], dim=-1)
    return R


def _prepare_transform_mtx(rotation=None, translation=None):
    mtx = torch.eye(4)[None]
    if rotation is not None:
        if len(mtx) != len(rotation):
            assert len(mtx) == 1
            mtx = mtx.repeat(len(rotation), 1, 1)
        mtx = mtx.to(rotation.device)
        mtx[:, :3, :3] = rotation
    if translation is not None:
        if len(mtx) != len(translation):
            assert len(mtx) == 1
            mtx = mtx.repeat(len(translation), 1, 1)
        mtx = mtx.to(translation.device)
        mtx[:, :3, 3] = translation
    return mtx


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """[Borrowed from PyTorch3D]
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """[Borrowed from PyTorch3D]
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def skinning(
    v_pos,
    bones_pred,
    kinematic_tree,
    rots_pred,
    vertices_to_bones,  # skinnig weights
    output_posed_bones=False,
):
    device = rots_pred.device
    batch_size = rots_pred.shape[0]
    shape = v_pos

    # Rotate vertices based on bone assignments
    frame_shape_pred = []
    if output_posed_bones:
        posed_bones = bones_pred.clone()
        if posed_bones.shape[0] != batch_size:
            posed_bones = posed_bones.repeat(
                batch_size, 1, 1, 1
            )  # Shape: (B, num_bones, 2, 3)

    # Go through each bone
    for bone_id, _ in kinematic_tree:
        # Establish a kinematic chain with current bone as the leaf bone
        ## TODO: this assumes the parents is always in the front of the list
        parents_ids = [
            parent_id for parent_id, children in kinematic_tree if bone_id in children
        ]
        chain_ids = parents_ids + [bone_id]
        # Chain from leaf to root
        chain_ids = chain_ids[::-1]

        # Go through the kinematic chain from leaf to root and compose transformation
        transform_mtx = torch.eye(4)[None].to(device)
        for i in chain_ids:
            # Establish transformation
            rest_joint = bones_pred[:, i, 0, :].view(-1, 3)
            rest_bone_vector = bones_pred[:, i, 1, :] - bones_pred[:, i, 0, :]
            rest_bone_rot = _estimate_bone_rotation(rest_bone_vector.view(-1, 3))
            rest_bone_mtx = _prepare_transform_mtx(
                rotation=rest_bone_rot, translation=rest_joint
            )
            rest_bone_inv_mtx = _invert_transform_mtx(rest_bone_mtx)

            # Transform to the bone local frame
            transform_mtx = torch.matmul(rest_bone_inv_mtx, transform_mtx)

            # Rotate the mesh in the bone local frame
            rot_pred = rots_pred[:, i]
            rot_pred_mat = euler_angles_to_matrix(
                rot_pred.view(-1, 3), convention="XYZ"
            )
            rot_pred_mtx = _prepare_transform_mtx(
                rotation=rot_pred_mat, translation=None
            )
            transform_mtx = torch.matmul(rot_pred_mtx, transform_mtx)

            # Transform to the world frame
            transform_mtx = torch.matmul(rest_bone_mtx, transform_mtx)

        # Transform vertices
        shape4 = torch.cat([shape, torch.ones_like(shape[..., :1])], dim=-1)
        seq_shape_bone = torch.matmul(shape4, transform_mtx.transpose(-2, -1))[..., :3]

        if output_posed_bones:
            bones4 = torch.cat(
                [posed_bones[:, bone_id], torch.ones(batch_size, 2, 1, device=device)],
                dim=-1,
            )
            posed_bones[:, bone_id] = torch.matmul(
                bones4, transform_mtx.transpose(-2, -1)
            )[..., :3]

        # Transform mesh with weights
        frame_shape_pred += [vertices_to_bones[bone_id, ..., None] * seq_shape_bone]

    frame_shape_pred = sum(frame_shape_pred)

    aux = {}
    aux["bones_pred"] = bones_pred
    aux["vertices_to_bones"] = vertices_to_bones
    if output_posed_bones:
        aux["posed_bones"] = posed_bones

    return frame_shape_pred, aux


# TODO: should it be defined here? Also then it depends on the mesh format so problably should be moved elsewhere
def mesh_skinning(
    mesh,
    *args,
    **kwargs,
):
    verts = mesh.v_pos
    articulated_verts, aux = skinning(verts, *args, **kwargs)
    mesh = make_mesh(
        articulated_verts, mesh.t_pos_idx, mesh.v_tex, mesh.t_tex_idx, mesh.material
    )
    return mesh, aux
