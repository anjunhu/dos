"""
Code for estimating the bones and skinning weights from a mesh.

TODO: standarize the ouput format
    - stills need disenanglement between bones estimation and skinning,
        _estimate_bone_rotation should be part of the bones estimation
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from dos.utils import geometry as geometry_utils


def _joints_to_bones(joints, bones_idxs):
    bones = []
    for a, b in bones_idxs:
        bones += [torch.stack([joints[:, a, :], joints[:, b, :]], dim=1)]
    bones = torch.stack(bones, dim=1)
    return bones


def _build_kinematic_chain(n_bones, start_bone_idx):
    # Build bones and kinematic chain starting from leaf bone (body joint)
    bones_to_joints = []
    kinematic_chain = []
    bone_idx = start_bone_idx
    # Bones from leaf to root
    dependent_bones = []
    for i in range(n_bones):
        bones_to_joints += [(i + 1, i)]
        kinematic_chain = [
            (bone_idx, dependent_bones)
        ] + kinematic_chain  # Parent is always in the front
        dependent_bones = dependent_bones + [bone_idx]
        bone_idx += 1
    return bones_to_joints, kinematic_chain, dependent_bones


def _update_body_kinematic_chain(
    kinematic_chain,
    leg_kinematic_chain,
    body_bone_idx,
    leg_bone_idxs,
    attach_legs_to_body=True,
):
    if attach_legs_to_body:
        for bone_idx, dependent_bones in kinematic_chain:
            if bone_idx == body_bone_idx or body_bone_idx in dependent_bones:
                dependent_bones += leg_bone_idxs
    kinematic_chain = (
        kinematic_chain + leg_kinematic_chain
    )  # Parent is always in the front
    return kinematic_chain


def estimate_bones(
    seq_shape,
    n_body_bones,
    resample=False,
    n_legs=4,
    n_leg_bones=0,
    body_bones_type="z_minmax",
    compute_kinematic_chain=True,
    aux=None,
    attach_legs_to_body=True,
    detached=True,
):
    """
    Estimate the position and structure of bones given the mesh vertex positions.

    Args:
        seq_shape: a tensor of shape (B, F, V, 3), the batched position of mesh vertices.
        n_body_bones: an integer, the desired number of bones.
    Returns:
        (bones_pred, kinematic_chain) where
        bones_pred: a tensor of shape (B, num_bones, 2, 3)
        kinematic_chain: a list of tuples of length n_body_bones; for each tuple, the first element is the bone index while
                         the second element is a list of bones indices of dependent bones.
    """
    if detached:
        seq_shape = seq_shape.detach()

    # preprocess shape
    if resample:
        b, n, _ = seq_shape.shape
        seq_shape = geometry_utils.sample_farthest_points(
            rearrange(seq_shape, "b f n d -> (b f) d n"), n // 4
        )
        seq_shape = rearrange(seq_shape, "(b f) d n -> b f n d", b=b)

    if body_bones_type == "z_minmax":
        # FIXME: Remove frame dimension!!!
        raise NotImplementedError()
        indices_max, indices_min = seq_shape[..., 2].argmax(dim=1), seq_shape[
            ..., 2
        ].argmin(dim=1)
        indices = torch.cat([indices_max[..., None], indices_min[..., None]], dim=2)
        indices_gather = indices[..., None].repeat(1, 1, 1, 3)  # Shape: (B, F, 2, 3)
        points = seq_shape.gather(2, indices_gather)
        point_a = points[:, :, 0, :]
        point_b = points[:, :, 1, :]
    elif body_bones_type == "z_minmax_y+":
        ## TODO: mean may not be very robust, as inside is noisy
        mid_point = seq_shape.mean(1)
        seq_shape_pos_y_mask = (
            seq_shape[..., 1] > (mid_point[:, None, 1] - 0.5)
        ).float()  # y higher than midpoint
        seq_shape_z = seq_shape[..., 2] * seq_shape_pos_y_mask - 1e6 * (
            1 - seq_shape_pos_y_mask
        )
        indices = seq_shape_z.argmax(1)
        indices_gather = indices[:, None, None].repeat(1, 1, 3)
        point_a = seq_shape.gather(1, indices_gather).squeeze(1)
        seq_shape_z = seq_shape[..., 2] * seq_shape_pos_y_mask + 1e6 * (
            1 - seq_shape_pos_y_mask
        )
        indices = seq_shape_z.argmin(1)
        indices_gather = indices[:, None, None].repeat(1, 1, 3)
        point_b = seq_shape.gather(1, indices_gather).squeeze(1)
    else:
        raise NotImplementedError

    # Place points on the symmetry axis.
    point_a[:, 0] = 0  # Shape: (B, 3)
    point_b[:, 0] = 0  # Shape: (B, 3)

    mid_point = seq_shape.mean(1)  # Shape: (B, 3)
    # Place points on the symmetry axis
    mid_point[:, 0] = 0
    if n_leg_bones > 0:
        # Lift mid point a bit higher if there are legs
        mid_point[:, 1] += 0.5

    assert n_body_bones % 2 == 0
    n_joints = n_body_bones + 1
    blend = torch.linspace(0.0, 1.0, math.ceil(n_joints / 2), device=point_a.device)[
        None, :, None
    ]  # Shape: (1, (n_joints + 1) / 2, 1)
    joints_a = (
        point_a[:, None, :] * (1 - blend) + mid_point[:, None, :] * blend
    )  # Point a to mid point
    joints_b = point_b[:, None, :] * blend + mid_point[:, None, :] * (
        1 - blend
    )  # Mid point to point b
    joints = torch.cat([joints_a[:, :-1], joints_b], 1)  # Shape: (B, n_joints, 3)

    # build bones and kinematic chain starting from leaf bones
    if compute_kinematic_chain:
        aux = {}
        half_n_body_bones = n_body_bones // 2
        bones_to_joints = []
        kinematic_chain = []
        bone_idx = 0
        # bones from point_a to mid_point
        dependent_bones = []
        for i in range(half_n_body_bones):
            bones_to_joints += [(i + 1, i)]
            kinematic_chain = [
                (bone_idx, dependent_bones)
            ] + kinematic_chain  # parent is always in the front
            dependent_bones = dependent_bones + [bone_idx]
            bone_idx += 1
        # bones from point_b to mid_point
        dependent_bones = []
        for i in range(n_body_bones - 1, half_n_body_bones - 1, -1):
            bones_to_joints += [(i, i + 1)]
            kinematic_chain = [
                (bone_idx, dependent_bones)
            ] + kinematic_chain  # parent is always in the front
            dependent_bones = dependent_bones + [bone_idx]
            bone_idx += 1
        aux["bones_to_joints"] = bones_to_joints
    else:
        bones_to_joints = aux["bones_to_joints"]
        kinematic_chain = aux["kinematic_chain"]

    bones_pred = _joints_to_bones(joints, bones_to_joints)  # Shape: (B, n_bones, 2, 3)

    if n_leg_bones > 0:
        assert n_legs == 4
        # attach four legs
        # y, z is symetry plain
        # y axis is up
        #
        # top down view:
        #
        #          |
        #      2   |   1
        #   -------|------ > x
        #      3   |   0
        #          ⌄
        #          z
        #
        # find a point with the lowest y in each quadrant
        xs, ys, zs = seq_shape.unbind(-1)
        x_margin = (xs.quantile(0.95) - xs.quantile(0.05)) * 0.2
        quadrant0 = torch.logical_and(xs > x_margin, zs > 0)
        quadrant1 = torch.logical_and(xs > x_margin, zs < 0)
        quadrant2 = torch.logical_and(xs < -x_margin, zs < 0)
        quadrant3 = torch.logical_and(xs < -x_margin, zs > 0)

        def find_leg_in_quadrant(quadrant, n_bones, body_bone_idx):
            batch_size = seq_shape.shape[0]
            all_joints = torch.zeros(
                [batch_size, n_bones + 1, 3],
                dtype=seq_shape.dtype,
                device=seq_shape.device,
            )
            for b in range(batch_size):
                # Find a point with the lowest y
                quadrant_points = seq_shape[b][quadrant[b]]
                assert (
                    len(quadrant_points.view(-1)) > 0
                ), "No vertices in the quadrant. Something is very wrong!"
                idx = torch.argmin(quadrant_points[:, 1])  # Lowest y
                foot = quadrant_points[idx]  # Shape: (3,)

                # Find closest point on the body joints (the end joint of the leg)
                if body_bone_idx is None:
                    body_bone_idx = int(
                        torch.argmin((bones_pred[b, :, 1, 2] - foot[None, 2]).abs())
                    )  # Closest in z axis
                body_joint = bones_pred[b, body_bone_idx, 1]  # Shape: (3,)

                # Create bone structure from the foot to the body joint
                blend = torch.linspace(0.0, 1.0, n_bones + 1, device=seq_shape.device)[
                    :, None
                ]  # Shape: (n_bones + 1, 1)
                joints = foot[None] * (1 - blend) + body_joint[None] * blend
                all_joints[b] = joints
            return all_joints, body_bone_idx

        quadrants = [quadrant0, quadrant1, quadrant2, quadrant3]
        body_bone_idxs = [None, None, None, None]
        # body_bone_idxs = [2, 6, 6, 2]
        body_bone_idxs = [2, 7, 7, 2]
        # body_bone_idxs = [3, 6, 6, 3]
        start_bone_idx = n_body_bones
        all_leg_bones = []
        if compute_kinematic_chain:
            leg_auxs = []
        else:
            leg_auxs = aux["legs"]
        for i, quadrant in enumerate(quadrants):
            if compute_kinematic_chain:
                leg_i_aux = {}
                body_bone_idx = body_bone_idxs[i]

                leg_joints, body_bone_idx = find_leg_in_quadrant(
                    quadrant, n_leg_bones, body_bone_idx=body_bone_idx
                )
                body_bone_idxs[i] = body_bone_idx

                (
                    leg_bones_to_joints,
                    leg_kinematic_chain,
                    leg_bone_idxs,
                ) = _build_kinematic_chain(n_leg_bones, start_bone_idx=start_bone_idx)
                kinematic_chain = _update_body_kinematic_chain(
                    kinematic_chain,
                    leg_kinematic_chain,
                    body_bone_idx,
                    leg_bone_idxs,
                    attach_legs_to_body=attach_legs_to_body,
                )
                leg_i_aux["body_bone_idx"] = body_bone_idx
                leg_i_aux["leg_bones_to_joints"] = leg_bones_to_joints
                start_bone_idx += n_leg_bones
            else:
                leg_i_aux = leg_auxs[i]
                body_bone_idx = leg_i_aux["body_bone_idx"]
                leg_joints, _ = find_leg_in_quadrant(
                    quadrant, n_leg_bones, body_bone_idx
                )
                leg_bones_to_joints = leg_i_aux["leg_bones_to_joints"]
            leg_bones = _joints_to_bones(leg_joints, leg_bones_to_joints)
            all_leg_bones += [leg_bones]
            if compute_kinematic_chain:
                leg_auxs += [leg_i_aux]

        all_bones = [bones_pred] + all_leg_bones
        all_bones = torch.cat(all_bones, dim=1)
    else:
        all_bones = bones_pred

    if detached:
        all_bones = all_bones.detach()

    if compute_kinematic_chain:
        aux["kinematic_chain"] = kinematic_chain
        if n_leg_bones > 0:
            aux["legs"] = leg_auxs
        return all_bones, kinematic_chain, aux
    else:
        return all_bones


def estimate_skinning_weigths(bones_pred, seq_shape_pred, temperature=1):
    """
    Make sure seq_shape_pred is detached.
    """
    vertices_to_bones = []
    for i in range(bones_pred.shape[1]):
        vertices_to_bones += [
            geometry_utils.line_segment_distance(
                bones_pred[:, i, 0], bones_pred[:, i, 1], seq_shape_pred
            )
        ]
    vertices_to_bones = nn.functional.softmax(
        -torch.stack(vertices_to_bones, dim=0) / temperature, dim=0
    )
    return vertices_to_bones


# TODO: should it be defined here?
@dataclass
class BonesEstimator(object):
    num_body_bones: int
    resample: bool = False
    num_legs: int = 4
    num_leg_bones: int = 0
    body_bones_type: str = "z_minmax_y+"
    compute_kinematic_chain: bool = True
    attach_legs_to_body: bool = True
    verts_detached: bool = True
    temperature: float = 1.0

    def __call__(self, verts):
        if self.verts_detached:
            verts = verts.detach()

        outputs = estimate_bones(
            verts,
            n_body_bones=self.num_body_bones,
            resample=self.resample,
            n_legs=self.num_legs,
            n_leg_bones=self.num_leg_bones,
            body_bones_type=self.body_bones_type,
            compute_kinematic_chain=self.compute_kinematic_chain,
            attach_legs_to_body=self.attach_legs_to_body,
        )
        if self.compute_kinematic_chain:
            bones_pred, kinematic_chain, aux = outputs
        else:
            bones_pred = outputs

        skinnig_weights = estimate_skinning_weigths(
            bones_pred, verts, temperature=self.temperature
        )

        estimator_outputs = {
            "bones_pred": bones_pred,
            "skinnig_weights": skinnig_weights,
        }

        if self.compute_kinematic_chain:
            estimator_outputs["kinematic_chain"] = kinematic_chain
            estimator_outputs["aux"] = aux

        return estimator_outputs
