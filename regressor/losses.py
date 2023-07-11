import torch


def rotation_loss(rotation_pred, rotation_gt):
    """
    rotation_pred and rotation_gt are quaternions, shape: (B, 4)
    """
    return (1 - torch.abs(torch.sum(rotation_pred * rotation_gt, dim=1))).mean()
