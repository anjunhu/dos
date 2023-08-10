import nvdiffrast.torch as dr
import torch

from ..nvdiffrec.render import renderutils


def get_visible_vertices(mesh, mvp, resolution):
    """
    Returns a tensor of shape (batch_size, num_vertices) with 1 for visible vertices and 0 for occluded vertices
    """
    # Render occlusion
    glctx = dr.RasterizeGLContext()
    v_pos_clip4 = renderutils.xfm_points(mesh.v_pos, mvp)
    rast, _ = dr.rasterize(glctx, v_pos_clip4, mesh.t_pos_idx[0].int(), resolution)
    face_ids = rast[..., -1]
    face_ids = face_ids.view(-1, resolution[0] * resolution[1])
    current_batch, num_verts, _ = mesh.v_pos.shape
    res = []

    # TODO: vectorize this?
    for b in range(current_batch):
        current_face_ids = face_ids[b]
        current_face_ids = current_face_ids[current_face_ids > 0]
        visible_verts = mesh.t_pos_idx[0][(current_face_ids - 1).long()].view(-1)
        visible_verts = torch.unique(visible_verts)
        visibility = torch.zeros(num_verts, device=visible_verts.device)
        visibility[visible_verts] = 1

        res += [visibility]
    res = torch.stack(res, dim=0)

    return res
