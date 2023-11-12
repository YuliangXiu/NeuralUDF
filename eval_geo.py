import numpy as np
import os
import torch
import trimesh
from typing import Tuple
from tqdm import tqdm

from pytorch3d import _C
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.structures import Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d.structures import Meshes

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """
    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None



def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    w = torch.cat([w0[..., None], w1[..., None], w2[..., None]], dim=2)

    return w



def sample_points_from_meshes(meshes, num_samples: int = 10000):
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)    # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)    # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)    # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        samples_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )    # (N, num_samples)
        samples_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Randomly generate barycentric coords.
    # w                 (N, num_samples, 3)
    # sample_face_idxs  (N, num_samples)
    # samples_verts     (N, num_samples, 3, 3)

    samples_bw = _rand_barycentric_coords(num_valid_meshes, num_samples, verts.dtype, verts.device)
    sample_verts = verts[faces][samples_face_idxs]
    samples[meshes.valid] = (sample_verts * samples_bw[..., None]).sum(dim=-2)

    return samples, samples_face_idxs, samples_bw



def point_mesh_distance(meshes, pcls, weighted=True):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()    # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]    # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face, idxs = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    if weighted:
        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls.packed_to_cloud_idx()    # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()    # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = torch.sqrt(point_to_face) * weights_p

    return point_to_face, idxs


def calculate_p2s(tgt_mesh, src_mesh):

    tgt_points = Pointclouds(tgt_mesh.verts_packed().unsqueeze(0))
    p2s_dist1 = point_mesh_distance(src_mesh, tgt_points)[0].sum() * 100.0
    
    samples_src, _, _ = sample_points_from_meshes(src_mesh, 100000)
    src_points = Pointclouds(samples_src)
    p2s_dist2 = point_mesh_distance(tgt_mesh, src_points)[0].sum() * 100.0

    p2s_dist = 0.5 * (p2s_dist1 + p2s_dist2
                      )
    return p2s_dist


# method = "NeuralUDF"
# method = "NeUDF"
method = "NeAT"
# method = "NeuS"
# method = "DMTet"
# method = "GShell"
# method = "other"
# method = "specular"

root_dir = f"/home/yxiu/Code/{method}"
# subjects  = ["30", "92", "117", "133", "164", "320", "448", "522", "591"]
subjects  = [0,2,4,6,8,10]

chamfer_lst_mc = []
chamfer_lst_udf = []

device = torch.device("cuda:0")

out_path = f"metric/{method}-specular.npy"

if not os.path.exists(out_path):
    pbar = tqdm(subjects)
    
    for subject in pbar:
        
        # gt_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/gt/{subject}/mesh.obj"
        gt_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/gt/30/mesh.obj"
        gt_mesh_trimesh = trimesh.load(gt_mesh_path)
        gt_mesh_ = Meshes(
                verts=[torch.tensor(gt_mesh_trimesh.vertices).float(),], 
                faces=[torch.tensor(gt_mesh_trimesh.faces).long(),]).to(device)
        
        if method == "NeuralUDF":
            mc_mesh_path = os.path.join(root_dir, 
                                f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_specular_mask/meshes/00400000_thresh0.0050_res512.ply")
            udf_mesh_path = os.path.join(root_dir, 
                                f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_specular_mask/udf_meshes/udf_res512_step400000.ply")
            
            # mc_mesh_path = os.path.join(root_dir, 
            #                     f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_mask/meshes/00400000_thresh0.0050_res512.ply")
            # udf_mesh_path = os.path.join(root_dir, 
            #                     f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_mask/udf_meshes/udf_res512_step400000.ply")
        elif method == "NeUDF":
            mc_mesh_path = os.path.join(root_dir, 
                                f"exp/{subject}/wmask_specular_open/meshes/00400000.ply")
            udf_mesh_path = os.path.join(root_dir, 
                                f"exp/{subject}/wmask_specular_open/meshes/mu00400000.ply")
        elif method == "NeAT":
            mc_mesh_path = os.path.join(root_dir, 
                                f"exp/{subject}/wmask_specular/meshes/00400000_sdf.ply")
            udf_mesh_path = os.path.join(root_dir, 
                                f"exp/{subject}/wmask_specular/meshes/00400000.ply")
        elif method == "NeuS":
            # mc_mesh_path = os.path.join(root_dir, 
            #                     f"exp/{subject}/garment_wmask/meshes/00400000.ply")
            mc_mesh_path = os.path.join(root_dir, 
                                f"exp/{subject}/specular_wmask/meshes/00400000.ply")
            udf_mesh_path = None
            
        elif method == "DMTet":
            udf_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/deepfashion_dmtet_batch/{subject}/mesh.obj"
            mc_mesh_path = None

        elif method == "GShell":
            udf_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/deepfashion_deeper_batch/{subject}/mesh.obj"
            mc_mesh_path = None  
            
        elif method == "specular":
            udf_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/deepfashion_specular/{subject}/mesh.obj"
            mc_mesh_path = None  
            
        elif method == "other":
            udf_mesh_path = f"/home/yxiu/Code/NeuralUDF/data/deepfashion_mesh/tmp.obj"
            mc_mesh_path = None  
        
        if mc_mesh_path is not None:
            mc_mesh_trimesh = trimesh.load(mc_mesh_path)
            mc_mesh_ = Meshes(
                verts=[torch.tensor(mc_mesh_trimesh.vertices).float(),], 
                faces=[torch.tensor(mc_mesh_trimesh.faces).long(),]).to(device)
            
            mc_p2s = calculate_p2s(tgt_mesh=gt_mesh_, src_mesh=mc_mesh_)
            pbar.set_description(f"{method}-{subject}-MC: {mc_p2s:.3f}")
            chamfer_lst_mc.append(mc_p2s.item())
            
        if udf_mesh_path is not None:
            udf_mesh_trimesh = trimesh.load(udf_mesh_path)
            udf_mesh_ = Meshes(
                verts=[torch.tensor(udf_mesh_trimesh.vertices).float(),], 
                faces=[torch.tensor(udf_mesh_trimesh.faces).long(),]).to(device)
            
            udf_p2s = calculate_p2s(tgt_mesh=gt_mesh_, src_mesh=udf_mesh_)
            pbar.set_description(f"{method}-{subject}-UDF: {udf_p2s:.3f}")
            chamfer_lst_udf.append(udf_p2s.item())
        

    np.save(out_path, {"mc": chamfer_lst_mc, "udf": chamfer_lst_udf}, allow_pickle=True)
else:
    chamfer_lst_mc = np.load(out_path, allow_pickle=True).item()["mc"]
    chamfer_lst_udf = np.load(out_path, allow_pickle=True).item()["udf"]
    
mc_avg = torch.mean(torch.tensor(chamfer_lst_mc)).item()
udf_avg = torch.mean(torch.tensor(chamfer_lst_udf)).item()
chamfer_lst_mc.append(mc_avg)
chamfer_lst_udf.append(udf_avg)

chamfer_mc_str = "MC: "
for data in chamfer_lst_mc:
    chamfer_mc_str+= f"{data:.3f} & " 
    
chamfer_udf_str = "UDF: "
for data in chamfer_lst_udf:
    chamfer_udf_str+= f"{data:.2f} & " 

print(f"{method}\n {chamfer_mc_str} \n {chamfer_udf_str}")
    