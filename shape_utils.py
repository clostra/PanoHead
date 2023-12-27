# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Utils for extracting 3D shapes using marching cubes. Based on code from DeepSDF (Park et al.)

Takes as input an .mrc file and extracts a mesh.

Ex.
    python shape_utils.py my_shape.mrc
Ex.
    python shape_utils.py myshapes_directory --level=12
"""


import time
import plyfile
import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import trimesh
import skimage.measure
import argparse
import mrcfile
import cv2
from tqdm import tqdm
        
def export_textured_mesh(v_np, f_np, path, h0=2048, w0=2048, ssaa=1, name='', color_lambda=None):
    assert color_lambda is not None, "Cannot generate texture. color_lambda must be provided"
    # v, f: torch Tensor
    device = 'cuda'
    # v_np = v.cpu().numpy() # [N, 3]
    # f_np = f.cpu().numpy() # [M, 3]
    v = torch.from_numpy(v_np.copy()).float().to(device)
    f = torch.from_numpy(f_np.copy()).int().to(device)

    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

    # unwrap uvs
    import xatlas
    import nvdiffrast.torch as dr
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4 # for faster unwrap...
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

    # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

    # render uv maps
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    if ssaa > 1:
        h = int(h0 * ssaa)
        w = int(w0 * ssaa)
    else:
        h, w = h0, w0
    
    if h <= 2048 and w <= 2048:
        glctx = dr.RasterizeCudaContext()
    else:
        glctx = dr.RasterizeGLContext()

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

    # masked query 
    xyzs = xyzs.view(-1, 3)
    mask = (mask > 0).view(-1)
    
    feats = torch.zeros(h0 * w0, 3, device=device, dtype=torch.float32)

    if mask.any():
        xyzs = xyzs[mask] # [M, 3]

        # batched inference to avoid OOM
        all_feats = []
        head = 0
        while head < xyzs.shape[0]:
            tail = min(head + 640000, xyzs.shape[0])
            cur_feats = color_lambda(xyzs[head:tail])
            all_feats.append(cur_feats.float())
            head += 640000
        all_feats = torch.cat(all_feats, dim=0)
        feats[mask] = all_feats
    
    feats = feats.view(h, w, -1)
    mask = mask.view(h, w)

    # quantize [0.0, 1.0] to [0, 255]
    feats = feats.cpu().numpy()

    # DEBUG
    # feats = (feats - feats.min()) / (feats.max() - feats.min())
    feats = (feats * 255).astype(np.uint8)

    ### NN search as an antialiasing ...
    mask = mask.cpu().numpy()

    inpaint_region = binary_dilation(mask, iterations=3)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=2)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

    feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

    # do ssaa after the NN search, in numpy
    if ssaa > 1:
        feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

    # save obj (v, vt, f /)
    obj_file = os.path.join(path, f'{name}mesh.obj')
    mtl_file = os.path.join(path, f'{name}mesh.mtl')

    print(f'[INFO] writing obj mesh to {obj_file}')
    with open(obj_file, "w") as fp:
        fp.write(f'mtllib {name}mesh.mtl \n')
        
        print(f'[INFO] writing vertices {v_np.shape}')
        for v in v_np:
            fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
    
        print(f'[INFO] writing vertices texture coords {vt_np.shape}')
        for v in vt_np:
            fp.write(f'vt {v[0]} {1 - v[1]} \n') 

        print(f'[INFO] writing faces {f_np.shape}')
        fp.write(f'usemtl mat0 \n')
        for i in range(len(f_np)):
            fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

    with open(mtl_file, "w") as fp:
        fp.write(f'newmtl mat0 \n')
        fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
        fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
        fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
        fp.write(f'Tr 1.000000 \n')
        fp.write(f'illum 1 \n')
        fp.write(f'Ns 0.000000 \n')
        fp.write(f'map_Kd {name}albedo.png \n')
def convert_sdf_samples_to_obj(
    numpy_3d_sdf_tensor,
    path,
    color_lambda=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    resolution = numpy_3d_sdf_tensor.shape[0]

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[1, 1, 1]
    )
    faces = faces[:, [1, 0, 2]]  # Pytorch3d uses the reverse ordering of vertices from skimage
    verts = verts / (resolution - 1) - 0.5
    os.makedirs(path, exist_ok=True)
    export_textured_mesh(verts, faces, path, color_lambda=color_lambda)

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")


def convert_mrc(input_filename, output_filename, isosurface_level=1):
    with mrcfile.open(input_filename) as mrc:
        convert_sdf_samples_to_ply(np.transpose(mrc.data, (2, 1, 0)), [0, 0, 0], 1, output_filename, level=isosurface_level)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mrc_path')
    parser.add_argument('--level', type=float, default=10, help="The isosurface level for marching cubes")
    args = parser.parse_args()

    if os.path.isfile(args.input_mrc_path) and args.input_mrc_path.split('.')[-1] == 'ply':
        output_obj_path = args.input_mrc_path.split('.mrc')[0] + '.ply'
        convert_mrc(args.input_mrc_path, output_obj_path, isosurface_level=1)

        print(f"{time.time() - start_time:02f} s")
    else:
        assert os.path.isdir(args.input_mrc_path)

        for mrc_path in tqdm(glob.glob(os.path.join(args.input_mrc_path, '*.mrc'))):
            output_obj_path = mrc_path.split('.mrc')[0] + '.ply'
            convert_mrc(mrc_path, output_obj_path, isosurface_level=args.level)