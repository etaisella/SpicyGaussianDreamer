import os
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import torch

from gaussiansplatting.scene.gaussian_model import GaussianModel
from mesh import Mesh
from mesh_utils import decimate_mesh, clean_mesh
import kiui

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

@torch.no_grad()
def extract_fields(gm, resolution=128, num_blocks=16, relax_ratio=1.5):
    # resolution: resolution of field
    
    block_size = 2 / num_blocks

    assert resolution % block_size == 0
    split_size = resolution // num_blocks

    opacities = gm.get_opacity

    # pre-filter low opacity gaussians to save computation
    mask = (opacities > 0.005).squeeze(1)

    opacities = opacities[mask]
    xyzs = gm.get_xyz[mask]
    stds = gm.get_scaling[mask]
    
    # normalize to ~ [-1, 1]
    mn, mx = xyzs.amin(0), xyzs.amax(0)
    gm.center = (mn + mx) / 2
    gm.scale = 1.8 / (mx - mn).amax().item()

    xyzs = (xyzs - gm.center) * gm.scale
    stds = stds * gm.scale

    covs = gm.covariance_activation(stds, 1, gm._rotation[mask])

    # tile
    device = opacities.device
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    X = torch.linspace(-1, 1, resolution).split(split_size)
    Y = torch.linspace(-1, 1, resolution).split(split_size)
    Z = torch.linspace(-1, 1, resolution).split(split_size)


    # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                # sample points [M, 3]
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                # in-tile gaussians mask
                vmin, vmax = pts.amin(0), pts.amax(0)
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                # if hit no gaussian, continue to next block
                if not mask.any():
                    continue
                mask_xyzs = xyzs[mask] # [L, 3]
                mask_covs = covs[mask] # [L, 6]
                mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                # query per point-gaussian pair.
                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                # batch on gaussian to avoid OOM
                batch_g = 1024
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                    val += (mask_opas[:, start:end] * w).sum(-1)
                
                # kiui.lo(val, mask_opas, w)
            
                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
    
    kiui.lo(occ, verbose=1)

    return occ


def extract_mesh(gm, density_thresh=1, resolution=128, decimate_target=1e5):

    occ = extract_fields(gm, resolution).detach().cpu().numpy()

    import mcubes
    vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
    vertices = vertices / (resolution - 1.0) * 2 - 1

    # transform back to the original space
    vertices = vertices / gm.scale + gm.center.detach().cpu().numpy()

    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

    print(
        f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
    )
    vv = v[:,1]
    v[:,1] = -vv
    v = v * 0.6
    mesh = Mesh(v=v, f=f, device='cuda')

    return mesh


ply_path = "outputs/last_3dgs.ply"
gm = GaussianModel(0)
gm.load_ply(ply_path)
mesh = extract_mesh(gm)
mesh.write("outputs/pc/last_3dgs_4.ply")