import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
# Declare a mean squared error loss, MSE between two images or tensors, this is our primary loss function for training NeRF.
img2mse = lambda x, y : torch.mean((x - y) ** 2)

# Peak Signal-to-Noise Ratio
# Convert MSE to PSNR. Ensure constants live on the same device as inputs to avoid CPU/GPU mismatches.
def mse2psnr(x: torch.Tensor) -> torch.Tensor:
	log10_const = torch.tensor(10.0, device=x.device, dtype=x.dtype)
	return -10.0 * torch.log(x) / torch.log(log10_const)

# Conversion to 8 bit format
# This converts floating poitn image data in the (0-1) range to 8bit integer format 0-255 for saving displaying images
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)


# We need functions to generate one camera ray per pixel for an image, with the following contents
# 1. The ray's origin, where it starts,
# 2. and it's direction (where it points)
# We use these rays to sample points in 3D space and render the colors/densiy along them

def get_rays(H, W, K, c2w):
    """
    Generate one camera ray per pixel for a pinhole camera.

    Given image size H×W, intrinsics K and camera-to-world transform c2w, this
    computes per-pixel world-space ray origins and directions.

    Args:
        H (int): Image height (number of pixels).
        W (int): Image width (number of pixels).
        K (torch.Tensor | numpy.ndarray): 3×3 camera intrinsics matrix following
            the pinhole model, with focal lengths and principal point:
            fx = K[0, 0], fy = K[1, 1], cx = K[0, 2], cy = K[1, 2]. May be a
            `torch.Tensor` or `numpy.ndarray`.
        c2w (torch.Tensor): 4×4 camera-to-world transform. The upper-left 3×3 is
            rotation R and the last column's first three elements are translation
            t. It converts camera-frame coordinates into world-frame coordinates.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - rays_o: World-space ray origins of shape (H, W, 3). All entries
              are the camera center t broadcast to the image grid.
            - rays_d: World-space ray directions of shape (H, W, 3). Computed by
              unprojecting pixel indices with K into camera frame and rotating
              by R into world frame. Not normalized.

    Conventions and assumptions:
        - Pixel indices use integer coordinates (no 0.5 offset). If you want
          true pixel-center rays, add 0.5 to i and j before unprojection.
        - Camera looks down its negative z-axis in camera frame, hence the -1 in
          the z component before rotation.
        - This implementation expects `c2w` as a torch tensor. If K is provided
          as numpy, only its scalar values are used in torch computations.
    """
    # Video Reference i used to learn this: https://www.youtube.com/watch?v=Hz8kz5aeQ44

    # Create a grid of pixel indices on the same device/dtype as c2w to avoid device mismatches
    device = c2w.device
    dtype = c2w.dtype
    #   i indexes the x (column) coordinate in [0, W-1]
    #   j indexes the y (row) coordinate in [0, H-1]
    # Shapes returned by meshgrid are (W, H) with the call order below, hence
    # the transpose to obtain (H, W) arrays matching image layout.
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    )

    # Transpose so that i, j have shape (H, W) instead of (W, H)
    i = i.t()
    j = j.t()
    # could this have been fixed by just changing the argument order tho???

    # Compute camera-frame ray directions for each pixel using intrinsics K.
    # x direction: (i - cx) / fx
    # y direction: -(j - cy) / fy  (negative because image y grows downward)
    # z direction: -1              (camera looks along -z)
    dirs = torch.stack([
        (i - K[0][2]) / K[0][0],
        -(j - K[1][2]) / K[1][1],
        -torch.ones_like(i, device=device, dtype=dtype)
    ], dim=-1)

    # Rotate camera-frame directions to world frame using R = c2w[:3, :3].
    # Broadcasting performs a batched matrix-vector product over the (H, W) grid.
    # Result has shape (H, W, 3).
    # This calculates the world space direction per pixel (not normalized)
    # np.newaxis inserts a new axis of size 1 into tensor array shape without copying data.
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim=-1)

    # Ray origins are the camera center t = c2w[:3, -1], broadcast to (H, W, 3).
    # Each ray starts at the same origin t.
    # Expand does not allocate memory, so every pixel shares same origin?
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

# This is the numpy version of the get_rays function.
# It is used to generate rays for images that are not in torch format.
# This is a cpu utility
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Convert rays from world (or camera) space to Normalized Device Coordinates (NDC)
    for a forward-facing pinhole camera setup.

    Why NDC?
    - NDC maps the viewing frustum to a canonical cube (typically x,y in [-1, 1], z in [0, 1]).
    - This transformation improves numerical stability and allows consistent ray stepping
      for forward-facing datasets (e.g., LLFF), as used in the original NeRF code.

    Args:
        H (int): Image height in pixels.
        W (int): Image width in pixels.
        focal (float): Focal length in pixels (assumed fx = fy = focal).
        near (float): Near plane distance where rays first enter the view frustum.
        rays_o (torch.Tensor): Ray origins, shape (..., 3).
        rays_d (torch.Tensor): Ray directions, shape (..., 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - rays_o_ndc: Ray origins in NDC, shape (..., 3).
            - rays_d_ndc: Ray directions in NDC, shape (..., 3).

    Notes:
        - This follows the OpenGL-style NDC used by the LLFF dataset in NeRF.
        - Assumes rays have negative z in camera space (forward-facing).
        - Requires rays_d[..., 2] != 0 to avoid division by zero.
    """
    # 1) Shift ray origins to the near plane so that every ray "starts" at z = near.
    # Solve for t such that (rays_o + t * rays_d).z = -near in camera coordinates.
    # Here, sign conventions yield the following closed form used in the original NeRF code:
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # 2) Project shifted origins and directions into NDC.
    # The formulas below map x and y by dividing by z and scaling by focal and image size,
    # while z is remapped into [0, 1] based on the near plane.

    # Origin in NDC (element-wise operations preserve leading batch dims):
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    # Direction in NDC. This is derived by differentiating the projected origin wrt t
    # and evaluating at the shifted origin above. Intuitively: project a nearby point
    # along the ray and subtract the projected origin, then consider unit step.
    d0 = -1. / (W / (2. * focal)) * (
        rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]
    )
    d1 = -1. / (H / (2. * focal)) * (
        rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]
    )
    d2 = -2. * near / rays_o[..., 2]

    # 3) Pack back into (..., 3) tensors for origins and directions in NDC space.
    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)

    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Importance-sample additional points along rays using inverse transform sampling
    on a piecewise-constant PDF defined over `bins`.

    Where this fits in NeRF:
    - Coarse model samples uniformly along each ray and produces per-sample weights
      (how much each interval contributes). We treat these weights as an (unnormalized)
      PDF and resample more densely where weights are high (hierarchical sampling).

    Args:
        bins (torch.Tensor): Bin edges along each ray, shape [..., M]. Typically these are
            the midpoints or edges of intervals from the coarse pass.
        weights (torch.Tensor): Per-bin nonnegative weights, shape [..., M-1] or [..., M]
            depending on convention. They are normalized internally to form a PDF.
        N_samples (int): Number of new samples to draw from the PDF per ray.
        det (bool): If True, use deterministic uniform samples in [0,1] (stratified-like).
        pytest (bool): If True, use numpy-based deterministic random numbers for testing.

    Returns:
        torch.Tensor: Sampled positions with shape [..., N_samples], lying within the support
        of `bins`, with higher density in regions of larger `weights`.
    """
    # 1) Build a (numerically stable) PDF and its CDF from the provided weights.
    weights = weights + 1e-5  # avoid nans by preventing zero-division
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    # Prepend 0 so the CDF starts at 0 and ends near 1. Shape becomes [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # 2) Draw uniform samples u in [0, 1] to invert through the CDF.
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device, dtype=cdf.dtype)

    # Deterministic path for testing.
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.tensor(u, dtype=cdf.dtype, device=cdf.device)

    # 3) Invert the CDF. For each uniform sample u, find the CDF interval it falls into,
    #    then linearly interpolate within the corresponding bin.
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim=-1)  # [..., N_samples, 2]

    # Gather the CDF values and bin positions for the bracketing indices.
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Normalize u within the located CDF segment and interpolate within the bin.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
