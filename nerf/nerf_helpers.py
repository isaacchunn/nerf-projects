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

# SSIM and LPIPS metrics for image quality evaluation
def calculate_ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_map=False):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    This function matches the implementation from other NeRF repositories for consistency.
    Based on tf.image.ssim implementation.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image with shape [H, W, 3]
        img2 (torch.Tensor or np.ndarray): Second image with shape [H, W, 3]  
        max_val (float): The maximum magnitude that img1 or img2 can have.
        filter_size (int): Window size.
        filter_sigma (float): The bandwidth of the Gaussian used for filtering.
        k1 (float): One of the SSIM dampening parameters.
        k2 (float): One of the SSIM dampening parameters.
        return_map (bool): If True, will return the per-pixel SSIM map.
        
    Returns:
        float or torch.Tensor: Mean SSIM value, or SSIM map if return_map=True
    """
    import torch.nn.functional as F
    
    # Convert to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Ensure tensors are on the same device
    device = img1.device if torch.is_tensor(img1) else img2.device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Ensure images are in [0, max_val] range
    img1 = torch.clamp(img1, 0, max_val)
    img2 = torch.clamp(img2, 0, max_val)
    
    # Reshape to [..., width, height, num_channels] format expected by the function
    ori_shape = img1.size()
    if len(ori_shape) == 3:  # [H, W, C]
        width, height, num_channels = ori_shape
        img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)  # [B, C, H, W]
        img2 = img2.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
        batch_size = img1.shape[0]
    else:
        raise ValueError(f"Expected 3D tensor [H, W, C], got {ori_shape}")

    # Construct a 1D Gaussian blur filter
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution)
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Compose the blurs
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img1)
    mu1 = filt_fn(img2)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img1 ** 2) - mu00
    sigma11 = filt_fn(img2 ** 2) - mu11
    sigma01 = filt_fn(img1 * img2) - mu01

    # Clip the variances and covariances to valid values
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    
    if return_map:
        return ssim_map
    else:
        return ssim.item() if ssim.numel() == 1 else ssim.mean().item()

def calculate_lpips(img1, img2, net='vgg', device='cuda', normalize=True):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    
    This function matches the implementation from other NeRF repositories for consistency.
    Uses VGG backbone by default as in the reference implementation.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image with shape [H, W, 3]
        img2 (torch.Tensor or np.ndarray): Second image with shape [H, W, 3]
        net (str): Network to use ('alex', 'vgg', 'squeeze'). Default: 'vgg' for consistency
        device (str): Device to run computation on. Default: 'cuda'
        normalize (bool): Whether to normalize inputs. Default: True
        
    Returns:
        float: LPIPS distance (lower is better, 0 = identical)
    """
    import lpips
    
    # Convert numpy to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Ensure tensors are on the correct device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Convert [H, W, 3] to [3, H, W] format expected by LPIPS
    img1 = img1.permute(2, 0, 1).contiguous()  # [H, W, 3] -> [3, H, W]
    img2 = img2.permute(2, 0, 1).contiguous()
    
    # Initialize LPIPS model with VGG backbone (cached after first call)
    if not hasattr(calculate_lpips, 'loss_fn') or calculate_lpips.net != net:
        calculate_lpips.loss_fn = lpips.LPIPS(net=net).eval().to(device)
        calculate_lpips.net = net
    
    # Calculate LPIPS distance
    with torch.no_grad():
        distance = calculate_lpips.loss_fn(img1, img2, normalize=normalize)
    
    return distance.item()

def calculate_metrics(img1, img2, include_lpips=True, lpips_net='vgg', device='cuda'):
    """
    Calculate comprehensive image quality metrics between two images.
    
    This follows the standard evaluation methodology used in NeRF papers:
    - PSNR: Peak Signal-to-Noise Ratio (higher is better)
    - SSIM: Structural Similarity Index (0-1, higher is better) 
    - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image with shape [H, W, 3]
        img2 (torch.Tensor or np.ndarray): Second image with shape [H, W, 3]
        include_lpips (bool): Whether to calculate LPIPS (slower). Default: True
        lpips_net (str): Network to use for LPIPS ('alex', 'vgg', 'squeeze'). Default: 'vgg'
        device (str): Device for LPIPS computation. Default: 'cuda'
        
    Returns:
        dict: Dictionary containing 'mse', 'psnr', 'ssim', and optionally 'lpips'
        
    Note: 
        - Images should be in [0,1] range (this function will clip to ensure this)
        - SSIM uses data_range=1.0 for [0,1] images
        - LPIPS uses VGG backbone by default (consistent with reference implementations)
    """
    # Ensure images are in [0,1] range and proper format
    if isinstance(img1, np.ndarray):
        img1_tensor = torch.from_numpy(np.clip(img1, 0, 1)).float()
    else:
        img1_tensor = torch.clamp(img1.float(), 0, 1)
    
    if isinstance(img2, np.ndarray):
        img2_tensor = torch.from_numpy(np.clip(img2, 0, 1)).float()
    else:
        img2_tensor = torch.clamp(img2.float(), 0, 1)
    
    # Calculate MSE and PSNR (standard NeRF evaluation)
    mse = img2mse(img1_tensor, img2_tensor).item()
    psnr = mse2psnr(torch.tensor(mse)).item()
    
    # Calculate SSIM
    ssim_val = calculate_ssim(img1, img2)
    
    metrics = {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_val
    }
    
    # Calculate LPIPS if requested
    if include_lpips:
        try:
            lpips_val = calculate_lpips(img1, img2, net=lpips_net, device=device)
            metrics['lpips'] = lpips_val
        except Exception as e:
            print(f"Warning: Could not calculate LPIPS: {e}")
            metrics['lpips'] = None
    
    return metrics


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
