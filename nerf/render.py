"""
NeRF rendering utilities

This module contains helper functions to render volumes using Neural Radiance Fields (NeRF).
If you're new to NeRF, here's the high-level idea you'll see reflected in the code:

- A NeRF model takes a 3D point (and often a viewing direction) and predicts color (RGB)
  and density (sigma) at that point.
- To render an image, we cast a ray through each pixel, sample many points along the ray,
  evaluate the network at those points, and composite the colors using volume rendering.
- We optionally do this in two passes: a coarse pass to find where the scene is, then a
  fine pass that samples more densely in the important regions (importance sampling).

The functions below help with:
- Splitting big tensors/ray sets into smaller chunks to avoid out-of-memory issues.
- Converting raw network outputs into rendered RGB/depth/opacity via volume rendering.
- Orchestrating the full per-ray rendering with optional hierarchical sampling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import imageio
import os
from tqdm import tqdm

from nerf_helpers import *

DEBUG = False

"""
The following function wraps a function fn so it processes a large input tensor in smaller chunks
to reduce peak memory usage, then stitches the result together.

This splits inputs along the first dimension into slices of size chunk and applies fn to each slice.
Then it concatenates the results back together along the first dimension.

This reduces gpu and cpu spikes when fn is heavy (i.e neural network forward pass) and preserves
autograd, as gradients flow through torch.cat and slicing.
"""
def batchify(fn, chunk):
    """Wrap a function so it runs on smaller input chunks to reduce peak memory.

    This is useful for heavy functions like neural network forward passes. Instead of
    evaluating the entire input tensor in one go, we split it along the first dimension
    into slices of size ``chunk`` and then concatenate the results.

    Args:
        fn (Callable[[Tensor], Tensor]): The function to run on chunks.
        chunk (Optional[int]): Number of rows to process per call. If ``None``,
            no chunking is performed.

    Returns:
        Callable[[Tensor], Tensor]: A wrapper that applies ``fn`` on input chunks and
        stitches the outputs along the first dimension.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

"""
Render a large set of rays by splitting them into manageable minibatches to avoid
out-of-memory (OOM) issues, then stitch the per-batch results back together.

High-level intuition:
- Rendering involves evaluating many rays (often millions). Each ray requires sampling
  points, running an MLP, and compositing colors/densities, which can be memory-heavy.
- Instead of rendering all rays at once, we process them in chunks (minibatches), which
  keeps peak memory usage bounded.
- We collect and concatenate the results for each output quantity (e.g., rgb, depth, acc).
"""

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """
    Render rays in smaller minibatches to avoid OOM.

    Args:
        rays_flat (torch.Tensor): Rays flattened along the batch dimension, shape [N, Cray].
            Each row encodes a single ray's data (e.g., origin, direction, near/far, etc.).
        chunk (int): Number of rays to render per minibatch.
        **kwargs: Additional keyword arguments forwarded to `render_rays` (e.g., models,
            sampling counts, noise parameters).

    Returns:
        Dict[str, torch.Tensor]: A dictionary where each key corresponds to a rendered
        quantity (e.g., 'rgb_map', 'disp_map', 'acc_map', etc.), and each tensor has
        shape [N, ...], formed by concatenating per-chunk outputs along the first dimension.
    """
    # Accumulate lists of per-chunk outputs in a dictionary keyed by output name.
    all_ret = {}

    # Iterate over rays in chunks: [0:chunk], [chunk:2*chunk], ...
    for i in range(0, rays_flat.shape[0], chunk):
        # Render a minibatch of rays using the provided rendering function and settings.
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)

        # For each output field produced by the renderer, append the minibatch result
        # to a growing list so we can concatenate later.
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # Concatenate lists of chunk results into full [N, ...] tensors for each field.
    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}

    return all_ret

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    Convert raw network predictions along rays into rendered outputs using
    volumetric rendering (accumulation over samples).

    High-level intuition (NeRF volume rendering):
    - The network predicts, per sampled point along a ray, an RGB value and a density
      (often called sigma). Density indicates how much light is absorbed/emitted there.
    - We turn densities into per-sample opacities (alpha) based on the distance between
      adjacent samples along the ray.
    - We compute transmittance (how much light makes it to a sample without being blocked)
      via a cumulative product, then form per-sample weights = transmittance * alpha.
    - We composite colors, depths, and other quantities by weighted sums over samples.

    Args:
        raw (torch.Tensor): [N_rays, N_samples, 4] raw predictions per sample. The first
            3 channels are RGB logits (before sigmoid), and the last channel is density (sigma).
        z_vals (torch.Tensor): [N_rays, N_samples] sample depths or t-values along each ray.
        rays_d (torch.Tensor): [N_rays, 3] direction vectors for each ray. Used to scale
            step sizes from parametric units to metric distances.
        raw_noise_std (float): Stddev of Gaussian noise added to sigma during training for
            regularization. Set to 0.0 at eval time.
        white_bkgd (bool): If True, composite the result over a white background (useful
            for synthetic datasets rendered on white).
        pytest (bool): If True, use deterministic numpy noise for reproducible tests.

    Returns:
        tuple:
            - rgb_map (torch.Tensor): [N_rays, 3] rendered RGB color per ray.
            - disp_map (torch.Tensor): [N_rays] disparity (inverse depth) per ray.
            - acc_map (torch.Tensor): [N_rays] accumulated opacity per ray (sum of weights).
            - weights (torch.Tensor): [N_rays, N_samples] per-sample contribution weights.
            - depth_map (torch.Tensor): [N_rays] expected depth per ray.
    """
    # Map density (sigma) and step size (distance between samples) to opacity (alpha):
    #   alpha = 1 - exp(-relu(sigma) * delta)
    # relu ensures sigma is non-negative, as negative density is not physical.
    raw2alpha = lambda raw_sigma, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw_sigma) * dists)

    # Compute distances between adjacent samples along each ray in z (or t) space.
    # Shape after diff: [N_rays, N_samples-1]
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # For the last sample on each ray, append a very large distance so that its
    # contribution is properly modeled as the ray exiting the volume.
    # Resulting shape: [N_rays, N_samples]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], dim=-1)

    # Convert parametric distances to metric distances by multiplying by the ray length.
    # This accounts for non-unit ray directions. rays_d[..., None, :] has shape [N_rays, 1, 3]
    # and we take its L2 norm to scale each sample distance.
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Convert raw RGB logits to [0,1] colors per sample.
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    # Optional noise added to densities during training for regularization.
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device, dtype=raw.dtype) * raw_noise_std

        # Deterministic noise path for unit tests.
        if pytest:
            np.random.seed(0)
            noise_np = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.tensor(noise_np, device=raw.device, dtype=raw.dtype)

    # Opacity per sample from density and distance.
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # Compute per-sample weights using transmittance. The transmittance T_k is the product of
    # (1 - alpha) of all previous samples. We build it via a cumulative product.
    # Prepend a column of ones to represent T_0 = 1, then drop the last column to make it exclusive.
    transmittance_prefix = torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device, dtype=alpha.dtype),
                                      1.0 - alpha + 1e-10], dim=-1)
    transmittance = torch.cumprod(transmittance_prefix, dim=-1)[:, :-1]
    weights = alpha * transmittance  # [N_rays, N_samples]

    # Rendered color is the weighted sum of per-sample colors along the ray.
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]

    # Expected depth is the weighted sum of sample depths.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity is inverse depth. We divide expected depth by total weight (visibility)
    # and guard with epsilon to avoid divide-by-zero when the ray hits nothing.
    denom = torch.max(1e-10 * torch.ones_like(depth_map), torch.sum(weights, dim=-1))
    disp_map = 1.0 / torch.clamp(depth_map / denom, min=1e-10)

    # Accumulated opacity along the ray (how much of the ray got "stopped").
    acc_map = torch.sum(weights, dim=-1)

    # If the scene assumes a white background, composite the missing transmittance as white.
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Render a bundle of rays using NeRF-style volume rendering.

    High-level steps for each ray:
    1) Sample N points between near and far along the ray (evenly in depth or inverse depth).
    2) Query the network at those 3D points (and optionally view directions) to get raw RGB+sigma.
    3) Convert raw predictions to colors via volume rendering (accumulate with alphas/weights).
    4) If enabled, run hierarchical (importance) sampling: take a second set of samples drawn
       from a PDF defined by the coarse weights, re-evaluate the network (fine), and re-render.

    Args:
        ray_batch (torch.Tensor): [num_rays, Cray]. Per-ray data packed together. The first
            3 entries are ray origins, next 3 are ray directions, next 2 are near/far bounds,
            and the last 3 (if present) are unit view directions for view-dependent effects.
        network_fn (Callable): The coarse NeRF MLP. Given points (and viewdirs), predicts
            raw RGB (logits) and density (sigma).
        network_query_fn (Callable): A helper that formats inputs and calls the network.
        N_samples (int): Number of stratified samples for the coarse pass.
        retraw (bool): If True, also return the raw outputs from the last pass.
        lindisp (bool): If True, sample uniformly in inverse depth (disparity) instead of depth.
            This concentrates samples near the camera, helpful for scenes with large depth ranges.
        perturb (float): If > 0, enable stratified sampling noise during training for anti-aliasing.
        N_importance (int): Extra samples for the fine pass (hierarchical sampling). 0 disables it.
        network_fine (Optional[Callable]): A separate fine MLP. If None, reuse ``network_fn``.
        white_bkgd (bool): If True, composite the result over a white background.
        raw_noise_std (float): Stddev of Gaussian noise added to density during training.
        verbose (bool): If True, print additional debug information.
        pytest (bool): If True, make randomness deterministic for unit tests.

    Returns:
        Dict[str, torch.Tensor]: Always includes:
            - ``rgb_map`` [num_rays, 3]: Rendered color from the last pass (fine if enabled).
            - ``disp_map`` [num_rays]: Disparity (1/depth) from the last pass.
            - ``acc_map`` [num_rays]: Accumulated opacity from the last pass.
        Optionally includes:
            - ``raw`` [num_rays, num_samples, 4]: Raw outputs of the last pass if ``retraw``.
            - ``rgb0``, ``disp0``, ``acc0``: Coarse pass results when ``N_importance > 0``.
            - ``z_std`` [num_rays]: Std. dev. of fine samples (measures sampling concentration).
    """
    N_rays = ray_batch.shape[0]
    # Unpack packed ray data: origins, directions, near/far bounds, and optional viewdirs.
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # Step 1: Choose parametric sample positions t in [0, 1] and map to depths z in [near, far].
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # Uniform samples in depth
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Uniform samples in inverse depth (places more samples closer to the camera)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # During training, jitter samples within each interval for stratified sampling.
        # This reduces aliasing and improves robustness.
        # Get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # Draw stratified samples inside those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # Compute 3D sample locations along each ray: o + t*d for each sampled depth t (z_vals).
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # raw = run_network(pts)
    # Query the (coarse) network at all sample points. ``viewdirs`` enables view-dependent effects.
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Hierarchical sampling (importance sampling):
        # Build a PDF from coarse weights and draw additional samples where the scene is likely.
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # Exclude the first and last weights to avoid boundary artifacts when forming the PDF.
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        # Detach so gradients do not flow through the sampling operation.
        z_samples = z_samples.detach()

        # Merge coarse and fine samples, then sort along the ray.
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        # Use dedicated fine network if provided, else reuse the coarse network.
        run_fn = network_fn if network_fine is None else network_fine
        # raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        # Include coarse pass results for potential losses or visualization.
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        # Standard deviation of fine samples: indicates how concentrated sampling is per ray.
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render a full image or a provided set of rays using NeRF.

    There are two common ways to call this function:
    - Full image: pass a camera-to-world matrix ``c2w`` and camera intrinsics ``K``.
      The function will generate one ray per pixel and render all of them.
    - Custom rays: pass ``rays=(rays_o, rays_d)`` to render only those rays.

    Args:
        H (int): Image height in pixels.
        W (int): Image width in pixels.
        K (torch.Tensor or np.ndarray): 3x3 camera intrinsics matrix. We use ``K[0,0]``
            (focal length in pixels) when converting to NDC for forward-facing scenes.
        chunk (int): Max number of rays to process per minibatch to control memory usage.
        rays (tuple[Tensor, Tensor], optional): Tuple ``(rays_o, rays_d)`` with shapes
            [..., 3] each, giving ray origins and directions. If provided, ``c2w`` is ignored.
        c2w (torch.Tensor or np.ndarray, optional): [3,4] camera-to-world matrix. If provided,
            rays for the full image are generated via ``get_rays``.
        ndc (bool): If True, convert rays to normalized device coordinates (recommended for
            forward-facing scenes as in the original NeRF paper).
        near (float): Near plane distance used to initialize per-ray near bounds.
        far (float): Far plane distance used to initialize per-ray far bounds.
        use_viewdirs (bool): If True, pass unit viewing directions to the network to enable
            view-dependent appearance (specularities).
        c2w_staticcam (torch.Tensor or np.ndarray, optional): If provided with ``use_viewdirs``
            enabled, generate rays from ``c2w_staticcam`` but keep view directions from ``c2w``.
            This is useful to visualize how view-dependent effects change with direction.
        **kwargs: Forwarded to ``render_rays`` (e.g., networks, sample counts, noise settings).

    Returns:
        list: ``[rgb_map, disp_map, acc_map, extras]``
            - rgb_map: [H, W, 3] rendered colors
            - disp_map: [H, W] disparity (1/depth)
            - acc_map: [H, W] accumulated opacity
            - extras: dict with any additional outputs from ``render_rays``
    """
    if c2w is not None:
        # Special case: render a full image by generating one ray per pixel.
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # Use the provided custom ray batch.
        rays_o, rays_d = rays

    if use_viewdirs:
        # Provide normalized ray directions to the network for view-dependent effects.
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # Visualize only the effect of changing view direction while keeping camera fixed.
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # Convert to NDC (assumes a pinhole camera model), commonly used for LLFF/forward-facing scenes.
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # Initialize per-ray near/far bounds and pack rays into a single tensor expected by render_rays.
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render all rays in memory-friendly chunks, then reshape results back to image grids.
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """Render a sequence of camera poses to produce a video or trajectory.

    This convenience function loops over a list/array of camera-to-world matrices and calls
    ``render`` for each pose. Optionally writes frames to ``savedir`` and/or renders at a
    lower resolution for speed.

    Args:
        render_poses (Iterable[Tensor or np.ndarray]): Sequence of [3,4] camera-to-world matrices.
        hwf (tuple): ``(H, W, focal)`` from dataset metadata. Only ``H`` and ``W`` are used here.
        K (Tensor or np.ndarray): 3x3 intrinsics matrix passed through to ``render``.
        chunk (int): Chunk size forwarded to ``render``.
        render_kwargs (dict): Keyword args forwarded to ``render`` (e.g., networks and settings).
        gt_imgs (optional): Ground-truth images; if provided, you can compute metrics (example below).
        savedir (str, optional): If provided, write each rendered RGB frame as a PNG to this folder.
        render_factor (int): If > 0, downsample H and W by this factor to render faster.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(rgbs, disps)`` where both are stacks over time with shapes
        [N, H, W, 3] and [N, H, W] respectively.
    """

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed by reducing both resolution and focal length proportionally.
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # Simple timing print to monitor rendering speed
        print(i, time.time() - t)
        t = time.time()

        # Render the current pose; we discard the accumulated opacity and extras here
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        # Example: compute PSNR vs. ground truth if you have it (kept commented by default)
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        # Optionally write the frame to disk as an 8-bit PNG.
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    # Stack lists into contiguous arrays with a time dimension.
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps
