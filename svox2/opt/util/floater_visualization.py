"""
Floater visualization utilities for TensorBoard logging

This module helps visualize ghosting artifacts (floaters) detected by FDR.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2


def create_floater_overlay_grid(grid, fdr_results, highlight_color=(255, 0, 0)):
    """
    Create a modified density grid that highlights floaters
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR containing floater mask
        highlight_color: RGB color for floaters (default: red)
        
    Returns:
        floater_mask: Boolean tensor indicating floater voxels
        floater_coords: Nx3 tensor of floater voxel coordinates
    """
    if 'FDR_floater_mask_3d' not in fdr_results:
        return None, None
    
    labeled = fdr_results['FDR_floater_mask_3d']
    floater_ids = fdr_results['FDR_floater_component_ids']
    
    # Create boolean mask for floater voxels
    floater_mask_3d = np.isin(labeled, floater_ids)
    
    # Get coordinates of floater voxels
    floater_coords = np.stack(np.where(floater_mask_3d), axis=-1)
    
    return torch.from_numpy(floater_mask_3d), torch.from_numpy(floater_coords)


def render_floater_heatmap(grid, fdr_results, camera, render_func):
    """
    Render a view with floaters highlighted in a heatmap overlay
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        camera: Camera object for rendering
        render_func: Function to render images (from grid)
        
    Returns:
        rgb_with_overlay: Image with floater overlay [H, W, 3]
        floater_density: Floater-only rendering [H, W]
    """
    # Get floater mask
    floater_mask, floater_coords = create_floater_overlay_grid(grid, fdr_results)
    
    if floater_mask is None:
        return None, None
    
    # TODO: Implement actual rendering with floater highlighting
    # This would require modifying the rendering to separate floaters
    # For now, return placeholder
    
    return None, None


def find_best_floater_views(grid, fdr_results, cameras, n_views=3):
    """
    Find camera views that best show the floaters
    
    Strategy: Find views where floater density is highest
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        cameras: List of camera objects
        n_views: Number of best views to return
        
    Returns:
        best_view_indices: Indices of cameras with most floaters visible
    """
    if 'FDR_floater_mask_3d' not in fdr_results or len(cameras) == 0:
        return []
    
    floater_mask, floater_coords = create_floater_overlay_grid(grid, fdr_results)
    
    if floater_coords is None or len(floater_coords) == 0:
        return []
    
    # Convert floater coordinates to world space
    reso = torch.tensor(grid.links.shape)
    
    # Simple heuristic: compute distance to camera origins
    # Views closer to floaters should see them better
    floater_coords_world = floater_coords.float()
    
    # Normalize to [-1, 1] grid coordinates
    floater_coords_norm = (floater_coords_world / reso.float()) * 2 - 1
    
    # Score each camera by proximity to floaters
    scores = []
    for i, cam in enumerate(cameras):
        # Get camera position (simplified - assumes c2w available)
        if hasattr(cam, 'c2w'):
            cam_pos = cam.c2w[:3, 3]
            
            # Compute mean distance to floaters
            # (In practice, you'd want to project and check visibility)
            distances = torch.norm(floater_coords_world - cam_pos.unsqueeze(0), dim=1)
            score = 1.0 / (distances.mean() + 1e-6)  # Closer = higher score
            scores.append((i, score.item()))
    
    # Sort by score and return top n
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:n_views]]


def create_floater_density_slice(grid, fdr_results, slice_axis=2, slice_position=0.5):
    """
    Create a 2D slice through the 3D floater density field
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        slice_axis: Axis to slice along (0=X, 1=Y, 2=Z)
        slice_position: Position along axis (0-1)
        
    Returns:
        slice_image: 2D visualization of floater density [H, W, 3]
    """
    if 'FDR_floater_mask_3d' not in fdr_results:
        return None
    
    labeled = fdr_results['FDR_floater_mask_3d']
    floater_ids = fdr_results['FDR_floater_component_ids']
    main_id = fdr_results['FDR_main_component_id']
    
    # Create visualization: main object = gray, floaters = red, empty = black
    reso = labeled.shape
    slice_idx = int(reso[slice_axis] * slice_position)
    
    # Extract slice
    if slice_axis == 0:
        slice_2d = labeled[slice_idx, :, :]
    elif slice_axis == 1:
        slice_2d = labeled[:, slice_idx, :]
    else:  # axis == 2
        slice_2d = labeled[:, :, slice_idx]
    
    # Create RGB visualization
    vis = np.zeros((*slice_2d.shape, 3), dtype=np.uint8)
    
    # Main object = white/gray
    vis[slice_2d == main_id] = [200, 200, 200]
    
    # Floaters = red (intensity based on component size)
    for fid in floater_ids:
        mask = slice_2d == fid
        if mask.any():
            vis[mask] = [255, 0, 0]
    
    return vis


def create_floater_comparison_grid(before_slice, after_slice, labels=None):
    """
    Create side-by-side comparison of floater visualizations
    
    Args:
        before_slice: Floater visualization before improvements
        after_slice: Floater visualization after improvements
        labels: Optional text labels
        
    Returns:
        comparison_image: Side-by-side visualization
    """
    if before_slice is None or after_slice is None:
        return None
    
    # Ensure same size
    h = max(before_slice.shape[0], after_slice.shape[0])
    w = max(before_slice.shape[1], after_slice.shape[1])
    
    before_resized = cv2.resize(before_slice, (w, h))
    after_resized = cv2.resize(after_slice, (w, h))
    
    # Add labels if provided
    if labels:
        before_labeled = before_resized.copy()
        after_labeled = after_resized.copy()
        cv2.putText(before_labeled, labels[0], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(after_labeled, labels[1], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        comparison = np.hstack([before_labeled, after_labeled])
    else:
        comparison = np.hstack([before_resized, after_resized])
    
    return comparison


def project_floaters_to_view(grid, fdr_results, camera, render_size=None):
    """
    Project 3D floater positions onto a 2D camera view
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR with floater mask
        camera: Camera object for projection
        render_size: Optional (H, W) for output size
        
    Returns:
        floater_heatmap: 2D heatmap of floater projections [H, W]
    """
    if 'FDR_floater_mask_3d' not in fdr_results:
        return None
    
    floater_mask, floater_coords = create_floater_overlay_grid(grid, fdr_results)
    
    if floater_coords is None or len(floater_coords) == 0:
        return None
    
    # Get device from grid
    device = grid.links.device
    
    # Get grid parameters (ensure on correct device)
    reso = torch.tensor(grid.links.shape, device=device)
    
    # Ensure grid radius and center are on the correct device
    radius = grid.radius.to(device) if hasattr(grid.radius, 'to') else grid.radius
    center = grid.center.to(device) if hasattr(grid.center, 'to') else grid.center
    
    # Convert voxel coordinates to world space
    # Plenoxels grid: voxel [0,0,0] to [reso-1, reso-1, reso-1]
    # World space: [-radius, radius] centered at grid.center
    floater_coords_norm = (floater_coords.float().to(device) / reso.float()) * 2 - 1  # [-1, 1]
    floater_coords_world = floater_coords_norm * radius.unsqueeze(0) + center.unsqueeze(0)
    
    # Project to camera space
    if not hasattr(camera, 'c2w'):
        return None
    
    # Ensure camera transform is on the same device as grid
    c2w = camera.c2w.to(device) if hasattr(camera.c2w, 'to') else camera.c2w
    w2c = torch.inverse(c2w)  # World to camera transform
    
    # Transform floater points to camera space
    floater_coords_homo = torch.cat([
        floater_coords_world,
        torch.ones(len(floater_coords_world), 1, device=device)
    ], dim=1)  # [N, 4]
    
    floater_cam = (w2c @ floater_coords_homo.T).T  # [N, 4]
    
    # Project to image plane
    # Convert camera intrinsics to scalars (handle both tensors and floats)
    fx = float(camera.fx) if hasattr(camera.fx, 'item') else camera.fx
    fy = float(camera.fy) if camera.fy is not None and hasattr(camera.fy, 'item') else (float(camera.fy) if camera.fy is not None else fx)
    if camera.fy is None:
        fy = fx
    
    cx = camera.width * 0.5
    cy = camera.height * 0.5
    if camera.cx is not None:
        cx = float(camera.cx) if hasattr(camera.cx, 'item') else camera.cx
    if camera.cy is not None:
        cy = float(camera.cy) if hasattr(camera.cy, 'item') else camera.cy
    
    # Perspective projection (results stay on device)
    x_img = (floater_cam[:, 0] / floater_cam[:, 2]) * fx + cx
    y_img = (floater_cam[:, 1] / floater_cam[:, 2]) * fy + cy
    z_depth = floater_cam[:, 2]
    
    # Filter points behind camera or outside image
    valid = (z_depth > 0) & (x_img >= 0) & (x_img < camera.width) & \
            (y_img >= 0) & (y_img < camera.height)
    
    x_img_valid = x_img[valid].long()
    y_img_valid = y_img[valid].long()
    
    # Create heatmap (on CPU for final output)
    H, W = camera.height, camera.width
    if render_size is not None:
        H, W = render_size
    
    heatmap = torch.zeros(H, W, dtype=torch.float32)
    
    # Move image coordinates back to CPU for heatmap creation
    x_img_valid = x_img_valid.cpu()
    y_img_valid = y_img_valid.cpu()
    
    if len(x_img_valid) > 0:
        # Accumulate floater projections
        for x, y in zip(x_img_valid, y_img_valid):
            if 0 <= y < H and 0 <= x < W:
                heatmap[y, x] += 1
    
    heatmap_np = heatmap.numpy()
    
    # No dilation - pixel-perfect floater locations
    return heatmap_np


def create_floater_overlay_on_render(rgb_image, floater_heatmap, alpha=0.8):
    """
    Overlay floater heatmap onto rendered RGB image with transparent background
    
    Args:
        rgb_image: Rendered RGB image [H, W, 3] in range [0, 1]
        floater_heatmap: 2D heatmap of floater projections [H, W]
        alpha: Opacity of floater highlights (default 0.8 for bright red)
        
    Returns:
        overlay_image: RGB image with floater overlay [H, W, 3]
    """
    if floater_heatmap is None or rgb_image is None:
        return rgb_image
    
    # Normalize heatmap
    if floater_heatmap.max() > 0:
        heatmap_norm = floater_heatmap / floater_heatmap.max()
    else:
        return rgb_image
    
    # Make background much more transparent (dim to 30% opacity where no floaters)
    background_opacity = 0.3
    floater_mask = heatmap_norm > 0
    
    # Start with dimmed background
    result = rgb_image * background_opacity
    
    # Where there are floaters, blend in bright red
    red_overlay = np.zeros_like(rgb_image)
    red_overlay[:, :, 0] = heatmap_norm  # Bright red channel
    
    # Blend: dim background everywhere, bright red where floaters exist
    result = result * (1 - alpha * heatmap_norm[:, :, np.newaxis]) + \
             red_overlay * alpha + \
             rgb_image * (1 - background_opacity) * (1 - floater_mask[:, :, np.newaxis])
    
    result = np.clip(result, 0, 1)
    
    return result


def render_density_from_camera(grid, camera, colormap='viridis'):
    """
    Render the density field from a camera view for validation
    
    This helps visualize what the grid "sees" to validate floater detection.
    Floaters should appear as small, disconnected bright regions.
    
    Args:
        grid: SparseGrid instance
        camera: Camera object
        colormap: Color map name ('viridis', 'plasma', 'hot', 'gray')
        
    Returns:
        density_render: Rendered density visualization [H, W, 3]
    """
    # Volume render density (not RGB)
    # We'll sample along rays and accumulate density
    import svox2
    
    # Render with special mode to get density accumulation
    with torch.no_grad():
        # Use volume_render_depth as a proxy - it shows density-weighted depth
        # which gives us a sense of where density exists
        try:
            depth = grid.volume_render_depth_image(camera, sigma_thresh=0.0)
            depth_np = depth.cpu().numpy()
            
            # Normalize to [0, 1]
            if depth_np.max() > depth_np.min():
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            else:
                depth_norm = np.zeros_like(depth_np)
            
            # Apply colormap
            if colormap == 'viridis':
                cmap = cv2.COLORMAP_VIRIDIS
            elif colormap == 'plasma':
                cmap = cv2.COLORMAP_PLASMA
            elif colormap == 'hot':
                cmap = cv2.COLORMAP_HOT
            else:  # gray
                cmap = cv2.COLORMAP_BONE
            
            density_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cmap)
            density_colored = cv2.cvtColor(density_colored, cv2.COLOR_BGR2RGB)
            
            return density_colored / 255.0
        except Exception as e:
            print(f"Warning: Could not render density: {e}")
            return None


def visualize_grid_structure(grid, camera, base_image=None, render_size=None, max_points=None, chunk_subdivisions=8, background_opacity=0.75, mode='boundary'):
    """
    Visualize the sparse grid structure by showing active regions
    
    Shows:
    - Overall grid bounding box (white, thick)
    - Active chunk boxes (cyan) - subdivisions of the grid that contain active voxels
    - Active voxel centers (green dots) - ALL active voxels from sparse grid
    
    This gives an exact 1:1 visualization of the sparse grid structure.
    Every voxel with grid.links[i,j,k] >= 0 is rendered.
    
    Args:
        grid: SparseGrid instance
        camera: Camera object with c2w, intrinsics, etc.
        base_image: Optional base image to overlay on [H, W, 3], if None uses black background
        render_size: Optional (H, W) tuple for output size (only used if base_image is None)
        max_points: Maximum number of voxel centers to show as dots (None = show all)
        chunk_subdivisions: Number of subdivisions per axis (e.g., 8 means 8x8x8=512 chunks)
        background_opacity: Opacity of background image (0.0-1.0), lower = more dimmed
        mode: 'boundary' (clean shell) or 'all' (show all chunk boxes)
        
    Returns:
        RGB image with grid structure overlay [H, W, 3]
    """
    if base_image is not None:
        # Apply opacity to background for better grid visibility
        vis_image = (base_image * 255 * background_opacity).astype(np.uint8).copy()
        H, W = vis_image.shape[:2]
    else:
        if render_size is None:
            render_size = (800, 800)
        H, W = render_size
        vis_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    device = grid.sh_data.device
    
    # Helper function to project 3D point to 2D
    def project_point(world_pt, c2w, fx, fy, cx, cy):
        """Project a 3D world point to 2D image coordinates"""
        w2c = torch.inverse(c2w)
        world_pt_h = torch.cat([world_pt, torch.ones(1, device=device)])
        cam_pt = w2c @ world_pt_h
        
        if cam_pt[2] <= 0:  # Behind camera
            return None
            
        x = (cam_pt[0] / cam_pt[2]) * fx + cx
        y = (cam_pt[1] / cam_pt[2]) * fy + cy
        
        return (int(x.item()), int(y.item()))
    
    # Get camera parameters
    c2w = camera.c2w.to(device)
    fx = float(camera.fx) if hasattr(camera, 'fx') else float(camera.intrins[0, 0])
    fy = float(camera.fy) if hasattr(camera, 'fy') else float(camera.intrins[1, 1])
    cx = float(camera.cx) if hasattr(camera, 'cx') else float(camera.intrins[0, 2])
    cy = float(camera.cy) if hasattr(camera, 'cy') else float(camera.intrins[1, 2])
    
    # 1. Draw overall grid bounding box (8 corners, 12 edges)
    grid_center = grid.center.to(device)
    grid_radius = grid.radius.to(device)
    
    # 8 corners of the bounding box
    corners_offset = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
    ], device=device, dtype=torch.float32)
    
    corners_world = corners_offset * grid_radius + grid_center
    
    # Project corners
    corners_2d = []
    for corner in corners_world:
        pt_2d = project_point(corner, c2w, fx, fy, cx, cy)
        corners_2d.append(pt_2d)
    
    # Draw edges of bounding box (cyan color)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    for i, j in edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            pt1, pt2 = corners_2d[i], corners_2d[j]
            if 0 <= pt1[0] < W*2 and 0 <= pt1[1] < H*2 and 0 <= pt2[0] < W*2 and 0 <= pt2[1] < H*2:
                cv2.line(vis_image, pt1, pt2, (255, 255, 255), 3)  # White, thick
    
    # 2. Subdivide grid into chunks and draw boxes for active chunks (octree-like visualization)
    reso = torch.tensor(grid.links.shape, device=device, dtype=torch.float32)
    chunk_size = (reso / chunk_subdivisions).int()
    
    active_mask = grid.links >= 0
    
    if mode == 'all':
        # Mode 1: Draw ALL active chunk boxes (shows overlaps, more detailed)
        # Also compute chunk density for color coding
        for i in range(chunk_subdivisions):
            for j in range(chunk_subdivisions):
                for k in range(chunk_subdivisions):
                    start_i = i * chunk_size[0]
                    start_j = j * chunk_size[1]
                    start_k = k * chunk_size[2]
                    end_i = min((i + 1) * chunk_size[0], int(reso[0]))
                    end_j = min((j + 1) * chunk_size[1], int(reso[1]))
                    end_k = min((k + 1) * chunk_size[2], int(reso[2]))
                    
                    chunk_region = active_mask[start_i:end_i, start_j:end_j, start_k:end_k]
                    num_active = chunk_region.sum().item()
                    chunk_volume = chunk_region.numel()
                    
                    if num_active > 0:
                        # Compute density: percentage of voxels that are active
                        density = num_active / chunk_volume
                        chunk_min = torch.tensor([i, j, k], device=device, dtype=torch.float32)
                        chunk_max = torch.tensor([i+1, j+1, k+1], device=device, dtype=torch.float32)
                        
                        chunk_min_norm = (chunk_min / chunk_subdivisions) * 2 - 1
                        chunk_max_norm = (chunk_max / chunk_subdivisions) * 2 - 1
                        
                        chunk_corners = torch.tensor([
                            [chunk_min_norm[0], chunk_min_norm[1], chunk_min_norm[2]],
                            [chunk_max_norm[0], chunk_min_norm[1], chunk_min_norm[2]],
                            [chunk_max_norm[0], chunk_max_norm[1], chunk_min_norm[2]],
                            [chunk_min_norm[0], chunk_max_norm[1], chunk_min_norm[2]],
                            [chunk_min_norm[0], chunk_min_norm[1], chunk_max_norm[2]],
                            [chunk_max_norm[0], chunk_min_norm[1], chunk_max_norm[2]],
                            [chunk_max_norm[0], chunk_max_norm[1], chunk_max_norm[2]],
                            [chunk_min_norm[0], chunk_max_norm[1], chunk_max_norm[2]]
                        ], device=device, dtype=torch.float32)
                        
                        chunk_corners_world = chunk_corners * grid_radius + grid_center
                        
                        chunk_corners_2d = []
                        for corner in chunk_corners_world:
                            pt_2d = project_point(corner, c2w, fx, fy, cx, cy)
                            chunk_corners_2d.append(pt_2d)
                        
                        # Draw all 12 edges
                        for edge_i, edge_j in edges:
                            if chunk_corners_2d[edge_i] is not None and chunk_corners_2d[edge_j] is not None:
                                pt1, pt2 = chunk_corners_2d[edge_i], chunk_corners_2d[edge_j]
                                if 0 <= pt1[0] < W*2 and 0 <= pt1[1] < H*2 and 0 <= pt2[0] < W*2 and 0 <= pt2[1] < H*2:
                                    cv2.line(vis_image, pt1, pt2, (0, 255, 255), 1)  # Cyan
    
    else:  # mode == 'boundary'
        # Mode 2: Draw ONLY boundary edges (clean shell, no overlaps)
        # Create a 3D grid indicating which chunks are active
        chunk_grid = torch.zeros((chunk_subdivisions, chunk_subdivisions, chunk_subdivisions), dtype=torch.bool, device=device)
        
        for i in range(chunk_subdivisions):
            for j in range(chunk_subdivisions):
                for k in range(chunk_subdivisions):
                    start_i = i * chunk_size[0]
                    start_j = j * chunk_size[1]
                    start_k = k * chunk_size[2]
                    end_i = min((i + 1) * chunk_size[0], int(reso[0]))
                    end_j = min((j + 1) * chunk_size[1], int(reso[1]))
                    end_k = min((k + 1) * chunk_size[2], int(reso[2]))
                    
                    chunk_grid[i, j, k] = active_mask[start_i:end_i, start_j:end_j, start_k:end_k].any()
        
        # Draw only boundary edges (where active meets inactive or boundary)
        for i in range(chunk_subdivisions):
            for j in range(chunk_subdivisions):
                for k in range(chunk_subdivisions):
                    if not chunk_grid[i, j, k]:
                        continue  # Skip inactive chunks
                    
                    # Get chunk bounds in normalized coords
                    chunk_min = torch.tensor([i, j, k], device=device, dtype=torch.float32)
                    chunk_max = torch.tensor([i+1, j+1, k+1], device=device, dtype=torch.float32)
                    
                    chunk_min_norm = (chunk_min / chunk_subdivisions) * 2 - 1
                    chunk_max_norm = (chunk_max / chunk_subdivisions) * 2 - 1
                    
                    # Define 6 faces and check neighbors
                    faces = [
                        # (face_corners, neighbor_offset, face_name)
                        ([0, 1, 2, 3], (-1, 0, 0), 'x_min'),  # Left face
                        ([4, 5, 6, 7], (1, 0, 0), 'x_max'),   # Right face
                        ([0, 1, 5, 4], (0, -1, 0), 'y_min'),  # Front face
                        ([2, 3, 7, 6], (0, 1, 0), 'y_max'),   # Back face
                        ([0, 3, 7, 4], (0, 0, -1), 'z_min'),  # Bottom face
                        ([1, 2, 6, 5], (0, 0, 1), 'z_max'),   # Top face
                    ]
                    
                    # Get 8 corners of this chunk
                    chunk_corners = torch.tensor([
                        [chunk_min_norm[0], chunk_min_norm[1], chunk_min_norm[2]],
                        [chunk_max_norm[0], chunk_min_norm[1], chunk_min_norm[2]],
                        [chunk_max_norm[0], chunk_max_norm[1], chunk_min_norm[2]],
                        [chunk_min_norm[0], chunk_max_norm[1], chunk_min_norm[2]],
                        [chunk_min_norm[0], chunk_min_norm[1], chunk_max_norm[2]],
                        [chunk_max_norm[0], chunk_min_norm[1], chunk_max_norm[2]],
                        [chunk_max_norm[0], chunk_max_norm[1], chunk_max_norm[2]],
                        [chunk_min_norm[0], chunk_max_norm[1], chunk_max_norm[2]]
                    ], device=device, dtype=torch.float32)
                    
                    chunk_corners_world = chunk_corners * grid_radius + grid_center
                    
                    # Project all corners once
                    chunk_corners_2d = []
                    for corner in chunk_corners_world:
                        pt_2d = project_point(corner, c2w, fx, fy, cx, cy)
                        chunk_corners_2d.append(pt_2d)
                    
                    # Draw edges for faces that are boundaries (neighbor is inactive or out of bounds)
                    for face_corners, neighbor_offset, _ in faces:
                        ni, nj, nk = i + neighbor_offset[0], j + neighbor_offset[1], k + neighbor_offset[2]
                        
                        # Draw face if neighbor is out of bounds or inactive
                        is_boundary = (ni < 0 or ni >= chunk_subdivisions or
                                       nj < 0 or nj >= chunk_subdivisions or
                                       nk < 0 or nk >= chunk_subdivisions or
                                       not chunk_grid[ni, nj, nk])
                        
                        if is_boundary:
                            # Draw the 4 edges of this face
                            face_edges = [
                                (face_corners[0], face_corners[1]),
                                (face_corners[1], face_corners[2]),
                                (face_corners[2], face_corners[3]),
                                (face_corners[3], face_corners[0])
                            ]
                            
                            for idx1, idx2 in face_edges:
                                if chunk_corners_2d[idx1] is not None and chunk_corners_2d[idx2] is not None:
                                    pt1, pt2 = chunk_corners_2d[idx1], chunk_corners_2d[idx2]
                                    if 0 <= pt1[0] < W*2 and 0 <= pt1[1] < H*2 and 0 <= pt2[0] < W*2 and 0 <= pt2[1] < H*2:
                                        cv2.line(vis_image, pt1, pt2, (0, 255, 255), 1)  # Cyan
    
    # 3. Draw ALL active voxel centers (green dots) - 1:1 accurate to sparse grid
    # Get indices of ALL active voxels (where grid.links >= 0)
    active_indices = torch.nonzero(active_mask, as_tuple=False).float()  # [N, 3]
    
    total_active = active_indices.shape[0]
    print(f"  Rendering {total_active:,} active voxels (1:1 with sparse grid)")
    
    # Sample if max_points is specified
    if max_points is not None and total_active > max_points:
        print(f"  WARNING: Sampling {max_points:,} voxels (set max_points=None for full accuracy)")
        indices = torch.randperm(total_active)[:max_points]
        sampled_indices = active_indices[indices]
    else:
        sampled_indices = active_indices
    
    # Convert voxel indices to world coordinates
    # IMPORTANT: grid.links uses indices [i, j, k] where each is in [0, reso-1]
    # The voxel CENTER is at (i+0.5, j+0.5, k+0.5) in voxel space
    voxel_coords = (sampled_indices + 0.5) / reso  # Normalize to [0, 1], centers at 0.5/reso spacing
    voxel_coords = voxel_coords * 2 - 1  # Scale to [-1, 1]
    world_coords = voxel_coords * grid_radius + grid_center
    
    # Project to image
    w2c = torch.inverse(c2w)
    world_coords_h = torch.cat([world_coords, torch.ones(world_coords.shape[0], 1, device=device)], dim=1)
    cam_coords = (w2c @ world_coords_h.T).T[:, :3]
    
    x = (cam_coords[:, 0] / cam_coords[:, 2]) * fx + cx
    y = (cam_coords[:, 1] / cam_coords[:, 2]) * fy + cy
    z = cam_coords[:, 2]
    
    valid = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x_valid = x[valid].cpu().numpy()
    y_valid = y[valid].cpu().numpy()
    
    for xi, yi in zip(x_valid, y_valid):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(vis_image, (xi, yi), 1, (0, 255, 0), -1)  # Green
    
    return vis_image


def log_floater_visualizations_to_tensorboard(
    summary_writer,
    grid,
    fdr_results,
    global_step,
    n_slices=3,
    cameras=None,
    rendered_images=None,
    gt_images=None,
    max_render_views=3,
    log_density_renders=False
):
    """
    Log comprehensive floater visualizations to TensorBoard
    
    Args:
        summary_writer: TensorBoard SummaryWriter
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        global_step: Training iteration for logging
        n_slices: Number of slices to visualize per axis
        cameras: Optional list of camera objects for projection overlays
        rendered_images: Optional list of rendered RGB images [H, W, 3]
        gt_images: Optional list of ground truth RGB images [H, W, 3]
        max_render_views: Maximum number of rendered views to overlay
        log_density_renders: If True, also log density field renders for validation
    """
    if 'FDR_floater_mask_3d' not in fdr_results:
        print("  Warning: No floater mask available for visualization")
        return
    
    # Create slices along different axes
    for axis, axis_name in enumerate(['X', 'Y', 'Z']):
        for i, pos in enumerate(np.linspace(0.3, 0.7, n_slices)):
            slice_vis = create_floater_density_slice(
                grid, fdr_results,
                slice_axis=axis,
                slice_position=pos
            )
            
            if slice_vis is not None:
                # Convert to format TensorBoard expects [C, H, W]
                slice_vis_tb = slice_vis.transpose(2, 0, 1)
                summary_writer.add_image(
                    f'floaters/slice_{axis_name}_{i}',
                    slice_vis_tb,
                    global_step=global_step,
                    dataformats='CHW'
                )
    
    # Create floater projection overlays on rendered views
    if cameras is not None and rendered_images is not None:
        n_views = min(len(cameras), len(rendered_images), max_render_views)
        
        for i in range(n_views):
            camera = cameras[i]
            rgb_image = rendered_images[i]
            gt_image = gt_images[i] if gt_images is not None and i < len(gt_images) else None
            
            # Always log GT and render views (these should always work)
            try:
                if gt_image is not None:
                    gt_tb = gt_image.transpose(2, 0, 1)  # [C, H, W]
                    summary_writer.add_image(
                        f'floaters/gt_view_{i}',
                        gt_tb,
                        global_step=global_step,
                        dataformats='CHW'
                    )
                
                rgb_tb = rgb_image.transpose(2, 0, 1)  # [C, H, W]
                summary_writer.add_image(
                    f'floaters/render_view_{i}',
                    rgb_tb,
                    global_step=global_step,
                    dataformats='CHW'
                )
                
                # Optionally render density field for validation
                if log_density_renders:
                    density_render = render_density_from_camera(grid, camera, colormap='hot')
                    if density_render is not None:
                        density_tb = density_render.transpose(2, 0, 1)
                        summary_writer.add_image(
                            f'floaters/density_view_{i}',
                            density_tb,
                            global_step=global_step,
                            dataformats='CHW'
                        )
                
                # Log grid structure visualization (always) - overlay on rendered image
                # Mode 1: Boundary only (clean shell) with ALL voxels
                grid_structure_boundary = visualize_grid_structure(
                    grid, 
                    camera,
                    base_image=rgb_image,  # Overlay on rendered image
                    max_points=None,  # Render ALL active voxels (1:1 accuracy)
                    chunk_subdivisions=8,  # Divide grid into 8x8x8 chunks
                    background_opacity=0.75,  # Dim background for better visibility
                    mode='boundary'
                )
                grid_boundary_normalized = grid_structure_boundary / 255.0
                grid_boundary_tb = grid_boundary_normalized.transpose(2, 0, 1)
                summary_writer.add_image(
                    f'floaters/grid_boundary_view_{i}',
                    grid_boundary_tb,
                    global_step=global_step,
                    dataformats='CHW'
                )
                
                # Mode 2: All boxes (shows every active chunk with overlaps) with ALL voxels
                grid_structure_all = visualize_grid_structure(
                    grid, 
                    camera,
                    base_image=rgb_image,  # Overlay on rendered image
                    max_points=None,  # Render ALL active voxels (1:1 accuracy)
                    chunk_subdivisions=8,  # Divide grid into 8x8x8 chunks
                    background_opacity=0.75,  # Dim background for better visibility
                    mode='all'
                )
                grid_all_normalized = grid_structure_all / 255.0
                grid_all_tb = grid_all_normalized.transpose(2, 0, 1)
                summary_writer.add_image(
                    f'floaters/grid_all_boxes_view_{i}',
                    grid_all_tb,
                    global_step=global_step,
                    dataformats='CHW'
                )
            except Exception as e:
                print(f"  Warning: Failed to log images for view {i}: {e}")
            
            # Try to project floaters and create overlays
            try:
                # Debug: Check devices
                # print(f"  Debug view {i}: grid.links device = {grid.links.device}, camera.c2w device = {camera.c2w.device if hasattr(camera.c2w, 'device') else 'N/A'}")
                floater_heatmap = project_floaters_to_view(grid, fdr_results, camera)
                
                if floater_heatmap is not None:
                    # Create overlay with enhanced visibility (dimmed background, bright red floaters)
                    overlay_image = create_floater_overlay_on_render(
                        rgb_image, floater_heatmap  # Uses default alpha=0.8 for bright highlights
                    )
                    
                    # Log overlay to TensorBoard
                    overlay_tb = overlay_image.transpose(2, 0, 1)  # [C, H, W]
                    summary_writer.add_image(
                        f'floaters/overlay_view_{i}',
                        overlay_tb,
                        global_step=global_step,
                        dataformats='CHW'
                    )
                    
                    # Also log just the heatmap
                    heatmap_colored = cv2.applyColorMap(
                        (floater_heatmap / max(floater_heatmap.max(), 1e-6) * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    heatmap_tb = heatmap_colored.transpose(2, 0, 1)
                    summary_writer.add_image(
                        f'floaters/heatmap_view_{i}',
                        heatmap_tb,
                        global_step=global_step,
                        dataformats='CHW'
                    )
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to create floater overlay for view {i}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
    
    # Log summary text
    summary_text = f"""
    FDR: {fdr_results['FDR']:.2%}
    Num Floaters: {fdr_results['FDR_num_floaters']}
    Main Volume: {fdr_results['FDR_main_volume']:,} voxels
    Floater Volume: {fdr_results['FDR_floater_volume']:,} voxels
    Largest Floater: {fdr_results['FDR_largest_floater']:,} voxels
    """
    
    summary_writer.add_text(
        'floaters/summary',
        summary_text,
        global_step=global_step
    )
    
    n_vis = n_slices * 3
    if cameras is not None and rendered_images is not None:
        # gt + render + heatmap + overlay = 4 per view (or 3 if no GT)
        n_views = min(len(cameras), len(rendered_images), max_render_views)
        imgs_per_view = 4 if gt_images is not None else 3
        n_vis += n_views * imgs_per_view
    print(f"  Logged {n_vis} floater visualizations to TensorBoard")

