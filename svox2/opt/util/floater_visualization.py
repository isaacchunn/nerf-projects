"""
Floater visualization utilities for TensorBoard logging

Provides visualization functions for FDR (Floater Detection Ratio) analysis:
- Multi-object detection with unique colors per object
- Floater overlay on rendered views
- Grid structure visualization
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2


# ============================================================================
# Core Visualization Functions
# ============================================================================


def create_floater_overlay_grid(grid, fdr_results, highlight_color=(255, 0, 0)):
    """Extract floater voxel coordinates from FDR results"""
    if 'FDR_floater_mask_3d' not in fdr_results:
        return None, None
    
    labeled = fdr_results['FDR_floater_mask_3d']
    floater_ids = fdr_results['FDR_floater_component_ids']
    floater_mask_3d = np.isin(labeled, floater_ids)
    floater_coords = np.stack(np.where(floater_mask_3d), axis=-1)
    
    return torch.from_numpy(floater_mask_3d), torch.from_numpy(floater_coords)


def create_main_object_overlay_grid(grid, fdr_results):
    """Extract main object voxel coordinates from FDR results"""
    if 'FDR_floater_mask_3d' not in fdr_results:
        return None, None
    
    labeled = fdr_results['FDR_floater_mask_3d']
    
    if 'FDR_main_component_ids' in fdr_results:
        main_ids = fdr_results['FDR_main_component_ids']
    elif 'FDR_main_component_id' in fdr_results:
        main_ids = [fdr_results['FDR_main_component_id']]
    else:
        return None, None
    
    main_mask_3d = np.isin(labeled, main_ids)
    main_coords = np.stack(np.where(main_mask_3d), axis=-1)
    
    return torch.from_numpy(main_mask_3d), torch.from_numpy(main_coords)


# ============================================================================
# 3D Projection and Overlay
# ============================================================================


def project_floaters_to_view(grid, fdr_results, camera, render_size=None, 
                              filter_occluded=True, min_density=0.1):
    """
    Project 3D floater positions onto a 2D camera view with visibility filtering
    
    This function now filters floaters to only show VISIBLE artifacts:
    1. Density filtering: Removes imperceptible low-density floaters (< min_density)
    2. Occlusion filtering: Removes floaters behind rendered geometry
    3. Result: Only shows floaters that actually contribute to visible artifacts
    
    Args:
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR with floater mask
        camera: Camera object for projection
        render_size: Optional (H, W) for output size
        filter_occluded: If True, filter floaters behind rendered geometry (default: True)
        min_density: Minimum density threshold for visible floaters (default: 0.1)
        
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
    
    # Filter by density threshold to remove imperceptible floaters
    if min_density > 0:
        # Get density values for floater voxels
        floater_densities = []
        for coord in floater_coords:
            i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
            link_idx = grid.links[i, j, k].item()
            if link_idx >= 0:
                density = grid.density_data[link_idx].item()
                floater_densities.append(density)
            else:
                floater_densities.append(0.0)
        
        floater_densities = torch.tensor(floater_densities, device=device)
        density_mask = floater_densities >= min_density
        
        # Filter out low-density floaters
        floater_coords = floater_coords[density_mask.cpu()]
        
        if len(floater_coords) == 0:
            # No visible floaters after density filtering
            H, W = camera.height, camera.width
            if render_size is not None:
                H, W = render_size
            return np.zeros((H, W), dtype=np.float32)
    
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
    z_depth_valid = z_depth[valid]
    
    # Create heatmap (on CPU for final output)
    H, W = camera.height, camera.width
    if render_size is not None:
        H, W = render_size
    
    heatmap = torch.zeros(H, W, dtype=torch.float32)
    
    # Filter occluded floaters by comparing with rendered depth
    if filter_occluded and len(x_img_valid) > 0:
        try:
            # Render depth map from this camera view
            depth_map = grid.volume_render_depth_image(camera, sigma_thresh=0.0)
            depth_map = depth_map.to(device)
            
            # Sample depth at floater pixel locations
            scene_depths = []
            visibility_mask = []
            
            for i, (x, y, floater_depth) in enumerate(zip(x_img_valid, y_img_valid, z_depth_valid)):
                if 0 <= y < H and 0 <= x < W:
                    scene_depth = depth_map[y, x].item()
                    # Floater is visible if it's in front of (or near) the rendered surface
                    # Add small epsilon to handle numerical precision
                    is_visible = (floater_depth < scene_depth + 0.05) or (scene_depth < 0.01)
                    visibility_mask.append(is_visible)
                else:
                    visibility_mask.append(False)
            
            visibility_mask = torch.tensor(visibility_mask, device=device)
            x_img_valid = x_img_valid[visibility_mask]
            y_img_valid = y_img_valid[visibility_mask]
            z_depth_valid = z_depth_valid[visibility_mask]
            
            print(f"  Floater visibility: {visibility_mask.sum().item()}/{len(visibility_mask)} visible after occlusion filtering")
        except Exception as e:
            print(f"  Warning: Could not filter occluded floaters: {e}")
    
    # Move to CPU for heatmap creation
    x_img_valid = x_img_valid.cpu()
    y_img_valid = y_img_valid.cpu()
    z_depth_valid = z_depth_valid.cpu().numpy()
    
    if len(x_img_valid) > 0:
        # Accumulate floater projections with depth
        for x, y, depth in zip(x_img_valid, y_img_valid, z_depth_valid):
            if 0 <= y < H and 0 <= x < W:
                heatmap[y, x] += 1
    
    heatmap_np = heatmap.numpy()
    
    # Dilate to represent voxel volume (not just centers)
    # Each floater voxel affects nearby pixels during rendering
    if heatmap_np.max() > 0:
        import cv2
        kernel_size = 3  # Adjust based on voxel size relative to image resolution
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        heatmap_np = cv2.dilate(heatmap_np, kernel, iterations=1)
    
    return heatmap_np


def create_floater_overlay_on_render(rgb_image, floater_heatmap, alpha=0.9):
    """
    Overlay floater heatmap onto rendered RGB image with improved visibility
    
    Args:
        rgb_image: Rendered RGB image [H, W, 3] in range [0, 1]
        floater_heatmap: 2D heatmap of floater projections [H, W]
        alpha: Opacity of floater highlights (default 0.9 for high visibility)
        
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
    
    # Create binary mask for floater regions
    floater_mask = heatmap_norm > 0
    
    # Keep original image brightness, just add red tint to floaters
    result = rgb_image.copy()
    
    # Add pure red overlay to floater regions (consistent with all views)
    floater_color = np.array([1.0, 0.0, 0.0])  # Pure red
    
    # Blend floater color with original image
    for c in range(3):
        result[:, :, c] = np.where(
            floater_mask,
            (1 - alpha) * rgb_image[:, :, c] + alpha * floater_color[c] * heatmap_norm,
            rgb_image[:, :, c]
        )
    
    # Add subtle bright borders around floater regions for better visibility
    import cv2
    floater_mask_uint8 = (floater_mask * 255).astype(np.uint8)
    # Find edges
    edges = cv2.Canny(floater_mask_uint8, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = edges_dilated > 0
    
    # Add bright red border (consistent with floater color)
    border_color = np.array([1.0, 0.0, 0.0])  # Pure red
    for c in range(3):
        result[:, :, c] = np.where(edge_mask, border_color[c], result[:, :, c])
    
    result = np.clip(result, 0, 1)
    
    return result


def create_multi_object_voxel_overlay(rgb_image, grid, fdr_results, camera,
                                       max_points_per_object=50000, alpha=0.7,
                                       show_floaters=True, min_viz_size=5000):
    """
    Overlay ALL objects (main + floaters) onto rendered image with different colors
    
    Each main object gets a unique color, floaters are red. This clearly shows:
    - Which voxels belong to the scene (colored by object)
    - Spatial gaps between objects  
    - Which voxels are floaters (red)
    
    Args:
        rgb_image: Rendered RGB image [H, W, 3] in range [0, 1]
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        camera: Camera object
        max_points_per_object: Max voxels to show per object (for performance)
        alpha: Opacity of voxel markers (default 0.7)
        show_floaters: If True, also show floaters in red
        min_viz_size: Minimum voxel count to visualize an object (filters tiny components from view)
        
    Returns:
        overlay_image: RGB image with all objects overlaid [H, W, 3]
    """
    if rgb_image is None or fdr_results is None:
        return rgb_image
    
    if 'FDR_floater_mask_3d' not in fdr_results:
        return rgb_image
    
    labeled = fdr_results['FDR_floater_mask_3d']
    
    # Get main object IDs
    if 'FDR_main_component_ids' in fdr_results:
        main_ids = fdr_results['FDR_main_component_ids']
    elif 'FDR_main_component_id' in fdr_results:
        main_ids = [fdr_results['FDR_main_component_id']]
    else:
        return rgb_image
    
    # Get floater IDs
    floater_ids = fdr_results.get('FDR_floater_component_ids', [])
    
    # Define colors for each main object - 12 distinct colors
    object_colors = [
        [0, 255, 0],        # 1. Green
        [0, 150, 255],      # 2. Blue
        [255, 200, 0],      # 3. Yellow/Gold
        [255, 0, 255],      # 4. Magenta
        [0, 255, 255],      # 5. Cyan
        [255, 128, 0],      # 6. Orange
        [128, 0, 255],      # 7. Purple
        [255, 255, 128],    # 8. Light Yellow
        [255, 128, 255],    # 9. Pink
        [128, 255, 0],      # 10. Lime Green
        [0, 255, 128],      # 11. Spring Green
        [128, 128, 255],    # 12. Light Blue
    ]
    
    device = grid.links.device
    reso = torch.tensor(grid.links.shape, device=device)
    radius = grid.radius.to(device) if hasattr(grid.radius, 'to') else grid.radius
    center = grid.center.to(device) if hasattr(grid.center, 'to') else grid.center
    
    # Project camera matrices
    c2w = camera.c2w.to(device) if hasattr(camera.c2w, 'to') else camera.c2w
    w2c = torch.inverse(c2w)
    
    fx = float(camera.fx) if hasattr(camera.fx, 'item') else camera.fx
    fy = float(camera.fy) if camera.fy is not None else fx
    cx = camera.width * 0.5
    cy = camera.height * 0.5
    
    H, W = camera.height, camera.width
    
    # Create overlay with depth buffer for proper occlusion
    vis_image = (rgb_image * 255).astype(np.uint8)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)  # Initialize with infinity
    color_buffer = np.zeros((H, W, 3), dtype=np.uint8)  # RGB color at each pixel
    
    # Simple filtering: show all main objects that pass the min_viz_size threshold
    # No complex compactness checks - if FDR says it's a main object, visualize it
    filtered_main_ids = []
    for main_id in main_ids:
        obj_mask = (labeled == main_id)
        obj_size = np.sum(obj_mask)
        
        # Simple size filter
        if obj_size >= min_viz_size:
            filtered_main_ids.append(main_id)
    
    # Collect all points with their depths and colors first
    all_points = []  # List of (x, y, depth, color_bgr)
    
    # Process each main object with different color
    for obj_idx, main_id in enumerate(filtered_main_ids):
        # Get voxels for this object
        obj_mask = (labeled == main_id)
        obj_coords = np.stack(np.where(obj_mask), axis=-1)
        
        if len(obj_coords) == 0:
            continue
        
        # Subsample if needed
        if len(obj_coords) > max_points_per_object:
            indices = np.random.choice(len(obj_coords), max_points_per_object, replace=False)
            obj_coords = obj_coords[indices]
        
        # Convert to world space and project
        obj_coords_tensor = torch.from_numpy(obj_coords).float().to(device)
        obj_coords_norm = (obj_coords_tensor / reso.float()) * 2 - 1
        obj_coords_world = obj_coords_norm * radius.unsqueeze(0) + center.unsqueeze(0)
        
        obj_coords_homo = torch.cat([
            obj_coords_world,
            torch.ones(len(obj_coords_world), 1, device=device)
        ], dim=1)
        
        obj_cam = (w2c @ obj_coords_homo.T).T
        
        x_img = (obj_cam[:, 0] / obj_cam[:, 2]) * fx + cx
        y_img = (obj_cam[:, 1] / obj_cam[:, 2]) * fy + cy
        z_depth = obj_cam[:, 2]
        
        valid = (z_depth > 0) & (x_img >= 0) & (x_img < W) & (y_img >= 0) & (y_img < H)
        
        x_valid = x_img[valid].long().cpu().numpy()
        y_valid = y_img[valid].long().cpu().numpy()
        z_valid = z_depth[valid].cpu().numpy()
        
        # Get color for this object (BGR for OpenCV)
        color_rgb = object_colors[obj_idx % len(object_colors)]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        
        # Add points to list
        for x, y, z in zip(x_valid, y_valid, z_valid):
            if 0 <= x < W and 0 <= y < H:
                all_points.append((int(x), int(y), float(z), color_bgr))
    
    # Optionally show floaters in red
    if show_floaters and len(floater_ids) > 0:
        floater_mask = np.isin(labeled, floater_ids)
        floater_coords = np.stack(np.where(floater_mask), axis=-1)
        
        if len(floater_coords) > 0:
            # Subsample floaters
            max_floater_points = max_points_per_object
            if len(floater_coords) > max_floater_points:
                indices = np.random.choice(len(floater_coords), max_floater_points, replace=False)
                floater_coords = floater_coords[indices]
            
            # Project floaters
            floater_coords_tensor = torch.from_numpy(floater_coords).float().to(device)
            floater_coords_norm = (floater_coords_tensor / reso.float()) * 2 - 1
            floater_coords_world = floater_coords_norm * radius.unsqueeze(0) + center.unsqueeze(0)
            
            floater_coords_homo = torch.cat([
                floater_coords_world,
                torch.ones(len(floater_coords_world), 1, device=device)
            ], dim=1)
            
            floater_cam = (w2c @ floater_coords_homo.T).T
            
            x_img = (floater_cam[:, 0] / floater_cam[:, 2]) * fx + cx
            y_img = (floater_cam[:, 1] / floater_cam[:, 2]) * fy + cy
            z_depth = floater_cam[:, 2]
            
            valid = (z_depth > 0) & (x_img >= 0) & (x_img < W) & (y_img >= 0) & (y_img < H)
            
            x_valid = x_img[valid].long().cpu().numpy()
            y_valid = y_img[valid].long().cpu().numpy()
            z_valid = z_depth[valid].cpu().numpy()
            
            # Add floaters to points list (red color in BGR)
            for x, y, z in zip(x_valid, y_valid, z_valid):
                if 0 <= x < W and 0 <= y < H:
                    all_points.append((int(x), int(y), float(z), (0, 0, 255)))  # Red in BGR
    
    # Now render all points using depth buffer (back-to-front for proper occlusion)
    # Sort by depth (farthest first)
    all_points.sort(key=lambda p: -p[2])  # Negative for descending order
    
    # Render each point with depth test
    for x, y, depth, color_bgr in all_points:
        # Only draw if this is the closest point at this pixel
        if depth < depth_buffer[y, x]:
            depth_buffer[y, x] = depth
            cv2.circle(vis_image, (x, y), 2, color_bgr, -1)
    
    # Blend with original
    vis_image_float = vis_image.astype(np.float32) / 255.0
    result = (1 - alpha) * rgb_image + alpha * vis_image_float
    result = np.clip(result, 0, 1)
    
    return result


def create_main_object_voxel_overlay(rgb_image, grid, fdr_results, camera, 
                                      max_points=50000, alpha=0.7):
    """
    Overlay main object voxels onto rendered image to show spatial distribution
    
    Shows green dots for each main object voxel, allowing visual verification
    of which voxels belong to the largest connected component.
    
    Args:
        rgb_image: Rendered RGB image [H, W, 3] in range [0, 1]
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        camera: Camera object
        max_points: Maximum number of voxels to visualize (for performance)
        alpha: Opacity of voxel markers (default 0.7)
        
    Returns:
        overlay_image: RGB image with main object voxels overlaid [H, W, 3]
    """
    if rgb_image is None or fdr_results is None:
        return rgb_image
    
    # Get main object voxels
    main_mask, main_coords = create_main_object_overlay_grid(grid, fdr_results)
    
    if main_coords is None or len(main_coords) == 0:
        return rgb_image
    
    # Subsample if too many voxels
    if len(main_coords) > max_points:
        indices = np.random.choice(len(main_coords), max_points, replace=False)
        main_coords = main_coords[indices]
        print(f"  Subsampled main object voxels: {max_points}/{len(main_mask.nonzero()[0])} for visualization")
    
    # Project main object to 2D
    device = grid.links.device
    reso = torch.tensor(grid.links.shape, device=device)
    radius = grid.radius.to(device) if hasattr(grid.radius, 'to') else grid.radius
    center = grid.center.to(device) if hasattr(grid.center, 'to') else grid.center
    
    # Convert to world space
    main_coords_norm = (main_coords.float().to(device) / reso.float()) * 2 - 1
    main_coords_world = main_coords_norm * radius.unsqueeze(0) + center.unsqueeze(0)
    
    # Project to camera
    c2w = camera.c2w.to(device) if hasattr(camera.c2w, 'to') else camera.c2w
    w2c = torch.inverse(c2w)
    
    main_coords_homo = torch.cat([
        main_coords_world,
        torch.ones(len(main_coords_world), 1, device=device)
    ], dim=1)
    
    main_cam = (w2c @ main_coords_homo.T).T
    
    # Project to image
    fx = float(camera.fx) if hasattr(camera.fx, 'item') else camera.fx
    fy = float(camera.fy) if camera.fy is not None else fx
    cx = camera.width * 0.5
    cy = camera.height * 0.5
    
    x_img = (main_cam[:, 0] / main_cam[:, 2]) * fx + cx
    y_img = (main_cam[:, 1] / main_cam[:, 2]) * fy + cy
    z_depth = main_cam[:, 2]
    
    # Filter valid points (in front of camera and within image bounds)
    valid = (z_depth > 0) & (x_img >= 0) & (x_img < camera.width) & \
            (y_img >= 0) & (y_img < camera.height)
    
    x_valid = x_img[valid].long().cpu().numpy()
    y_valid = y_img[valid].long().cpu().numpy()
    
    H, W = camera.height, camera.width
    
    # Create overlay with green dots for main object voxels
    result = rgb_image.copy()
    voxel_color = np.array([0.0, 1.0, 0.3])  # Bright green
    
    # Draw small circles/dots for each voxel
    vis_image = (result * 255).astype(np.uint8)
    
    for x, y in zip(x_valid, y_valid):
        if 0 <= x < W and 0 <= y < H:
            # Draw a small filled circle (2 pixel radius)
            cv2.circle(vis_image, (int(x), int(y)), 2, 
                      (int(voxel_color[2]*255), int(voxel_color[1]*255), int(voxel_color[0]*255)), 
                      -1)  # -1 = filled
    
    # Blend with original image
    vis_image_float = vis_image.astype(np.float32) / 255.0
    result = (1 - alpha) * rgb_image + alpha * vis_image_float
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


def log_floater_visualizations_to_tensorboard(
    summary_writer,
    grid,
    fdr_results,
    global_step,
    cameras=None,
    rendered_images=None,
    gt_images=None,
    max_render_views=3,
    log_density_renders=True,
    max_voxels_per_object=200000
):
    """
    Log floater visualizations to TensorBoard
    
    Args:
        summary_writer: TensorBoard SummaryWriter
        grid: SparseGrid instance
        fdr_results: Results from compute_FDR
        global_step: Training iteration for logging
        cameras: Optional list of camera objects for projection overlays
        rendered_images: Optional list of rendered RGB images [H, W, 3]
        gt_images: Optional list of ground truth RGB images [H, W, 3]
        max_render_views: Maximum number of rendered views to overlay
        log_density_renders: If True, log density field renders (default: True)
        max_voxels_per_object: Max voxels to render per object (default: 200000)
    """
    if 'FDR_floater_mask_3d' not in fdr_results:
        print("  Warning: No floater mask available for visualization")
        return
    
    # Create floater projection overlays on rendered views
    if cameras is not None and rendered_images is not None:
        n_views = min(len(cameras), len(rendered_images), max_render_views)
        
        for i in range(n_views):
            camera = cameras[i]
            rgb_image = rendered_images[i]
            gt_image = gt_images[i] if gt_images is not None and i < len(gt_images) else None
            
            try:
                if gt_image is not None:
                    summary_writer.add_image(
                        f'floaters/gt_view_{i}', 
                        gt_image.transpose(2, 0, 1),
                        global_step=global_step,
                        dataformats='CHW'
                    )
                
                summary_writer.add_image(
                    f'floaters/render_view_{i}', 
                    rgb_image.transpose(2, 0, 1),
                    global_step=global_step,
                    dataformats='CHW'
                )
                
                if log_density_renders:
                    density_render = render_density_from_camera(grid, camera, colormap='hot')
                    if density_render is not None:
                        summary_writer.add_image(
                            f'floaters/density_view_{i}', 
                            density_render.transpose(2, 0, 1),
                            global_step=global_step,
                            dataformats='CHW'
                        )
            except Exception as e:
                print(f"  Warning: Failed to log images for view {i}: {e}")
            
            try:
                floater_heatmap = project_floaters_to_view(grid, fdr_results, camera)
                
                if floater_heatmap is not None:
                    overlay_image = create_floater_overlay_on_render(rgb_image, floater_heatmap)
                    summary_writer.add_image(
                        f'floaters/overlay_view_{i}', 
                        overlay_image.transpose(2, 0, 1),
                        global_step=global_step,
                        dataformats='CHW'
                    )
                    
                    heatmap_colored = cv2.applyColorMap(
                        (floater_heatmap / max(floater_heatmap.max(), 1e-6) * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    summary_writer.add_image(
                        f'floaters/heatmap_view_{i}',
                        heatmap_colored.transpose(2, 0, 1),
                        global_step=global_step,
                        dataformats='CHW'
                    )
            except Exception as e:
                print(f"  Warning: Failed to create floater overlay for view {i}: {e}")
            
            try:
                main_voxel_overlay = create_main_object_voxel_overlay(
                    rgb_image, grid, fdr_results, camera, 
                    max_points=max_voxels_per_object, alpha=0.7
                )
                
                if main_voxel_overlay is not None:
                    summary_writer.add_image(
                        f'floaters/main_object_voxels_view_{i}',
                        main_voxel_overlay.transpose(2, 0, 1),
                        global_step=global_step,
                        dataformats='CHW'
                    )
            except Exception as e:
                print(f"  Warning: Failed to create main object overlay for view {i}: {e}")
            
            try:
                multi_obj_overlay = create_multi_object_voxel_overlay(
                    rgb_image, grid, fdr_results, camera, 
                    max_points_per_object=max_voxels_per_object, alpha=0.7, 
                    show_floaters=True, min_viz_size=500  # Match FDR threshold
                )
                
                if multi_obj_overlay is not None:
                    summary_writer.add_image(
                        f'floaters/multi_object_colored_view_{i}',
                        multi_obj_overlay.transpose(2, 0, 1),
                        global_step=global_step,
                        dataformats='CHW'
                    )
            except Exception as e:
                print(f"  Warning: Failed to create multi-object overlay for view {i}: {e}")
    
    # Create detailed summary with per-object breakdown
    main_ids = fdr_results.get('FDR_main_component_ids', [])
    labeled = fdr_results.get('FDR_floater_mask_3d', None)
    
    summary_lines = [
        f"FDR: {fdr_results['FDR']:.2%}",
        f"Method: {fdr_results.get('FDR_detection_method', 'unknown')}",
        f"Main Objects: {fdr_results.get('FDR_num_main_objects', 1)}",
        f"Floaters: {fdr_results['FDR_num_floaters']}",
        f"Main Volume: {fdr_results['FDR_main_volume']:,} voxels ({fdr_results['FDR_main_volume']/fdr_results['FDR_total_volume']*100:.1f}%)",
        f"Floater Volume: {fdr_results['FDR_floater_volume']:,} voxels ({fdr_results['FDR']:.2%})",
        ""
    ]
    
    # Add per-object size breakdown
    if labeled is not None and len(main_ids) > 0:
        summary_lines.append("Main Object Sizes:")
        from scipy import ndimage
        for i, main_id in enumerate(main_ids[:8]):  # Top 8 objects
            obj_size = np.sum(labeled == main_id)
            percent = (obj_size / fdr_results['FDR_total_volume']) * 100
            summary_lines.append(f"  Object #{i+1}: {obj_size:,} voxels ({percent:.1f}%)")
    
    summary_text = "\n".join(summary_lines)
    
    summary_writer.add_text(
        'floaters/summary',
        summary_text,
        global_step=global_step
    )
    
    if cameras is not None and rendered_images is not None:
        n_views = min(len(cameras), len(rendered_images), max_render_views)
        imgs_per_view = 6 - (0 if gt_images else 1)
        n_vis = n_views * imgs_per_view
        print(f"  Logged {n_vis} floater visualizations to TensorBoard")

