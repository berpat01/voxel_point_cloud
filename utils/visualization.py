import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import binned_statistic_2d

from matplotlib import colormaps
cmap = colormaps['jet']

def pc_to_o3d(pc): # point cloud as np.array or torch.tensor
    "turn a point cloud, represented as a np.array or torch.tensor to an [Open3D.geometry.PointCloud](http://www.open3d.org/docs/0.16.0/python_api/open3d.geometry.PointCloud.html)"
    pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pc)
    )
    return pc

def quick_vis_batch(batch, grid=(8, 4), x_offset=2.5, y_offset=2.5):
    
    batch = batch.detach().cpu().clone()
    
    assert len(grid) == 2
    
    if batch.shape[0] <= np.prod(grid): batch = batch[:np.prod(grid)]
            
    x_offset_start = - x_offset * grid[0] // 2
    x_offset_start = x_offset_start + x_offset / 2 if grid[0] % 2 == 0 else x_offset_start
    
    y_offset_start = - y_offset * grid[1] // 2
    y_offset_start = y_offset_start + y_offset / 2 if grid[1] % 2 == 0 else y_offset_start
    
    pcts = []
    
    k=0
    for i in range(grid[0]):
        for j in range(grid[1]):
            
            # get point cloud to cpu
            pc = batch[k]
            
            # translate the point cloud properly
            pc[:, 0] += x_offset_start + i * x_offset
            pc[:, 1] += y_offset_start + j * y_offset
            
            # turn in into an open3d point cloud
            pct = pc_to_o3d(pc)
            
            # append it to the pcts list
            pcts.append(pct)
            
            # incriment k
            k+=1
            
        if k > batch.shape[0]-1: break
            
    o3d.visualization.draw_geometries(pcts)

def vis_pc_sphere(pc, radius=0.1, resolution=30, color=None):
    
    pc = pc.detach().cpu().squeeze().numpy()

    if color is None:
        # sample a colormap based on the z-direction of the point cloud
        color_val = pc[:, -1]
        # normalize the color values in range [0, 1]
        color_val = (color_val - color_val.min()) / (color_val.max() - color_val.min())
        # get the color 
        color = cmap(color_val)[:, :3]

    
    # create a mesh that will contain all the spheres
    mesh = o3d.geometry.TriangleMesh()
    # create a sphere for each point in the point cloud
    for i, p in enumerate(pc):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution).translate(p)
        sphere.paint_uniform_color(color[i])
        mesh += sphere

    o3d.visualization.draw_geometries([mesh])

def compute_point_density(x, y, bins=50):
    """
    Compute local point density for 2D projection using binning.

    Args:
        x: X coordinates
        y: Y coordinates
        bins: Number of bins for histogram (controls resolution)

    Returns:
        density: Density value for each point
    """
    # Create 2D histogram to count points in each bin
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Find which bin each point belongs to
    x_bin_idx = np.searchsorted(xedges, x, side='right') - 1
    y_bin_idx = np.searchsorted(yedges, y, side='right') - 1

    # Clamp to valid range (edge cases)
    x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)

    # Get density for each point from the histogram
    density = H[x_bin_idx, y_bin_idx]

    # Apply log scale for better visualization (dense areas don't dominate)
    density = np.log1p(density)  # log(1 + density) to avoid log(0)

    return density

def create_2d_projections(point_clouds, figsize=(15, 5), point_size=1, cmap='viridis', use_density=False, density_bins=50):
    """
    Create 2D projections (XY, XZ, YZ) of a batch of point clouds for wandb logging.

    Args:
        point_clouds: Tensor of shape (B, N, 3) or (N, 3) containing point cloud(s)
        figsize: Figure size for matplotlib
        point_size: Size of points in scatter plot
        cmap: Colormap for coloring points
        use_density: If True, color by local point density. If False, color by 3rd dimension coordinate
        density_bins: Number of bins for density computation (higher = finer resolution)

    Returns:
        List of matplotlib figures, one per point cloud in batch
    """
    # Convert to numpy and handle batching
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.detach().cpu().numpy()

    # Add batch dimension if single point cloud
    if point_clouds.ndim == 2:
        point_clouds = point_clouds[np.newaxis, ...]

    figures = []

    for pc in point_clouds:
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # XY projection (top view)
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        if use_density:
            # Color by local point density
            density_xy = compute_point_density(x, y, bins=density_bins)
            density_xz = compute_point_density(x, z, bins=density_bins)
            density_yz = compute_point_density(y, z, bins=density_bins)

            # XY view - color by density
            im0 = axes[0].scatter(x, y, c=density_xy, cmap=cmap, s=point_size, alpha=0.6)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].set_title('XY Projection (Top View)')
            axes[0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0], label='log(Density)')

            # XZ view - color by density
            im1 = axes[1].scatter(x, z, c=density_xz, cmap=cmap, s=point_size, alpha=0.6)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Z')
            axes[1].set_title('XZ Projection (Front View)')
            axes[1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1], label='log(Density)')

            # YZ view - color by density
            im2 = axes[2].scatter(y, z, c=density_yz, cmap=cmap, s=point_size, alpha=0.6)
            axes[2].set_xlabel('Y')
            axes[2].set_ylabel('Z')
            axes[2].set_title('YZ Projection (Side View)')
            axes[2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[2], label='log(Density)')
        else:
            # Color by coordinate in 3rd dimension
            # XY view - color by Z
            im0 = axes[0].scatter(x, y, c=z, cmap=cmap, s=point_size, alpha=0.6)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].set_title('XY Projection (Top View)')
            axes[0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0], label='Z')

            # XZ view - color by Y
            im1 = axes[1].scatter(x, z, c=y, cmap=cmap, s=point_size, alpha=0.6)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Z')
            axes[1].set_title('XZ Projection (Front View)')
            axes[1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1], label='Y')

            # YZ view - color by X
            im2 = axes[2].scatter(y, z, c=x, cmap=cmap, s=point_size, alpha=0.6)
            axes[2].set_xlabel('Y')
            axes[2].set_ylabel('Z')
            axes[2].set_title('YZ Projection (Side View)')
            axes[2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[2], label='X')

        plt.tight_layout()
        figures.append(fig)

    return figures

def create_2d_projection_grid(point_clouds, grid_size=(4, 4), figsize=(16, 16), point_size=1, cmap='viridis'):
    """
    Create a grid of 2D projections for multiple point clouds.
    Each row shows XY, XZ, YZ views of one point cloud.

    Args:
        point_clouds: Tensor of shape (B, N, 3) containing point clouds
        grid_size: (rows, cols) - number of samples to show
        figsize: Figure size
        point_size: Size of points
        cmap: Colormap

    Returns:
        matplotlib figure
    """
    # Convert to numpy
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.detach().cpu().numpy()

    n_samples = min(grid_size[0], len(point_clouds))

    # Create figure with grid: each sample gets 3 columns (XY, XZ, YZ)
    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)

    # Handle single row case
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        pc = point_clouds[i]
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        # XY projection
        im0 = axes[i, 0].scatter(x, y, c=z, cmap=cmap, s=point_size, alpha=0.6)
        axes[i, 0].set_aspect('equal')
        if i == 0:
            axes[i, 0].set_title('XY (Top)')
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')

        # XZ projection
        im1 = axes[i, 1].scatter(x, z, c=y, cmap=cmap, s=point_size, alpha=0.6)
        axes[i, 1].set_aspect('equal')
        if i == 0:
            axes[i, 1].set_title('XZ (Front)')
        if i == n_samples - 1:
            axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Z')

        # YZ projection
        im2 = axes[i, 2].scatter(y, z, c=x, cmap=cmap, s=point_size, alpha=0.6)
        axes[i, 2].set_aspect('equal')
        if i == 0:
            axes[i, 2].set_title('YZ (Side)')
        if i == n_samples - 1:
            axes[i, 2].set_xlabel('Y')
        axes[i, 2].set_ylabel('Z')

    plt.tight_layout()
    return fig