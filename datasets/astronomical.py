"""
Dataset loader for astronomical simulations (IllustrisTNG, EAGLE, etc.)
Supports loading coordinates of Gas (PartType0), Dark Matter (PartType1), and Stars (PartType4)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from .utils import NoiseSchedulerDDPM


# Mapping of particle types
PARTICLE_TYPES = {
    'gas': 'PartType0',
    'dark_matter': 'PartType1',
    'dm': 'PartType1',
    'stars': 'PartType4',
    'stellar': 'PartType4'
}


class AstronomicalDataset(Dataset):
    """
    Dataset for astronomical particle data (gas, dark matter, stars)

    Args:
        hdf5_path: Path to HDF5 file
        particle_type: Type of particles to load ('gas', 'dark_matter'/'dm', 'stars'/'stellar')
        n_points: Number of points to sample per subhalo
        normalize: Whether to normalize coordinates
        center: Whether to center coordinates at origin
        random_rotation: Whether to apply random rotation augmentation
        random_subsample: Whether to randomly subsample points during loading
    """

    def __init__(self, hdf5_path, particle_type='stars', n_points=2048,
                 normalize=True, center=True, random_rotation=False,
                 random_subsample=True):
        super().__init__()

        self.hdf5_path = hdf5_path
        self.n_points = n_points
        self.normalize = normalize
        self.center = center
        self.random_rotation = random_rotation
        self.random_subsample = random_subsample

        # Map particle type string to HDF5 group name
        if particle_type.lower() not in PARTICLE_TYPES:
            raise ValueError(f"Unknown particle type: {particle_type}. "
                           f"Choose from: {list(PARTICLE_TYPES.keys())}")
        self.particle_type = PARTICLE_TYPES[particle_type.lower()]

        # Load dataset metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata about available subhalos and particle counts"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get all subhalo keys
            self.subhalo_keys = [k for k in f.keys() if k.startswith('Subhalo_')]

            # Filter subhalos that have the requested particle type with enough particles
            valid_subhalos = []
            particle_counts = []

            for key in self.subhalo_keys:
                if self.particle_type in f[key]:
                    coords = f[key][self.particle_type]['Coordinates']
                    n_particles = coords.shape[0]
                    if n_particles >= self.n_points:
                        valid_subhalos.append(key)
                        particle_counts.append(n_particles)

            self.subhalo_keys = valid_subhalos
            self.particle_counts = np.array(particle_counts)

            print(f"Found {len(self.subhalo_keys)} subhalos with {self.particle_type}")
            print(f"Particle counts - min: {self.particle_counts.min()}, "
                  f"max: {self.particle_counts.max()}, "
                  f"mean: {self.particle_counts.mean():.0f}")

    def __len__(self):
        return len(self.subhalo_keys)

    def _load_coordinates(self, idx):
        """Load coordinates for a specific subhalo"""
        with h5py.File(self.hdf5_path, 'r') as f:
            subhalo_key = self.subhalo_keys[idx]
            coords = f[subhalo_key][self.particle_type]['Coordinates'][:]
        return coords

    def _process_pointcloud(self, coords):
        """Process point cloud: subsample, center, normalize, rotate"""
        # Subsample to desired number of points
        n_available = coords.shape[0]

        if self.random_subsample:
            # Random sampling
            indices = np.random.choice(n_available, self.n_points, replace=False)
        else:
            # Take first n_points
            indices = np.arange(self.n_points)

        pc = coords[indices].astype(np.float32)

        # Center at origin
        if self.center:
            centroid = pc.mean(axis=0)
            pc = pc - centroid

        # Normalize to unit sphere
        if self.normalize:
            max_dist = np.max(np.linalg.norm(pc, axis=1))
            if max_dist > 0:
                pc = pc / max_dist

        # Random rotation augmentation
        if self.random_rotation:
            # Random rotation matrix (SO(3))
            theta = np.random.uniform(0, 2 * np.pi, 3)

            # Rotation around Z
            Rz = np.array([
                [np.cos(theta[2]), -np.sin(theta[2]), 0],
                [np.sin(theta[2]), np.cos(theta[2]), 0],
                [0, 0, 1]
            ])

            # Rotation around Y
            Ry = np.array([
                [np.cos(theta[1]), 0, np.sin(theta[1])],
                [0, 1, 0],
                [-np.sin(theta[1]), 0, np.cos(theta[1])]
            ])

            # Rotation around X
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta[0]), -np.sin(theta[0])],
                [0, np.sin(theta[0]), np.cos(theta[0])]
            ])

            R = Rz @ Ry @ Rx
            pc = pc @ R.T

        return pc

    def __getitem__(self, idx):
        # Load coordinates
        coords = self._load_coordinates(idx)

        # Process point cloud
        pc = self._process_pointcloud(coords)

        return torch.from_numpy(pc)


class AstronomicalDatasetSparseNoisy(AstronomicalDataset):
    """
    Astronomical dataset with sparse tensor conversion and DDPM noise
    """

    def __init__(self, hdf5_path, particle_type='stars', n_points=2048,
                 normalize=True, center=True, random_rotation=False,
                 random_subsample=True, pres=1e-5):
        super().__init__(hdf5_path, particle_type, n_points, normalize,
                        center, random_rotation, random_subsample)

        self.pres = pres
        self.noise_scheduler = None

    def set_noise_params(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear'):
        """Set DDPM noise parameters"""
        self.noise_scheduler = NoiseSchedulerDDPM(
            beta_min=beta_min,
            beta_max=beta_max,
            n_steps=n_steps,
            mode=mode
        )

    def __getitem__(self, idx):
        # Get base point cloud
        pc = super().__getitem__(idx)

        # Apply DDPM noise
        if self.noise_scheduler is None:
            raise ValueError("Must call set_noise_params() before using dataset")

        x0 = pc
        noisy_pc, t, noise = self.noise_scheduler(x0)

        # Convert to numpy for sparse_quantize
        pts = noisy_pc.numpy()
        noise_np = noise.numpy()

        # Remove minimum to ensure positive coordinates for voxelization
        coords = pts - np.min(pts, axis=0, keepdims=True)

        # Quantize coordinates and get indices
        coords, indices = sparse_quantize(coords, self.pres, return_index=True)

        # Convert back to torch tensors
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(pts[indices], dtype=torch.float)
        noise_feats = torch.tensor(noise_np[indices], dtype=torch.float)

        # Create sparse tensors
        sparse_input = SparseTensor(coords=coords, feats=feats)
        sparse_noise = SparseTensor(coords=coords, feats=noise_feats)
        t = torch.tensor(t)

        return {
            'input': sparse_input,
            't': t,
            'noise': sparse_noise
        }


def get_datasets(hdf5_path, particle_type='stars', n_points=2048,
                beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear',
                pres=1e-5, train_split=0.8):
    """
    Create train and validation datasets

    Args:
        hdf5_path: Path to HDF5 file
        particle_type: 'gas', 'dark_matter'/'dm', 'stars'/'stellar'
        n_points: Number of points per sample
        beta_min, beta_max, n_steps, mode: DDPM noise parameters
        pres: Voxel resolution
        train_split: Fraction of data for training
    """
    # Create full dataset to get total length
    full_dataset = AstronomicalDataset(
        hdf5_path=hdf5_path,
        particle_type=particle_type,
        n_points=n_points,
        normalize=True,
        center=True,
        random_rotation=False,
        random_subsample=True
    )

    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    print(f"Dataset splits - Train: {train_size}, Val: {val_size}")

    # Create train dataset
    tr_dataset = AstronomicalDatasetSparseNoisy(
        hdf5_path=hdf5_path,
        particle_type=particle_type,
        n_points=n_points,
        normalize=True,
        center=True,
        random_rotation=True,  # Augmentation for training
        random_subsample=True,
        pres=pres
    )

    # Create validation dataset
    te_dataset = AstronomicalDatasetSparseNoisy(
        hdf5_path=hdf5_path,
        particle_type=particle_type,
        n_points=n_points,
        normalize=True,
        center=True,
        random_rotation=False,  # No augmentation for validation
        random_subsample=True,
        pres=pres
    )

    # Set noise parameters
    tr_dataset.set_noise_params(beta_min, beta_max, n_steps, mode)
    te_dataset.set_noise_params(beta_min, beta_max, n_steps, mode)

    # Manually split by using subset indices
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))

    # Store indices for filtering
    tr_dataset.indices = train_indices
    te_dataset.indices = val_indices

    # Override __len__ and __getitem__ to use indices
    original_getitem_tr = tr_dataset.__getitem__
    original_getitem_te = te_dataset.__getitem__

    tr_dataset.__len__ = lambda: len(train_indices)
    te_dataset.__len__ = lambda: len(val_indices)

    tr_dataset.__getitem__ = lambda idx: original_getitem_tr(train_indices[idx])
    te_dataset.__getitem__ = lambda idx: original_getitem_te(val_indices[idx])

    return tr_dataset, te_dataset


def get_dataloaders(hdf5_path, particle_type='stars', n_points=2048,
                   beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear',
                   pres=1e-5, batch_size=32, num_workers=4, train_split=0.8):
    """
    Create train and validation dataloaders

    Args:
        hdf5_path: Path to HDF5 file
        particle_type: 'gas', 'dark_matter'/'dm', 'stars'/'stellar'
        n_points: Number of points per sample
        beta_min, beta_max, n_steps, mode: DDPM noise parameters
        pres: Voxel resolution
        batch_size: Batch size
        num_workers: Number of dataloader workers
        train_split: Fraction for training
    """
    tr_dataset, te_dataset = get_datasets(
        hdf5_path, particle_type, n_points,
        beta_min, beta_max, n_steps, mode, pres, train_split
    )

    tr_dl = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=sparse_collate_fn
    )

    te_dl = DataLoader(
        te_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sparse_collate_fn
    )

    return tr_dl, te_dl
