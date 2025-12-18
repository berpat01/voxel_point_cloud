import argparse
import torch
import lightning as L
from models import *
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from datasets.astronomical import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="SPVD Training Script for Astronomical Data")

    # Model arguments
    parser.add_argument('--version', type=str, choices=['S', 'M', 'L'], default='S',
                        help='Model version: S, M, or L (default: S)')

    # Data arguments
    parser.add_argument('--hdf5_path', type=str, default='just_small_subset.hdf5',
                        help='Path to HDF5 file (default: just_small_subset.hdf5)')
    parser.add_argument('--particle_type', type=str,
                        choices=['gas', 'dark_matter', 'dm', 'stars', 'stellar'],
                        default='stars',
                        help='Particle type to model: gas, dark_matter/dm, stars/stellar (default: stars)')
    parser.add_argument('--n_points', type=int, default=2048,
                        help='Number of points per sample (default: 2048)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training (default: 0.8)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--precision', type=str, choices=['medium', 'high'], default='medium',
                        help='Floating point precision: medium or high (default: medium)')

    # Checkpoint arguments
    parser.add_argument('--ckpt_name', type=str, default='SPVD_astro',
                        help='Checkpoint name (default: SPVD_astro)')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='spvd-astronomical',
                        help='Wandb project name (default: spvd-astronomical)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run wandb in offline mode')

    # Sampling arguments
    parser.add_argument('--sample_every_n_epochs', type=int, default=5,
                        help='Generate samples every N epochs (default: 5)')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to generate (default: 4)')
    parser.add_argument('--use_density_coloring', action='store_true', default=True,
                        help='Use density-based coloring for projections (default: True)')
    parser.add_argument('--no_density_coloring', action='store_false', dest='use_density_coloring',
                        help='Use coordinate-based coloring instead of density')
    parser.add_argument('--density_bins', type=int, default=50,
                        help='Number of bins for density computation (default: 50)')

    # Diffusion arguments
    parser.add_argument('--beta_min', type=float, default=0.0001,
                        help='Minimum beta for noise schedule (default: 0.0001)')
    parser.add_argument('--beta_max', type=float, default=0.02,
                        help='Maximum beta for noise schedule (default: 0.02)')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='Number of diffusion steps (default: 1000)')
    parser.add_argument('--pres', type=float, default=1e-5,
                        help='Voxel resolution (default: 1e-5)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Set floating point precision based on argument
    if args.precision == 'medium':
        torch.set_float32_matmul_precision('medium')
    elif args.precision == 'high':
        torch.set_float32_matmul_precision('high')

    # Load model based on the version argument
    if args.version == 'S':
        m = SPVD_S()
    elif args.version == 'M':
        m = SPVD()
    elif args.version == 'L':
        m = SPVD_L()

    # Initialize model with sampling configuration
    model = DiffusionBase(
        m,
        lr=args.lr,
        sample_every_n_epochs=args.sample_every_n_epochs,
        n_samples=args.n_samples,
        n_points=args.n_points,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        n_steps=args.n_steps,
        use_density_coloring=args.use_density_coloring,
        density_bins=args.density_bins
    )

    # Get dataloaders
    print(f"\n{'='*80}")
    print(f"Loading {args.particle_type} particles from {args.hdf5_path}")
    print(f"{'='*80}\n")

    tr_dl, te_dl = get_dataloaders(
        hdf5_path=args.hdf5_path,
        particle_type=args.particle_type,
        n_points=args.n_points,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        n_steps=args.n_steps,
        mode='linear',
        pres=args.pres,
        batch_size=args.batch_size,
        num_workers=4,
        train_split=args.train_split
    )

    # Set up wandb logger
    run_name = args.wandb_name if args.wandb_name else f"SPVD-{args.version}_{args.particle_type}"

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        offline=args.wandb_offline,
        log_model=False
    )

    # Log hyperparameters
    wandb_logger.experiment.config.update({
        "model_version": args.version,
        "particle_type": args.particle_type,
        "n_points": args.n_points,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "n_steps": args.n_steps,
        "pres": args.pres,
        "train_split": args.train_split,
    })

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=args.ckpt_name + '-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=10.0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    # Train the model
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")

    trainer.fit(model=model, train_dataloaders=tr_dl, val_dataloaders=te_dl)

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
