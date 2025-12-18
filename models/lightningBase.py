import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from abc import ABC, abstractmethod
import wandb
import matplotlib.pyplot as plt
from utils.visualization import create_2d_projections
from utils.schedulers import DDPMSparseSchedulerGPU

class Task(ABC):
    @abstractmethod
    def prep_data(self, batch):
        pass
    @abstractmethod
    def loss_fn(self, pred, target):
        pass

class SparseGeneration(Task):
    def prep_data(self, batch):
        noisy_data, t, noise = batch['input'], batch['t'], batch['noise']
        inp = (noisy_data, t)
        return inp, noise.F
    def loss_fn(self, preds, target):
        return F.mse_loss(preds, target)

class DiffusionBase(L.LightningModule):

    def __init__(self, model, task=SparseGeneration(), lr=0.0002,
                 sample_every_n_epochs=5, n_samples=4, n_points=2048,
                 beta_min=0.0001, beta_max=0.02, n_steps=1000,
                 use_density_coloring=True, density_bins=50):
        super().__init__()
        self.model = model
        self.task = task
        self.learning_rate = lr

        # Sampling configuration
        self.sample_every_n_epochs = sample_every_n_epochs
        self.n_samples = n_samples
        self.n_points = n_points
        self.use_density_coloring = use_density_coloring
        self.density_bins = density_bins

        # Initialize scheduler for sampling
        self.sampler = DDPMSparseSchedulerGPU(
            beta_min=beta_min,
            beta_max=beta_max,
            n_steps=n_steps,
            pres=model.pres if hasattr(model, 'pres') else 1e-5
        )
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # get data from the batch
        inp, target = self.task.prep_data(batch)

        # activate the network for noise prediction
        preds = self(inp)

        # calculate the loss
        loss = self.task.loss_fn(preds, target)

        self.log('train_loss', loss, batch_size=self.tr_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = self.task.prep_data(batch)
        preds = self(inp)
        loss = self.task.loss_fn(preds, target)
        self.log('val_loss', loss, batch_size=self.vl_batch_size)
        return loss

    def on_validation_epoch_end(self):
        # Only generate samples every N epochs to save time
        if (self.current_epoch + 1) % self.sample_every_n_epochs == 0:
            self._generate_and_log_samples()

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)

        # Create a dummy scheduler (we will update `total_steps` later)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=1)

        # Return optimizer and scheduler (scheduler will be updated in `on_fit_start`)
        return [optimizer], [{'scheduler': self.lr_scheduler, 'interval': 'step'}]

    # Setting the OneCycle scheduler correct number of steps at the start of the fit loop, where the dataloaders are available.
    def on_train_start(self):
        # Access the dataloader and calculate total steps
        train_loader = self.trainer.train_dataloader  # Access the dataloader from the trainer
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        # Update the scheduler's `total_steps` dynamically
        self.lr_scheduler.total_steps = total_steps

        # Read the batch size for logging
        self.tr_batch_size = self.trainer.train_dataloader.batch_size

    def on_validation_start(self):
        val_loader = self.trainer.val_dataloaders
        if val_loader:
            self.vl_batch_size = val_loader.batch_size

    def _generate_and_log_samples(self):
        """Generate point cloud samples and log 2D projections to wandb"""
        # Set model to eval mode
        self.model.eval()

        with torch.no_grad():
            # Generate samples
            samples = self.sampler.sample(
                self.model,
                bs=self.n_samples,
                n_points=self.n_points
            )

            # samples is a tensor of shape (n_samples, n_points, 3)
            # Create separate 2D projection for each sample
            figures = create_2d_projections(
                samples,
                figsize=(15, 5),
                point_size=2,
                cmap='viridis',
                use_density=self.use_density_coloring,
                density_bins=self.density_bins
            )

            # Log to wandb if logger is available
            if self.logger is not None and isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                # Log each sample as a separate image under validation section
                log_dict = {}
                for i, fig in enumerate(figures):
                    log_dict[f"validation/sample_{i+1}"] = wandb.Image(fig)

                self.logger.experiment.log(log_dict)

            # Close all figures to free memory
            for fig in figures:
                plt.close(fig)

        # Set model back to train mode
        self.model.train()

