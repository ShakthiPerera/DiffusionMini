import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from functions.losses import default_loss, iso_loss


class DDPM(pl.LightningModule):
    """
    Plain vanilla DDPM module.

    Summary
    -------
    This module establishes a plain vanilla DDPM variant.
    It is basically a container and wrapper for an
    epsilon model and for the scheduling parameters.
    The class provides methods implementing the forward
    and reverse diffusion processes, respectively.
    Also, the stochastic loss can be computed for training.

    Parameters
    ----------
    eps_model : PyTorch module
        Trainable noise-predicting model.
    betas : array-like
        Beta parameter schedule.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Optimizer learning rate.

    """

    def __init__(
        self, eps_model, betas, criterion="mse", lr=1e-04, loss_type="default"
    ):
        super().__init__()

        # set trainable epsilon model
        self.eps_model = eps_model
        self.loss_type = loss_type

        # set loss function criterion
        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif criterion == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError("Criterion could not be determined")

        # set initial learning rate
        self.lr = abs(lr)

        # to save losses
        self.train_losses = []

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # set scheduling parameters
        betas = torch.as_tensor(betas).view(
            -1
        )  # note that betas[0] corresponds to t = 1.0

        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError("Invalid beta values encountered")

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(
            betas_tilde, pad=(1, 0), value=0.0
        )  # ensure betas_tilde[0] = 0.0

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("betas_tilde", betas_tilde)

    @property
    def num_steps(self):
        """Get the total number of time steps."""
        return len(self.betas)

    def forward(self, x, t):
        """Run the noise-predicting model."""
        return self.eps_model(x, t)

    def diffuse_step(self, x, tidx):
        """Simulate single forward process step."""
        beta = self.betas[tidx]
        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps
        return x_noisy

    def diffuse_all_steps_till_time(self, x0, time):
        """Simulate and return all forward process steps."""
        x_noisy = torch.zeros(time + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(time):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse_all_steps(self, x0):
        """Simulate and return all forward process steps."""
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse(self, x0, tids, return_eps=False):
        """Simulate multiple forward steps at once."""
        alpha_bar = self.alphas_bar[tids]
        eps = torch.randn_like(x0)

        missing_shape = [1] * (eps.ndim - alpha_bar.ndim)
        alpha_bar = alpha_bar.view(*alpha_bar.shape, *missing_shape)

        x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

        if return_eps:
            return x_noisy, eps
        else:
            return x_noisy

    def denoise_step(self, x, tids, random_sample=False):
        """Perform single reverse process step."""
        # set up time variables
        tids = torch.as_tensor(tids, device=x.device).view(
            -1, 1
        )  # ensure (batch_size>=1, 1)-shaped tensor
        ts = tids.to(x.dtype) + 1  # note that tidx = 0 corresponds to t = 1.0

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x, ts)

        # compute mean
        p = 1 / self.alphas[tids].sqrt()
        q = self.betas[tids] / (1 - self.alphas_bar[tids]).sqrt()

        missing_shape = [1] * (eps_pred.ndim - ts.ndim)
        p = p.view(*p.shape, *missing_shape)
        q = q.view(*q.shape, *missing_shape)

        x_denoised_mean = p * (x - q * eps_pred)

        # retrieve variance
        x_denoised_var = self.betas_tilde[tids]
        # x_denoised_var = self.betas[tids]

        # generate random sample
        if random_sample:
            eps = torch.randn_like(x_denoised_mean)
            x_denoised = x_denoised_mean + x_denoised_var.sqrt() * eps

        if random_sample:
            return x_denoised, eps_pred
        else:
            return x_denoised_mean, x_denoised_var, eps_pred

    @torch.no_grad()
    def denoise_all_steps(self, xT):
        """Perform and return all reverse process steps."""
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)
        x_eps = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)

        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):
            # generate random sample
            if tidx > 0:
                x_denoised[idx + 1], x_eps[idx] = self.denoise_step(
                    x_denoised[idx], tidx, random_sample=True
                )
            # take the mean in the last step
            else:
                x_denoised[idx + 1], _, x_eps[idx] = self.denoise_step(
                    x_denoised[idx], tidx, random_sample=False
                )

        return x_denoised, x_eps

    @torch.no_grad()
    def generate(self, sample_shape, num_samples=1):
        """Generate random samples through the reverse process."""
        x_denoised = torch.randn(
            num_samples, *sample_shape, device=self.device
        )  # Lightning modules have a device attribute
        isotropy = []
        for tidx in reversed(range(self.num_steps)):
            # generate random sample
            if tidx > 0:
                x_denoised, _ = self.denoise_step(x_denoised, tidx, random_sample=True)
                # iso = self.isotropy(x_denoised)
                # isotropy.append(iso)
            # take the mean in the last step
            else:
                x_denoised, _, _ = self.denoise_step(
                    x_denoised, tidx, random_sample=False
                )
                # iso = self.isotropy(x_denoised)
                # isotropy.append(iso)

        return x_denoised

    def loss(self, x):
        """Compute stochastic loss."""
        #  draw random time steps
        tids = torch.randint(0, self.num_steps, size=(x.shape[0], 1), device=x.device)

        ts = tids.to(x.dtype) + 1  # note that tidx = 0 corresponds to t = 1.0

        # perform forward process steps
        x_noisy, eps = self.diffuse(x, tids, return_eps=True)

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x_noisy, ts)

        if self.loss_type == "default":
            loss, norm_loss = default_loss(x, eps_pred, eps, self.criterion)
        elif self.loss_type == "iso":
            loss, norm_loss = iso_loss(x, eps_pred, self.criterion)

        return loss, loss, norm_loss

    def train_step(self, x_batch):
        self.optimizer.zero_grad()
        loss, simple_diff_loss, norm_loss = self.loss(x_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), simple_diff_loss.item(), norm_loss.item()

    def get_eps_pred_list(self):
        return self.eps_pred_list

    def get_eps_list(self):
        return self.eps_list

    def get_iso_difference_list(self):
        return self.iso_difference_list

    def validate(self, val_loader):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch in val_loader:
                val_loss, _, _ = self.loss(torch.stack(x_val_batch)).item()
                val_loss += val_loss
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss
