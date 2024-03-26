"""!@file cnn_module.py

@brief This file contains the DDPM model class and the DDPM schedules function.

@details This file contains the DDPM model class and the DDPM schedules function.
The DDPM model class is used to train the model and make predictions.
The DDPM schedules function is used to return pre-computed schedules
for DDPM sampling with a linear noise schedule.

@author Created by T.Breitburd on 23/03/2024
"""

# Importing the necessary libraries
from typing import Dict, Tuple
import torch
import torch.nn as nn


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """!@brief Returns pre-computed schedules for DDPM sampling with a linear noise schedule.

    @param beta1: float
    @param beta2: float
    @param T: int
    @return Dict[str, torch.Tensor]
    """

    # Check that beta1 and beta2 are in the correct range
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Create the linear noise schedule
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}


# ----------------------------------------------
# Define the actual diffusion model
# ----------------------------------------------


class DDPM(nn.Module):
    """!@brief The DDPM model class.

    @param gt: torch.nn.Module
    @param betas: Tuple[float, float]
    @param n_T: int
    @param criterion: torch.nn.Module

    @return None
    """

    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        # gt is the function that predicts the "error term" from the z_t, here we use a CNN
        # (see the cnn_module.py file for the CNN class definition)
        self.gt = gt

        # Get the noise schedule
        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!@brief The forward method of the DDPM model class, which follows
        Algorithm 18.1 in Prince.

        @param x: torch.Tensor

        @return torch.Tensor
        """

        # Sample a random time step, and compute the added noise for this step
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        # Degrade the image, following the algorithm
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """!@brief The sample method of the DDPM model class, which follows
        Algorithm 18.2 in Prince.

        @param n_sample: int
        @param size: Tuple
        @param device

        @return torch.Tensor
        """

        # Initialize the image as a full Gaussian image
        _one = torch.ones(n_sample, device=device)  # To reshape the i/n_T into a tensor
        # of the right shape
        z_t = torch.randn(n_sample, *size, device=device)

        # Save the degraded image for later
        degraded = z_t.clone()

        # Follow the algorithm 18.2 in Prince
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return degraded, z_t
