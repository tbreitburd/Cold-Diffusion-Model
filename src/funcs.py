"""!@file funcs.py

@brief This file contains some general use functions for the project,
for model validation as well as plotting.

@details This file contains the following functions:
- get_fid: to compute the FID score
- get_is: to compute the Inception Score
- get_is_custom: to compute the Inception Score for the custom model
- plot_losses: to plot the losses over the training process
- plot_fid: to plot the FID scores over the training process
- plot_is: to plot the IS scores over the training process
- plot_ddpm_degrade: to plot the DDPM degraded samples for a given timestep

@author Created by T.Breitburd on 23/03/2024
"""

import torch
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Set random seeds
torch.manual_seed(75016)
np.random.seed(75016)


def get_fid(generator, real_data, num_images, device):
    """!@brief Compute the FID score

    @details This function computes the FID score between the real images dataset
    and the generated images from a chosen diffusion model.

    @param generator: the generator model, either DDPM or a custom model
    @param real_data: the real images dataset, mnist
    @param num_images: number of images to sample, int
    @param device: device to use for computations

    @return fid_score: the FID score
    """

    with torch.no_grad():
        # Sample images from the real images dataset
        num_samples = len(real_data)
        idx = random.sample(range(num_samples), num_images)
        real_img = torch.stack([real_data[i][0].clone() for i in idx])

        # Make the image have 3 identical channels
        # so that it can be processed by the FID metric
        real_img = real_img.to(torch.uint8).repeat(1, 3, 1, 1)

        # Sample images from chosen diffusion model (DDPM)
        if generator.__class__.__name__ == "DDPM":
            _, gen_img = generator.sample(num_images, (1, 28, 28), device)
        else:
            # Sample images from chosen diffusion model(custom model)
            _, __, ___, gen_img = generator.sample(
                num_images, real_data, (1, 28, 28), device
            )

        # Make the image have 3 identical channels
        gen_img = gen_img.repeat(1, 3, 1, 1)

        # Put the images in the same device
        gen_img = gen_img.to(real_img.device)

        # Initialize the FID metric and compute the FID score
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_img, real=True)
        fid.update(gen_img, real=False)
        fid_score = fid.compute()

        return fid_score


def get_is(data_source, is_real, num_images, device):
    """!@brief Compute the Inception Score

    @details This function computes the Inception Score for either the real images
    dataset or the generated images from a chosen diffusion model.

    @param data_source: the source of the image, either mnist or a chosen diffusion model
    @param is_real: boolean to indicate if the images are real or generated
    @param num_images: number of images to sample
    @param device: device to use for computations

    @return is_score: the Inception Score
    """

    with torch.no_grad():
        if is_real:
            # Sample images from the real images dataset
            num_samples = len(data_source)
            idx = random.sample(range(num_samples), num_images)
            img = torch.stack([data_source[i][0].clone() for i in idx])
            img = img.repeat(1, 3, 1, 1)

        else:
            # Sample images from the diffusion model (DDPM)
            _, img = data_source.sample(num_images, (1, 28, 28), device)

            # Make the image have 3 identical channels
            img = img.repeat(1, 3, 1, 1)

            img = img.to("cpu")

        # Initialize and compute the IS metric
        is_score = InceptionScore("logits_unbiased", normalize=True)

        is_score.update(img)
        is_score = is_score.compute()

        return is_score


def get_is_custom(generator, data_source, is_real, num_images, device):
    """!@brief Compute the Inception Score for the custom model

    @details This function computes the Inception Score for either the real images
    dataset or the generated images from a chosen custom model.

    @param generator: the custom model, either row or column averaging
    @param data_source: the source of the image, either mnist or a chosen diffusion model
    @param is_real: boolean to indicate if the images are real or generated
    @param num_images: number of images to sample, int
    @param device: device to use for computations

    @return is_score: the Inception Score
    """

    with torch.no_grad():
        if is_real:
            # Sample images from the real images dataset
            num_samples = len(data_source)
            idx = random.sample(range(num_samples), num_images)
            img = torch.stack([data_source[i][0].clone() for i in idx])
            img = img.repeat(1, 3, 1, 1)

        else:
            # Sample images from the custom model
            _, __, ___, img = generator.sample(
                num_images, data_source, (1, 28, 28), device
            )

            # Make the image have 3 identical channels
            img = img.repeat(1, 3, 1, 1)

            img = img.to("cpu")

        # Initialize and compute the IS metric
        is_score = InceptionScore("logits_unbiased", normalize=True)

        is_score.update(img)
        is_score = is_score.compute()

        return is_score


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def plot_losses(losses, avg_losses, num_epochs, model):
    """!@brief Plot the losses over the training process

    @param losses: list of losses, list of floats
    @param avg_losses: list of average losses, list of floats
    @param num_epochs: number of epochs, int
    @param model: used model name, str

    @return None
    """

    # Create the x-axis
    x = np.arange(len(losses)) / (len(losses) // num_epochs)
    x_epoch = x[:: (len(losses) // num_epochs)].copy() + 1

    # Plot the losses
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, losses, color="green", label="Loss")
    plt.plot(
        x_epoch,
        avg_losses,
        linestyle="--",
        marker="+",
        color="black",
        label="Epoch average loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")  # Put the y-axis on a log scale
    plt.title(f"Losses over {num_epochs} epochs for {model}")
    plt.legend()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(
        plot_dir, "losses_for_" + model + "@" + str(num_epochs) + ".png"
    )
    plt.savefig(plot_path)
    plt.close()


def plot_fid(fid_scores, num_epochs, model):
    """!@brief Plot the FID scores over the training process

    @param fid_scores: list of FID scores, list of floats
    @param num_epochs: number of epochs, int
    @param model: used model name, str

    @return None
    """

    # Create the x-axis
    x = np.arange(num_epochs)

    # Plot the FID scores
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, fid_scores, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title(f"FID over {num_epochs} epochs for {model}")

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(
        plot_dir, "fid_for_" + model + "@" + str(num_epochs) + ".png"
    )
    plt.savefig(plot_path)
    plt.close()


def plot_is(is_scores, num_epochs, model):
    """!@brief Plot the IS scores over the training process

    @param is_scores: list of IS scores, list of floats
    @param num_epochs: number of epochs, int
    @param model: used model name, str

    @return None
    """

    # Create the x-axis
    x = np.arange(num_epochs)

    # Plot the IS scores
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, is_scores, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("IS")
    plt.title(f"IS over {num_epochs} epochs for {model}")

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(
        plot_dir, "incept_score_for_" + model + "@" + str(num_epochs) + ".png"
    )
    plt.savefig(plot_path)
    plt.close()


def plot_ddpm_degrade(mnist, timestep, beta, n_T):
    """!@brief Plot the DDPM degraded samples for a given timestep

    @details This function plots the DDPM degraded samples for a given timestep
    using the mnist dataset. It samples 16 images from the dataset and degrades
    them following DDPM's degradation process, according to the timestep.

    @param mnist: the mnist dataset
    @param timestep: the timestep to plot, int
    @param beta: the beta values, tuple of floats
    @param n_T: the number of timesteps, int

    @return None
    """

    # Load an image from the mnist dataset:
    idx = random.sample(range(len(mnist)), 16)
    z_t = torch.stack([mnist[i][0].clone() for i in idx])

    # Degrade the image following the DDPM degradation process
    assert beta[0] < beta[1] < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta[1] - beta[0]) * torch.arange(
        0, n_T + 1, dtype=torch.float32
    ) / n_T + beta[0]
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    eps = torch.randn_like(z_t)  # eps ~ N(0, 1)
    alpha_t = alpha_t[timestep, None, None, None]

    z_t = torch.sqrt(alpha_t) * z_t + torch.sqrt(1 - alpha_t) * eps

    # Create the grid and save the image
    grid = make_grid(z_t, nrow=4)

    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, f"ddpm_degraded_sample_{timestep}.png")

    save_image(grid, plot_path)  # noqa E231
