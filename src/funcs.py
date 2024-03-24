"""!@file funcs.py

@brief This file contains some general use functions for the project,
for model validation as well as plotting.

@details This file contains the following functions:

@author Created by T.Breitburd on 23/03/2024
"""

import torch
import random
import os
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def get_fid(generator, real_data, num_images, device):
    with torch.no_grad():
        # Sample images from the real images dataset
        num_samples = len(real_data)
        idx = random.sample(range(num_samples), num_images)
        real_img = torch.stack([real_data[i][0].clone() for i in idx])

        # Make the image have 3 identical channels
        # so that it can be processed by the FID metric
        real_img = real_img.to(torch.uint8).expand(-1, 3, -1, -1)

        # Sample images from the generator (DDPM)
        gen_img = generator.sample(num_images, (1, 28, 28), device)

        # Make the image have 3 identical channels
        gen_img = gen_img.expand(-1, 3, -1, -1)

        # Put the images in the same device
        gen_img = gen_img.to(real_img.device)

        # Initialize the FID metric
        fid = FrechetInceptionDistance(normalize=True)

        fid.update(real_img, real=True)
        fid.update(gen_img, real=False)
        fid_score = fid.compute()

        return fid_score


def get_is(data_source, is_real, num_images, device):
    with torch.no_grad():
        if is_real:
            # Sample images from the real images dataset
            num_samples = len(data_source)
            idx = random.sample(range(num_samples), num_images)
            img = torch.stack([data_source[i][0].clone() for i in idx])
            img = img.expand(-1, 3, -1, -1)

        else:
            # Sample images from the generator (DDPM)
            img = data_source.sample(num_images, (1, 28, 28), device)

            # Make the image have 3 identical channels
            img = img.expand(-1, 3, -1, -1)

            img = img.to("cpu")

        # Initialize the IS metric
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

    @param losses: list of losses
    @param avg_losses: list of average losses
    @param num_epochs: number of epochs
    @param model: used model name

    @return None
    """

    x = range(len(losses)) / num_epochs
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, losses, color="green", label="Loss")
    plt.plot(
        x,
        avg_losses,
        linestyle="--",
        marker="+",
        color="black",
        label="Epoch average loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")  # Put the y-axis on a log scale
    plt.title(f"Losses over {num_epochs} epochs")
    plt.legend()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "losses_for_" + model + ".png")
    plt.savefig(plot_path)
    plt.close()


def plot_fid(fid_scores, num_epochs, model):
    """!@brief Plot the FID scores over the training process

    @param fid_scores: list of FID scores
    @param num_epochs: number of epochs
    @param model: used model name

    @return None
    """
    x = range(len(fid_scores))
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, fid_scores, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("FID over epochs")

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "fid_for_" + model + ".png")
    plt.savefig(plot_path)
    plt.close()


def plot_is(is_scores, num_epochs, model):
    """!@brief Plot the IS scores over the training process

    @param is_scores: list of IS scores
    @param num_epochs: number of epochs
    @param model: used model name

    @return None
    """
    x = range(len(is_scores))
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(x, is_scores, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("IS")
    plt.title("IS over epochs")

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "incept_score_for_" + model + ".png")
    plt.savefig(plot_path)
    plt.close()
