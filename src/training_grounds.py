"""!@file training_grounds.py

@brief This file contains a training ground for the different diffusion models.

@details This file contains a training ground for the different diffusion models.
The code is used to train the model and save the model states.

@author Created by T.Breitburd on 23/03/2024
"""


# Importing the necessary libraries
import sys
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from cnn_module import CNN
from ddpm_module import DDPM
from custom_deg_module import Row_Averaging, Col_Averaging
import funcs


# Ignore UserWarning from inception score metric
warnings.filterwarnings("ignore", category=UserWarning)

# Get the number of epochs from the command line
num_epochs = int(sys.argv[1])
hyper_params = sys.argv[2]
custom_deg = sys.argv[3]
orientation = sys.argv[4]

# Set random seeds
torch.manual_seed(75016)
np.random.seed(75016)


# Content saving directory
project_dir = os.getcwd()
content_dir = os.path.join(project_dir, "contents")
content_custom_dir = os.path.join(project_dir, "contents_custom")
if not os.path.exists(content_dir):
    os.makedirs(content_dir)
if not os.path.exists(content_custom_dir):
    os.makedirs(content_custom_dir)

# --------- Training the model ------------
# Code from the coursework_starter notebook
# ----------------------------------------

if hyper_params == "default":
    # Default hyperparameters
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
elif hyper_params == "light":
    # Shallower CNN and less degradation
    betas = (1e-4, 0.02)
    n_T = 500
    lr = 2e-4
    n_hidden = (16, 64, 16)
    batch_size = 128
elif hyper_params == "more_capacity":
    # More capacity
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (64, 128, 256, 128, 64)
    batch_size = 128
elif hyper_params == "testing":
    # Testing
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
if hyper_params == "default_7":
    # Default hyperparameters
    order = 3
    grouping = "7"
    n_T = 7
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
elif hyper_params == "default_28":
    # Default hyperparameters for no grouping
    order = 3
    grouping = "28"
    n_T = 28
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
elif hyper_params == "more_capacity":
    # More capacity for custom degradation
    order = 3
    grouping = "28"
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (64, 128, 256, 128, 64)
    batch_size = 128

# Perform some basic preprocessing on the data loader
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)


# Create our model with a given choice of hidden layers, activation function,
# and learning rate
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)

# -----------------
# Define the model
# -----------------

if custom_deg == "True":
    if orientation == "row":
        dif_model = Row_Averaging(gt=gt, row_order=order, grouping=grouping, n_T=n_T)
    elif orientation == "col":
        dif_model = Col_Averaging(gt=gt, col_order=order, grouping=grouping, n_T=n_T)
else:
    dif_model = DDPM(gt=gt, betas=betas, n_T=n_T)


optim = torch.optim.Adam(dif_model.parameters(), lr=lr)

# We create an instance of the Accelerator class to handle device placement
accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
# which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
dif_model, optim, dataloader = accelerator.prepare(dif_model, optim, dataloader)


# Perform a quick sanity check, making sure everything works
for x, _ in dataloader:
    break

with torch.no_grad():
    dif_model(x)


# We train the model for the chosen number epochs

losses = []
FID = []
IS = []
avg_losses = []

for i in range(num_epochs):
    dif_model.train()

    pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
    for x, _ in pbar:
        optim.zero_grad()

        loss = dif_model(x)

        loss.backward()
        # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

        losses.append(loss.item())
        # fmt: off
        avg_loss = np.average(losses[max(len(losses) - 100, 0):])

        pbar.set_description(
            f"loss: {avg_loss:.3g}" # noqa E231
        )  # Show running average of loss in progress bar
        # fmt: on
        optim.step()

    dif_model.eval()
    avg_losses.append(avg_loss)

    # save model
    if i % 10 == 0:
        if custom_deg == "True":
            torch.save(
                dif_model.state_dict(),
                "./ddpm_mnist_" + str(i + 1) + "_" + hyper_params + ".pth",
            )  # noqa F541
        else:
            torch.save(
                dif_model.state_dict(),
                "./custom_mnist_"
                + str(i + 1)
                + "_"
                + orientation
                + "_"
                + hyper_params
                + ".pth",
            )  # noqa F541


if custom_deg:
    torch.save(
        dif_model.state_dict(),
        "./ddpm_mnist_" + str(num_epochs) + "_" + hyper_params + ".pth",
    )  # noqa F541
    string = f"custom_{orientation}_{hyper_params}"
else:
    torch.save(
        dif_model.state_dict(),
        "./custom_mnist_"
        + str(num_epochs)
        + "_"
        + orientation
        + "_"
        + hyper_params
        + ".pth",
    )  # noqa F541
    string = f"DDPM_{hyper_params}"

funcs.plot_losses(losses, avg_losses, num_epochs, string)
