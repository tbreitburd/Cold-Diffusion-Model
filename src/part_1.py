"""!@file part_1.py

@brief This file contains the first part of the project.

@details This file contains the first part of the project.
The first part of the project is the implementation of the standard DDPM module.
This code is used to train the model and make predictions,
and to evaluate those using FID and Inception Score. There are 2 extra inputs,
that define the number of epochs and the hyperparameters to be used.
The code is run from the command line with the following command:

python part_1.py num_epochs 'hyper_params'

Available hyperparameters are:
- 'default': Default hyperparameters, as defined in the coursework_starter notebook
- 'light': Shallower CNN and less
- 'more_capacity': Deeper CNN with more capacity
- 'testing': A 2nd set of default hyperparameters for comparison

The code saves the model and the generated images every 20 epochs,
and plots the losses, FID and IS scores over the training process.
The final FID and IS scores are also printed.

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
from torchvision.utils import save_image, make_grid
from cnn_module import CNN
from ddpm_module import DDPM
import funcs

# Ignore UserWarning from inception score metric
warnings.filterwarnings("ignore", category=UserWarning)


# Get the number of epochs and hyperparameters from the command line
num_epochs = int(sys.argv[1])
hyper_params = sys.argv[2]

# Set random seeds
torch.manual_seed(75016)
np.random.seed(75016)

# Define a content saving directory
project_dir = os.getcwd()
content_dir = os.path.join(project_dir, "contents")
if not os.path.exists(content_dir):
    os.makedirs(content_dir)


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
    activation = nn.GELU
elif hyper_params == "light":
    # Shallower CNN
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (16, 64, 16)
    batch_size = 128
    activation = nn.GELU
elif hyper_params == "more_capacity":
    # More capacity
    betas = (1e-4, 0.02)
    n_T = 1000
    lr = 2e-4
    n_hidden = (64, 128, 256, 128, 64)
    batch_size = 128
    activation = nn.GELU
elif hyper_params == "testing":
    # Testing
    betas = (1e-4, 0.02)
    n_T = 100
    lr = 6e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
    activation = nn.SELU

# Perform some basic preprocessing on the data loader
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)

# Create our model with a given choice of hidden layers, activation function,
# and learning rate
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=activation)

# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM(gt=gt, betas=betas, n_T=n_T)
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)


# We create an instance of the Accelerator class to handle device placement
accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
# which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, dataloader)


# Perform a quick sanity check, making sure everything works
for x, _ in dataloader:
    break

with torch.no_grad():
    ddpm(x)


# We train the model for the chosen number epochs
losses = []
FID = []
IS = []
avg_losses = []

for i in range(num_epochs):
    ddpm.train()

    pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
    for x, _ in pbar:
        optim.zero_grad()

        loss = ddpm(x)

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

    ddpm.eval()

    # Now generate some samples and evaluate the model
    with torch.no_grad():
        # Evaluate the model using FID and Inception Score
        avg_losses.append(avg_loss)
        fid_temp = funcs.get_fid(ddpm, dataset, 16, accelerator.device)
        is_temp = funcs.get_is(ddpm, False, 16, accelerator.device)[0]
        FID.append(fid_temp)
        IS.append(is_temp)

        # fmt: off
        # Save samples to `./contents` directory
        if i % 20 == 0:
            degraded, xh = ddpm.sample(
                                        16, (1, 28, 28), accelerator.device
                                        )  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)
            grid1 = make_grid(degraded, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{i:04d}_{hyper_params}.png") # noqa E231
            save_image(grid1, f"./contents/ddpm_degraded_{i:04d}_{hyper_params}.png") # noqa E231
        # fmt: on

        # Save the current model every 20 epochs
        if i % 20 == 0:
            torch.save(
                ddpm.state_dict(),
                "./ddpm_mnist_" + str(i) + "_" + hyper_params + ".pth",
            )  # noqa F541


with torch.no_grad():
    degraded, xh = ddpm.sample(
        16, (1, 28, 28), accelerator.device
    )  # Can get device explicitly with `accelerator.device`
    grid = make_grid(xh, nrow=4)
    grid1 = make_grid(degraded, nrow=4)
    save_image(
        grid, f"./contents/ddpm_sample_{num_epochs:04d}_{hyper_params}.png"  # noqa E231
    )
    save_image(
        grid1,
        f"./contents/ddpm_degraded_{num_epochs:04d}_{hyper_params}.png",  # noqa E231
    )

    # Save the final model, if not already saved
    torch.save(
        ddpm.state_dict(),
        "./ddpm_mnist_" + str(num_epochs) + "_" + hyper_params + ".pth",
    )  # noqa F541

    # Plot the losses, FID and IS scores over the training process
    string = "DDPM_" + hyper_params
    funcs.plot_losses(losses, avg_losses, num_epochs, string)
    funcs.plot_fid(FID, num_epochs, string)
    funcs.plot_is(IS, num_epochs, string)

    # Evaluate the full model using FID and Inception Score

    FID_end = funcs.get_fid(ddpm, dataset, 100, accelerator.device)

    print(f"FID after full training: {FID_end}")

    IS_end = funcs.get_is(ddpm, False, 100, accelerator.device)

    print(f"IS of generated images after full training: {IS_end[0]} +-", IS_end[1])

    IS_real_end = funcs.get_is(dataset, True, 100, accelerator.device)

    print(f"IS of real images: {IS_real_end[0]} +-", IS_real_end[1])
