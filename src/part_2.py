import sys
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
from custom_deg_module import Row_Averaging, Col_Averaging
import funcs


# Ignore UserWarning from inception score metric
warnings.filterwarnings("ignore", category=UserWarning)


# Get the number of epochs from the command line
num_epochs = int(sys.argv[1])
hyper_params = sys.argv[2]
orientation = sys.argv[3]

# Set random seeds
torch.manual_seed(75016)
np.random.seed(75016)

# ------------- Training the model ---------------
# Code partly from the coursework_starter notebook
# ------------------------------------------------

if hyper_params == "default_7":
    # Default hyperparameters
    order = 3
    grouping = "7"
    n_T = 7
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
elif hyper_params == "default_28":
    # Default hyperparameters
    order = 3
    grouping = "28"
    n_T = 28
    lr = 2e-4
    n_hidden = (16, 32, 32, 16)
    batch_size = 128
elif hyper_params == "more_capacity":
    # More capacity
    order = 3
    grouping = "7"
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


if orientation == "row":
    dif_model = Row_Averaging(gt=gt, row_order=order, grouping=grouping, n_T=n_T)
elif orientation == "col":
    dif_model = Col_Averaging(gt=gt, col_order=order, grouping=grouping, n_T=n_T)

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
    with torch.no_grad():
        original, degraded, direct, xh = dif_model.sample(
            16, dataset, (1, 28, 28), accelerator.device
        )  # Can get device explicitly with `accelerator.device`

        # Normalize the degraded images for better visualization
        for i in range(16):
            deg_min = torch.min(degraded[i])
            deg_max = torch.max(degraded[i])
            degraded[i] = (degraded[i] - deg_min) * (0.5 - (-0.5)) / (
                deg_max - deg_min
            ) - 0.5

        grid = make_grid(xh, nrow=4)
        grid1 = make_grid(original, nrow=4)
        grid2 = make_grid(degraded, nrow=4)
        grid3 = make_grid(direct, nrow=4)

        # Evaluate the model using FID and Inception Score
        avg_losses.append(avg_loss)
        fid_temp = funcs.get_fid(dif_model, dataset, 16, accelerator.device)
        is_temp = funcs.get_is_custom(
            dif_model, dataset, False, 16, accelerator.device
        )[0]
        FID.append(fid_temp)
        IS.append(is_temp)

        # fmt: off
        # Save samples to `./contents` directory
        save_image(grid, f"./contents_custom/custom_sample_{i:04d}.png") # noqa E231
        save_image(grid1, f"./contents_custom/original_sample_{i:04d}.png") # noqa E231
        save_image(grid2, f"./contents_custom/degraded_sample_{i:04d}.png") # noqa E231
        save_image(grid3, f"./contents_custom/direct_sample_{i:04d}.png") # noqa E231

        # fmt: on

        # save model
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


string = f"custom_{orientation}_{hyper_params}"
# Plot the losses, FID and IS scores over the training process
funcs.plot_losses(losses, avg_losses, num_epochs, string)
funcs.plot_fid(FID, num_epochs, string)
funcs.plot_is(IS, num_epochs, string)


# Evaluate the full model using FID and Inception Score

FID_end = funcs.get_fid(dif_model, dataset, 100, accelerator.device)

print(f"FID after full training: {FID_end}")

IS_end = funcs.get_is_custom(dif_model, dataset, False, 100, accelerator.device)

print(f"IS of generated images after full training: {IS_end[0]} +-", IS_end[1])

IS_real_end = funcs.get_is_custom(dif_model, dataset, True, 100, accelerator.device)

print(f"IS of real images: {IS_real_end[0]} +-", IS_real_end[1])