from typing import Dict

import random
import numpy as np
import torch
import torch.nn as nn


def schedules(order: int, T: int, type: str) -> Dict[str, torch.Tensor]:
    """Returns order and schedule for each row/column in the image."""

    assert order == 1 or order == 2 or order == 3, "Order must be either 1 or 2"

    half_point = T // 2 - 1
    backward = list(range(13, -1, -1))
    forward = list(range(13 + 1, 28))
    index = []
    if order == 1:
        for i in range(half_point + 1):
            index.append(forward[i])
            index.append(backward[i])
        if T % 2 == 1:
            index.append(forward[half_point + 1])
    elif order == 2:
        for i in range(1, half_point + 2):
            index.append(backward[-i])
            index.append(forward[-i])
        if T % 2 == 1:
            index.append(forward[-half_point - 2])
    else:
        # Unique random order
        index = random.sample(range(28), 28)

    index = np.array(index)

    return {type: torch.tensor(index)}


def schedules_7(order: int, T: int, type: str) -> Dict[str, torch.Tensor]:
    """Returns order and schedule for each 4 wide row in the image."""

    assert order == 1 or order == 2 or order == 3, "Order must be either 1 or 2"

    half_point = T // 2 - 1
    idx_groups = list(np.array_split(range(28), 7))
    backward = list(range(3, -1, -1))
    forward = list(range(4, 7))
    index = []
    if order == 1:
        for i in range(half_point + 1):
            index.append(idx_groups[backward[i]])
            index.append(idx_groups[forward[i]])
        if T % 2 == 1:
            index.append(idx_groups[backward[half_point + 1]])
    elif order == 2:
        for i in range(1, half_point + 2):
            index.append(idx_groups[forward[-i]])
            index.append(idx_groups[backward[-i]])
        if T % 2 == 1:
            index.append(idx_groups[-half_point - 2])
    else:
        # Unique random order
        idx = random.sample(range(7), 7)
        index = [idx_groups[i] for i in idx]

    index = np.array(index)

    return {type: torch.tensor(index)}


class Row_Averaging(nn.Module):
    def __init__(
        self,
        gt,
        row_order: int,
        grouping: str,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        if grouping == "7":
            row_schedule = schedules_7(row_order, n_T, "rows_t")
        else:
            row_schedule = schedules(row_order, n_T, "rows_t")

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("rows_t", row_schedule["rows_t"])
        self.rows_t  # Exists! Set by register_buffer

        self.n_T = n_T
        self.criterion = criterion

    def degrade(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Row averaging diffusion for a set time step"""

        rows_t = self.rows_t[: int(t[0].item())]

        z_t = x.clone()

        # Average the rows
        for i in range(x.shape[0]):
            for row in rows_t:
                z_t[i, :, row, :] = torch.mean(z_t[i, :, row, :])

        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Row averaging diffusion"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        z_t = x.clone()
        for i in range(x.shape[0]):
            rows_t = self.rows_t[: t[i]]
            for row in rows_t:
                z_t[i, :, row, :] = torch.mean(z_t[i, :, row, :])

        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(x, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, dataset, size, device) -> torch.Tensor:
        """Algorithm 2 in Bansal et al. (2022)"""

        num_images = len(dataset)
        idx = random.sample(range(num_images), n_sample)
        z_t = torch.stack([dataset[i][0].clone() for i in idx])
        z_t = z_t.to(device)

        original = z_t.clone()

        T = torch.Tensor([self.n_T])
        T = T.expand_as(torch.empty(n_sample))
        T = T.to(device)

        z_t = self.degrade(z_t, T)

        degraded = z_t.clone()

        # tensor_values = torch.FloatTensor(n_sample, 1, 28, 28).uniform_(-0.5, -0.2)
        # z_t = torch.mean(tensor_values, dim=2, keepdim=True).expand_as(tensor_values)
        # z_t = z_t.to(device)

        z_t_direct = self.gt(z_t, T / self.n_T)

        for t in reversed(range(0, self.n_T)):
            if t > 0:
                t = torch.Tensor([t])
                t = t.expand_as(torch.empty(n_sample))
                t = t.to(device)
                x_hat = self.gt(z_t, t / self.n_T)
                z_t = z_t - self.degrade(x_hat, t) + self.degrade(x_hat, t - 1)

            else:
                t = torch.Tensor([t])
                t = t.expand_as(torch.empty(n_sample))
                t = t.to(device)
                z_t = self.gt(z_t, t / self.n_T)

        return original, degraded, z_t_direct, z_t


class Col_Averaging(nn.Module):
    def __init__(
        self,
        gt,
        col_order: int,
        grouping: str,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        if grouping == "7":
            col_schedule = schedules_7(col_order, n_T, "cols_t")
        else:
            col_schedule = schedules(col_order, n_T, "cols_t")

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("cols_t", col_schedule["cols_t"])
        self.cols_t  # Exists! Set by register_buffer

        self.n_T = n_T
        self.criterion = criterion

    def degrade(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Column averaging diffusion for a set time step"""

        # Average the columns
        cols_t = self.cols_t[: int(t[0].item())]
        z_t = x.clone()
        for i in range(x.shape[0]):
            for col in cols_t:
                z_t[i, :, :, col] = torch.mean(z_t[i, :, :, col])

        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Column averaging diffusion"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        z_t = x.clone()
        for i in range(x.shape[0]):
            cols_t = self.cols_t[: t[i]]
            for col in cols_t:
                z_t[i, :, :, col] = torch.mean(z_t[i, :, :, col])

        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(x, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, dataset, size, device) -> torch.Tensor:
        """Algorithm 2 in Bansal et al. (2022)"""

        num_images = len(dataset)
        idx = random.sample(range(num_images), n_sample)
        z_t = torch.stack([dataset[i][0].clone() for i in idx])
        z_t = z_t.to(device)
        original = z_t.clone()
        T = torch.Tensor([self.n_T])
        T = T.expand_as(torch.empty(n_sample))
        T = T.to(device)

        z_t = self.degrade(z_t, T)

        z_t_degraded = z_t.clone()
        # tensor_values = torch.FloatTensor(n_sample, 1, 28, 28).uniform_(-0.5, -0.2)
        # z_t = torch.mean(tensor_values, dim=2, keepdim=True).expand_as(tensor_values)
        # z_t = z_t.to(device)

        z_t_direct = self.gt(z_t, T / self.n_T)

        for t in reversed(range(0, self.n_T)):
            if t > 0:
                t = torch.Tensor([t])
                t = t.expand_as(torch.empty(n_sample))
                t = t.to(device)
                x_hat = self.gt(z_t, t / self.n_T)
                z_t = z_t - self.degrade(x_hat, t) + self.degrade(x_hat, t - 1)

            else:
                t = torch.Tensor([t])
                t = t.expand_as(torch.empty(n_sample))
                t = t.to(device)
                z_t = self.gt(z_t, t / self.n_T)

        return original, z_t_degraded, z_t_direct, z_t
