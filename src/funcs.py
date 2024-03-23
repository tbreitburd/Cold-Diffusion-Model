"""!@file funcs.py

@brief This file contains some general use functions for the project,
for model validation as well as plotting.

@details This file contains the following functions:

@author Created by T.Breitburd on 23/03/2024
"""

import torch
import random
from torchmetrics.image.fid import FrechetInceptionDistance


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
