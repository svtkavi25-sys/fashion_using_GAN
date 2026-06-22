import torch
import torchvision
import os

def save_generated_images(images, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    grid = torchvision.utils.make_grid(images[:25], nrow=5, normalize=True)
    torchvision.utils.save_image(grid, f"{output_dir}/epoch_{epoch}.png")