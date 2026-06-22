import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_generated_images
import config

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(
    root="data/",
    train=True,
    transform=transform,
    download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

# Models
G = Generator(config.Z_DIM).to(config.DEVICE)
D = Discriminator().to(config.DEVICE)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=config.LR)
opt_D = optim.Adam(D.parameters(), lr=config.LR)

# Training
for epoch in range(config.EPOCHS):
    for real, _ in dataloader:

        real = real.to(config.DEVICE)
        batch_size = real.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(config.DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(config.DEVICE)

        # =====================
        # Train Discriminator
        # =====================
        noise = torch.randn(batch_size, config.Z_DIM).to(config.DEVICE)
        fake_images = G(noise)

        D_real = D(real)
        D_fake = D(fake_images.detach())

        loss_real = criterion(D_real, real_labels)
        loss_fake = criterion(D_fake, fake_labels)

        d_loss = loss_real + loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # =====================
        # Train Generator
        # =====================
        noise = torch.randn(batch_size, config.Z_DIM).to(config.DEVICE)
        fake_images = G(noise)

        output = D(fake_images)
        g_loss = criterion(output, real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{config.EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save samples
    save_generated_images(fake_images, epoch, "outputs/generated_images")