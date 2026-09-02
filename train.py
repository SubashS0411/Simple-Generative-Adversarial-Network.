import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import Generator, Discriminator


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128
IMAGE_SIZE = 28

NOISE_DIM = 100

EPOCHS = 5

LEARNING_RATE = 0.0002

BETA1 = 0.5
BETA2 = 0.999

SAMPLE_INTERVAL = 1


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MNIST GAN")
print("=" * 60)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ============================================================
# Create directories
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    # Convert [0, 1] -> [-1, 1]
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")


# ============================================================
# Create models
# ============================================================

generator = Generator(
    noise_dim=NOISE_DIM
).to(device)


discriminator = Discriminator().to(device)


# ============================================================
# Loss function
# ============================================================

criterion = nn.BCELoss()


# ============================================================
# Optimizers
# ============================================================

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


# ============================================================
# Fixed noise
# ============================================================
#
# We keep this noise fixed so that we can compare the
# generated images across different epochs.
#

fixed_noise = torch.randn(
    64,
    NOISE_DIM,
    device=device
)


# ============================================================
# Training
# ============================================================

print()
print("Starting training...")
print()


for epoch in range(1, EPOCHS + 1):

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for batch_index, (real_images, _) in enumerate(dataloader):

        # ----------------------------------------------------
        # Prepare real images
        # ----------------------------------------------------

        real_images = real_images.to(device)

        batch_size = real_images.size(0)


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        real_labels = torch.ones(
            batch_size,
            1,
            device=device
        )

        fake_labels = torch.zeros(
            batch_size,
            1,
            device=device
        )


        # ====================================================
        # 1. Train Discriminator
        # ====================================================

        optimizer_D.zero_grad()


        # ----------------------------------------------------
        # Real images
        # ----------------------------------------------------

        real_output = discriminator(real_images)

        real_loss = criterion(
            real_output,
            real_labels
        )


        # ----------------------------------------------------
        # Fake images
        # ----------------------------------------------------

        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=device
        )


        fake_images = generator(noise)


        # Detach so Generator isn't updated here
        fake_output = discriminator(
            fake_images.detach()
        )


        fake_loss = criterion(
            fake_output,
            fake_labels
        )


        # ----------------------------------------------------
        # Total discriminator loss
        # ----------------------------------------------------

        d_loss = real_loss + fake_loss


        d_loss.backward()

        optimizer_D.step()


        # ====================================================
        # 2. Train Generator
        # ====================================================

        optimizer_G.zero_grad()


        # Generate fresh fake images
        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=device
        )


        fake_images = generator(noise)


        fake_output = discriminator(fake_images)


        # Generator wants discriminator to think
        # fake images are REAL.
        g_loss = criterion(
            fake_output,
            real_labels
        )


        g_loss.backward()

        optimizer_G.step()


        # ----------------------------------------------------
        # Track losses
        # ----------------------------------------------------

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(dataloader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )


    # ========================================================
    # Average losses
    # ========================================================

    avg_d_loss = total_d_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)


    print()
    print(
        f"Epoch [{epoch}/{EPOCHS}] completed | "
        f"D Loss: {avg_d_loss:.4f} | "
        f"G Loss: {avg_g_loss:.4f}"
    )


    # ========================================================
    # Generate sample images
    # ========================================================

    if epoch % SAMPLE_INTERVAL == 0:

        generator.eval()

        with torch.no_grad():

            generated_images = generator(
                fixed_noise
            )

        save_image(
            generated_images,
            f"outputs/epoch_{epoch:03d}.png",
            nrow=8,
            normalize=True
        )


    # ========================================================
    # Save checkpoints
    # ========================================================

    torch.save(
        generator.state_dict(),
        "checkpoints/generator_latest.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "checkpoints/discriminator_latest.pth"
    )


    # Save epoch-specific Generator
    torch.save(
        generator.state_dict(),
        f"checkpoints/generator_epoch_{epoch:03d}.pth"
    )


print()
print("=" * 60)
print("Training completed!")
print("=" * 60)

print()
print("Generated images:")
print("    outputs/")
print()

print("Models:")
print("    checkpoints/")
print()
