i want to build a GAN

Yes. Let's build the **complete working MNIST GAN project** using **PyTorch**. This version is intentionally simple, but it is structured properly so you can understand it, run it, save checkpoints, and see the generator improve over time.

## 1. Project structure

Create a folder:

```text
simple-gan/
│
├── models.py
├── train.py
├── generate.py
├── requirements.txt
│
├── data/
├── outputs/
└── checkpoints/
```

---

## 2. `requirements.txt`

Create `requirements.txt`:

```text
torch
torchvision
matplotlib
numpy
```

Install everything:

```bash
pip install -r requirements.txt
```

---

# 3. `models.py`

This contains both the **Generator** and **Discriminator**.

```python
import torch
import torch.nn as nn


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):

    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()

        self.model = nn.Sequential(

            # 100 -> 256
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),

            # 256 -> 512
            nn.Linear(256, 512),
            nn.ReLU(True),

            # 512 -> 1024
            nn.Linear(512, 1024),
            nn.ReLU(True),

            # 1024 -> 784
            nn.Linear(1024, 784),

            # Output range: [-1, 1]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)

        # Convert 784 values into 28 x 28 image
        x = x.view(x.size(0), 1, 28, 28)

        return x


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            # 784 -> 1024
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            # 1024 -> 512
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 -> 1
            nn.Linear(512, 1),

            # Probability
            nn.Sigmoid()
        )

    def forward(self, x):

        # Flatten image:
        # [batch, 1, 28, 28]
        #       ↓
        # [batch, 784]

        x = x.view(x.size(0), -1)

        return self.model(x)
```

---

# 4. `train.py`

This is the **main training program**.

It will:

* Download MNIST
* Create Generator
* Create Discriminator
* Train both
* Save generated images
* Save models
* Print losses
* Use GPU automatically if available

```python
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

EPOCHS = 30

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
```

---

# 5. `generate.py`

After training, this script generates **new digits** from random noise.

```python
import os

import torch

from torchvision.utils import save_image

from models import Generator


# ============================================================
# Configuration
# ============================================================

NOISE_DIM = 100

NUM_IMAGES = 64

MODEL_PATH = "checkpoints/generator_latest.pth"

OUTPUT_PATH = "outputs/generated_digits.png"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Create Generator
# ============================================================

generator = Generator(
    noise_dim=NOISE_DIM
).to(device)


# ============================================================
# Load trained model
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please train the GAN first using train.py"
    )


generator.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


generator.eval()


# ============================================================
# Generate images
# ============================================================

noise = torch.randn(
    NUM_IMAGES,
    NOISE_DIM,
    device=device
)


with torch.no_grad():

    generated_images = generator(noise)


# ============================================================
# Save images
# ============================================================

save_image(
    generated_images,
    OUTPUT_PATH,
    nrow=8,
    normalize=True
)


print("=" * 60)
print("Generation completed!")
print("=" * 60)

print(f"Saved to: {OUTPUT_PATH}")
```

---

# 6. Run the project

Your folder should now look like:

```text
simple-gan/
│
├── models.py
├── train.py
├── generate.py
├── requirements.txt
│
├── data/
├── outputs/
└── checkpoints/
```

Open terminal inside the project folder.

### Create virtual environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Then:

```bash
python train.py
```

---

# 7. What happens when you run it?

First, PyTorch downloads MNIST.

You'll see something similar to:

```text
============================================================
MNIST GAN
============================================================
Device: cuda
GPU: NVIDIA ...
============================================================

Dataset size: 60000
Number of batches: 469

Starting training...
```

Then:

```text
Epoch [1/30] ...
Epoch [1/30] completed | D Loss: ... | G Loss: ...

Epoch [2/30] ...
Epoch [2/30] completed | D Loss: ... | G Loss: ...
```

And so on.

---

# 8. Check your generated images

During training, you'll get:

```text
outputs/
│
├── epoch_001.png
├── epoch_002.png
├── epoch_003.png
├── ...
├── epoch_030.png
```

This is one of the coolest parts of the project.

You can open:

```text
epoch_001.png
```

and compare it with:

```text
epoch_010.png
```

and:

```text
epoch_030.png
```

The progression should look roughly like:

```text
Epoch 1
┌─────────────────────────────┐
│ ░▒ ▓░ ░▒ ░▓ ▒░ ░▒ ▓░ ░▒    │
│ ▓░ ░▒ ▓░ ░░ ▒▓ ░▒ ░▓ ░▒    │
│      Mostly noise           │
└─────────────────────────────┘


Epoch 10
┌─────────────────────────────┐
│  3    7    1    8    4     │
│  5    2    9    3    6     │
│  Rough digits               │
└─────────────────────────────┘


Epoch 30
┌─────────────────────────────┐
│  3    7    1    8    4     │
│  5    2    9    3    6     │
│  Much more recognizable     │
└─────────────────────────────┘
```

The exact results will vary.

---

# 9. Generate completely new images

After training:

```bash
python generate.py
```

You'll get:

```text
outputs/generated_digits.png
```

The important thing is that these digits **were not directly copied from MNIST**.

The process is:

```text
Random Noise
     ↓
   Generator
     ↓
Generated Digit
```

For example:

```text
Random noise #1 → Generator → "7"
Random noise #2 → Generator → "3"
Random noise #3 → Generator → "8"
Random noise #4 → Generator → "1"
```

---

# 10. How the complete system works

You now have:

```text
                    MNIST DATASET
                         │
                         │
                         ▼
                 ┌───────────────┐
                 │ REAL IMAGES   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ DISCRIMINATOR │
                 └───────┬───────┘
                         │
                    Real / Fake
                         ▲
                         │
                         │
Random Noise ───► ┌───────────────┐
                  │   GENERATOR   │
                  └───────┬───────┘
                          │
                          ▼
                     FAKE IMAGE
                          │
                          ▼
                   DISCRIMINATOR
                          │
                          ▼
                    Real / Fake
```

And training continuously improves both networks.

---

# 11. Very important: don't worry about the losses

One common mistake when learning GANs is expecting:

```text
Epoch 1   Loss = 2.5
Epoch 2   Loss = 2.0
Epoch 3   Loss = 1.5
Epoch 4   Loss = 1.0
```

GANs don't necessarily behave like that.

You might see:

```text
D Loss: 1.20
G Loss: 0.75

D Loss: 0.91
G Loss: 1.12

D Loss: 1.04
G Loss: 0.83

D Loss: 0.97
G Loss: 1.01
```

That's not automatically a problem.

**Look at the generated images.**

---

# 12. If you have an NVIDIA GPU

Since you're using a Lenovo LOQ with an NVIDIA RTX GPU, the script will automatically try:

```python
torch.device("cuda")
```

You can check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If you get:

```text
True
```

you're good.

If:

```text
False
```

the project still works using CPU, although training will be slower.

---

# 13. One thing I recommend changing initially

For your **first test**, change:

```python
EPOCHS = 30
```

to:

```python
EPOCHS = 5
```

Run:

```bash
python train.py
```

Make sure everything works.

Then change it back:

```python
EPOCHS = 30
```

and train properly.

---

# 14. Your final GitHub project

Once it works, you can make the project much more impressive.

I'd structure the GitHub repository like this:

```text
MNIST-GAN/
│
├── models.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── epoch_001.png
│   ├── epoch_010.png
│   ├── epoch_020.png
│   └── epoch_030.png
│
├── checkpoints/
│
└── data/
```

And your README can explain:

```text
# MNIST GAN

A simple Generative Adversarial Network built using PyTorch
to generate handwritten digits.

## Architecture

Generator:
100 → 256 → 512 → 1024 → 784

Discriminator:
784 → 1024 → 512 → 1

## Dataset

MNIST

## Framework

PyTorch

## Results

Generated handwritten digits from random noise.
```

### After this works

Don't stop at this version. The **best next step** would be to build **DCGAN on MNIST/CIFAR-10** using `Conv2d` and `ConvTranspose2d`. That will teach you why convolutional GANs are much better for images and gives you a significantly stronger AI/ML project for your portfolio.

