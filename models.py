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
