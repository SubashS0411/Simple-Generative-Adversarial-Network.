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
