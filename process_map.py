from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import torch

# Open the image and convert to tensor
img_pil = Image.open("020000_env.png").convert("RGB")
grid_img = TF.to_tensor(img_pil)  # shape: [3, H, W], values in [0,1]

# Get dimensions
C, H_total, W = grid_img.shape
padding = 10

# Compute height of each image (assumes two stacked vertically with padding)
H_each = (H_total - 3 * padding) // 2

# Extract the two images (vertically)
img1 = grid_img[:, padding:padding + H_each, padding:padding + W]
img2 = grid_img[:, 2 * padding + H_each:2 * padding + 2 * H_each, padding:padding + W]

# Scale each image by its own max value
img2 = img2
img2_scaled = img2 / img2.max()

# Save the scaled images
save_image(img1, "env1.png")
save_image(img2_scaled, "scaled_env2.png")
