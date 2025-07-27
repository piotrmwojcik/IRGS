from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load image
img = Image.open('/Users/piotrwojcik/PycharmProjects/IRGS/spheres_cube_dataset_v5_statictimestep1/albedo/r_00660104.png').convert('RGBA')  # ensure 4 channels
img_np = np.array(img)

# Separate channels
r, g, b, a = img_np[..., 0], img_np[..., 1], img_np[..., 2], img_np[..., 3]

# Display
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(r, cmap='Reds')
axs[0].set_title('Red Channel')
axs[1].imshow(g, cmap='Greens')
axs[1].set_title('Green Channel')
axs[2].imshow(b, cmap='Blues')
axs[2].set_title('Blue Channel')
axs[3].imshow(a, cmap='gray')
axs[3].set_title('Alpha Channel')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()