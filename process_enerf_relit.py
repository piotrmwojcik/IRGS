import os
from PIL import Image

# Define the input and output directories
input_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts/test_lg0_rli_test_env_j6_i24_light/"
output_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts/test_lg0_rli_test_env_j6_i24_light_for_paper/"

os.makedirs(output_dir, exist_ok=True)

# Target image extensions
extensions = ('.png',)  # restrict to PNG files

# Walk through all files recursively
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(extensions):
            input_path = os.path.join(root, filename)
            try:
                with Image.open(input_path) as img:
                    # Resize if bigger than 500x300
                    if img.width > 500 or img.height > 300:
                        new_size = (img.width // 2, img.height // 2)
                        img = img.resize(new_size, Image.LANCZOS)

                    # Crop the image
                    left = 115
                    top = 25
                    right = img.width - 85
                    bottom = img.height - 5

                    # Ensure cropping is within bounds
                    if right > left and bottom > top:
                        img = img.crop((left, top, right, bottom))
                    else:
                        print(f"Skipping crop for {input_path}: crop area too small.")
                        continue

                    # Create a unique filename if duplicates exist
                    rel_path = os.path.relpath(input_path, input_dir)
                    flat_name = rel_path.replace(os.sep, "_")
                    save_path = os.path.join(output_dir, flat_name)

                    img.save(save_path)
                    print(f"Processed: {save_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

print("Processing complete.")
