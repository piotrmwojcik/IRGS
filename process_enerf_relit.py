import os
from PIL import Image

# Define input-output folder name suffixes (base path is shared)
base_input_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts"
base_output_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts"

# Folder name suffixes to process
folder_suffixes = [
    "test_lg0_rli_chapel_day_4k_32x16_rot0_light",
    "test_lg0_rli_test_env_j6_i24_light",
    "test_lg0_rli_golden_bay_4k_32x16_rot330_light",
]

# File extensions to process
extensions = ('.png',)

# Process each folder
for suffix in folder_suffixes:
    input_dir = os.path.join(base_input_dir, suffix)
    output_dir = os.path.join(base_output_dir, f"{suffix}_for_paper")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing folder: {input_dir}")

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                input_path = os.path.join(root, filename)
                try:
                    with Image.open(input_path) as img:
                        # Resize if bigger than 500x300
                        if img.width > 500 or img.height > 300:
                            new_size = (img.width // 4, img.height // 4)
                            img = img.resize(new_size, Image.LANCZOS)

                        # Crop the image
                        left = 115
                        top = 25
                        right = img.width - 85
                        bottom = img.height - 5

                        if right > left and bottom > top:
                            img = img.crop((left, top, right, bottom))
                        else:
                            print(f"Skipping crop: {input_path} — crop area too small.")
                            continue

                        # Flatten file path to avoid collisions
                        rel_path = os.path.relpath(input_path, input_dir)
                        flat_name = rel_path.replace(os.sep, "_")
                        save_path = os.path.join(output_dir, flat_name)

                        img.save(save_path)
                        print(f"Saved: {save_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

print("\nAll folders processed.")
