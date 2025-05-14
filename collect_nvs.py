import os
import json

# List of scenes
scenes = [
    "hook150_v3_transl_statictimestep1",
    "jumpingjacks150_v3_tex_statictimestep75",
    "mouse150_v2_transl_statictimestep1",
    "spheres_cube_dataset_v5_statictimestep1",
    "standup150_v3_statictimestep75"
]

# List of maps
maps = [
    #'"chapel_day_4k_32x16_rot0",
    "dam_wall_4k_32x16_rot90",
    #"golden_bay_4k_32x16_rot330"
]

# Base directory
base_dir = "/home/pwojcik/IRGS/outputs"

# Accumulators
psnr_total = 0.0
ssim_total = 0.0
lpips_total = 0.0
count = 0

# Iterate over map and scene combinations
for map_name in maps:
    for scene in scenes:
        json_path = os.path.join(base_dir, f"s2_{map_name}", f"irgs_{scene}", "test", "nvs_results.json")

        if not os.path.isfile(json_path):
            print(f"Missing: {json_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                psnr_total += data.get("psnr_avg", 0.0)
                ssim_total += data.get("ssim_avg", 0.0)
                lpips_total += data.get("lpips_avg", 0.0)
                count += 1
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

# Compute and print averages
if count > 0:
    print("\n✅ Global Averages Across All Valid JSONs:")
    print(f"psnr_avg:  {psnr_total / count:.4f}")
    print(f"ssim_avg:  {ssim_total / count:.4f}")
    print(f"lpips_avg: {lpips_total / count:.4f}")
else:
    print("❌ No valid JSON files found.")
