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
    #"chapel_day_4k_32x16_rot0",
    "dam_wall_4k_32x16_rot90",
    #"golden_bay_4k_32x16_rot330"
]

# Base directory
base_dir = "/home/pwojcik/IRGS/outputs"

# Accumulators
psnr_albedo_total = 0.0
ssim_albedo_total = 0.0
lpips_albedo_total = 0.0
count = 0

# Iterate over all map and scene combinations
for map_name in maps:
    for scene in scenes:
        json_path = os.path.join(base_dir, f"s2_{map_name}", f"irgs_{scene}", "material_results.json")

        if not os.path.isfile(json_path):
            print(f"Missing: {json_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                psnr_albedo_total += data.get("psnr_albedo_avg", 0.0)
                ssim_albedo_total += data.get("ssim_albedo_avg", 0.0)
                lpips_albedo_total += data.get("lpips_albedo_avg", 0.0)
                count += 1
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

# Compute and print averages
if count > 0:
    print("\n✅ Global Averages for Albedo Metrics:")
    print(f"psnr_albedo_avg:  {psnr_albedo_total / count:.4f}")
    print(f"ssim_albedo_avg:  {ssim_albedo_total / count:.4f}")
    print(f"lpips_albedo_avg: {lpips_albedo_total / count:.4f}")
else:
    print("❌ No valid material_results.json files found.")
