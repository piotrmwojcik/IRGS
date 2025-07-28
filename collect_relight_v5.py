import os
import json
import math

# List of scenes
scenes = [
    "hook150_v5_spec32_statictimestep1",
    "jumpingjacks150_v5_spec32_statictimestep75",
    "mouse150_v5_spec32_statictimestep1",
    "spheres_v5_spec32_statictimestep1",
    "standup150_v5_spec32_statictimestep75"
]

# Environment light source → target pairs
light_pairs = [
    ("chapel_day_4k_32x16_rot0", "golden_bay_4k_32x16_rot330"),
    ("dam_wall_4k_32x16_rot90", "small_harbour_sunset_4k_32x16_rot270"),
    ("golden_bay_4k_32x16_rot330", "dam_wall_4k_32x16_rot90"),
]

# Base path
base_path = "/home/pwojcik/IRGS/outputs_specular"

# Function to compute mean and standard deviation
def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, std

# Process each pair
for src_light, tgt_light in light_pairs:
    psnr_values = []
    ssim_values = []
    lpips_values = []

    print(f"\n🔦 Source: {src_light} → Target: {tgt_light}")

    for scene in scenes:
        # Construct path to relighting_results.json
        subfolder = f"s2_{src_light}/irgs_{scene}/test_rli_{tgt_light}_light"
        json_path = os.path.join(base_path, subfolder, "relighting_results.json")

        if not os.path.isfile(json_path):
            print(f"❌ Missing: {json_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                psnr_values.append(data.get("psnr_pbr_avg", 0.0))
                ssim_values.append(data.get("ssim_pbr_avg", 0.0))
                lpips_values.append(data.get("lpips_pbr_avg", 0.0))
        except Exception as e:
            print(f"⚠️ Error reading {json_path}: {e}")

    if psnr_values:
        psnr_mean, psnr_std = compute_mean_std(psnr_values)
        ssim_mean, ssim_std = compute_mean_std(ssim_values)
        lpips_mean, lpips_std = compute_mean_std(lpips_values)

        print(f"✅ Results over {len(psnr_values)} scenes:")
        print(f"  PSNR PBR Avg : {psnr_mean:.3f} ± {psnr_std:.3f}")
        print(f"  SSIM PBR Avg : {ssim_mean:.3f} ± {ssim_std:.3f}")
        print(f"  LPIPS PBR Avg: {lpips_mean:.3f} ± {lpips_std:.3f}")
    else:
        print("⚠️ No valid relighting_results.json files found.")
