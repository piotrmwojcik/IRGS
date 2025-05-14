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

# Environment light source → target pairs
light_pairs = [
    ("chapel_day_4k_32x16_rot0", "golden_bay_4k_32x16_rot330"),
    ("dam_wall_4k_32x16_rot90", "small_harbour_sunset_4k_32x16_rot270"),
    ("golden_bay_4k_32x16_rot330", "dam_wall_4k_32x16_rot90"),
]

# Base path
base_path = "/home/pwojcik/IRGS/outputs"

# Process each pair
for src_light, tgt_light in light_pairs:
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    count = 0

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
                psnr_total += data.get("psnr_pbr_avg", 0.0)
                ssim_total += data.get("ssim_pbr_avg", 0.0)
                lpips_total += data.get("lpips_pbr_avg", 0.0)
                count += 1
        except Exception as e:
            print(f"⚠️ Error reading {json_path}: {e}")

    if count > 0:
        print(f"✅ Averages over {count} scenes:")
        print(f"  PSNR PBR Avg : {psnr_total / count:.4f}")
        print(f"  SSIM PBR Avg : {ssim_total / count:.4f}")
        print(f"  LPIPS PBR Avg: {lpips_total / count:.4f}")
    else:
        print("⚠️ No valid relighting_results.json files found.")
