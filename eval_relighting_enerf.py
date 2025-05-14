import json
import sys
from scene import Scene
import os
from gaussian_renderer import render_ir
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.light import EnvMap, EnvLight
from utils.graphics_utils import focal2fov, fov2focal, rgb_to_srgb, srgb_to_rgb
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.system_utils import Timing
from scene.dataset_readers import load_img_rgb
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("-e", "--extra", default='', type=str)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    gaussians = GaussianModel(3)
    #print('!!!!! ', gaussians.env_map.shape)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    #gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
        
    # deal with each item
    scene = Scene(dataset, gaussians, load_iteration=loaded_iter)
    frames = scene.getTrainCameras()
    #print(frames[0])

    gaussians.build_bvh()

    #eval_to = os.environ.get("EVAL_TO", "")
    map_path = os.environ.get("MAP_PATH", "")
    map_name = os.environ.get("MAP_NAME", "")

    task_dict = {
        "env6": {
            "capture_list": ["render", "render_env"],
            "envmap_path": map_path,
        },
        #"env12": {
        #    "capture_list": ["render", "render_env"],
        #    "envmap_path": "assets/env_map/envmap12.exr",
        #}
    }
    results_dict = {}

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    
    results_dir = os.path.join(args.model_path, f"test_lg0_rli_{map_name}" + (f"_{args.extra}" if len(args.extra)>0 else ""))
    os.makedirs(results_dir, exist_ok=True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    with open(os.path.join(results_dir, "cmd.txt"), 'w') as cmd_f:
        cmd_f.write(full_cmd)
    
    if args.albedo_rescale == 0:
        base_color_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    else:
        with open(os.path.join(args.model_path, "albedo_scale.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict[str(args.albedo_rescale)], dtype=torch.float32, device="cuda")
    
    for task_name in task_dict:
        results_dict[task_name] = {}
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        gaussians.env_map = EnvLight(path=task_dict[task_name]["envmap_path"], device='cuda', max_res=1024, activation='none').cuda()

        gaussians.env_map.build_mips()
        gaussians.env_map.update_pdf()
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        # sample for colmap convention. Without it envmap is sampled for blender convention
        colmap_rot = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32, device="cuda")
        transform = transform @ colmap_rot
        gaussians.env_map.set_transform(transform)

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "training": False,
            "relight": True,
            "base_color_scale": base_color_scale,
        }

        
        capture_list = task_dict[task_name]["capture_list"]
        if not args.no_save:
            for capture_type in capture_list:
                capture_type_dir = os.path.join(task_dir, capture_type)
                os.makedirs(capture_type_dir, exist_ok=True)

            os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "gt_env"), exist_ok=True)
            
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            mapname = os.environ.get("MAP_NAME", "")
            image_path = os.path.join(args.source_path, frame.image_path.split("/")[-1] + ".png")

            with torch.no_grad():
                render_pkg = render_ir(viewpoint_camera=frame, **render_kwargs)

            mask = torch.ones_like(render_pkg["render"])
            render_pkg["render"] = render_pkg["render"] * mask + (1 - mask) * bg
            #gt_image_env = gt_image + render_pkg["env_only"] * (1 - mask)
            gt_image = frame.original_image
            if not args.no_save:
                save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                #save_image(gt_image_env, os.path.join(task_dir, "gt_env", f"{idx}.png"))
                render_pkg["base_color"] = render_pkg["base_color"] * mask + (1 - mask) * bg
                save_image(render_pkg["base_color"].clamp(0, 1), os.path.join(task_dir, f"base_color_{idx}.png"))
                render_pkg["base_color_linear"] = render_pkg["base_color_linear"] * mask + (1 - mask) * bg
                save_image(render_pkg["base_color_linear"].clamp(0, 1), os.path.join(task_dir, f"base_color_linear_{idx}.png"))

                for capture_type in capture_list:
                    save_image(render_pkg[capture_type], os.path.join(task_dir, capture_type, f"{idx}.png"))
