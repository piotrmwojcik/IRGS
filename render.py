#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from scene.light import EnvLight
from torchvision.utils import save_image, make_grid
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_ir as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.dataset_readers import load_img_rgb
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
import json

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    path_prefix = os.path.join(model_path, name, "ours_{}".format(iteration))
    gts_path = os.path.join(path_prefix, "gt")
    
    keys = ["render", "render_env", "diffuse", "specular", "roughness", "base_color", "base_color_linear", "rend_alpha", "rend_normal", "visibility"]
    
    makedirs(gts_path, exist_ok=True)
    for key in keys:
        makedirs(os.path.join(path_prefix, key), exist_ok=True)
    
    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        
        psnr_avg += psnr(image, gt_image).mean().double().item()
        ssim_avg += ssim(image, gt_image).mean().double().item()
        if not args.no_lpips:
            lpips_avg += lpips(image, gt_image, net_type='vgg').mean().double().item()
        
        if args.no_save:
            continue
        
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        for key in keys:
            out = render_pkg[key]
            if 'normal' in key:
                mask_nonblack = (out != 0).any(dim=0, keepdim=True)  # shape: [1, H, W]
                out = (out + 1) / 2
                out = out * mask_nonblack
            if 'position' in key:
                out = (out + 1) / 2
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            torchvision.utils.save_image(out,
                                        os.path.join(path_prefix, key, '{0:05d}'.format(idx) + ".png"))
            #orchvision.utils.save_image(out * view.mask.float().cuda(), os.path.join(path_prefix, key, '{0:05d}'.format(idx) + ".png"))
            
    psnr_avg /= len(views)
    ssim_avg /= len(views)
    lpips_avg /= len(views)
    results_dict = {}
    results_dict["psnr_avg"] = psnr_avg
    results_dict["ssim_avg"] = ssim_avg
    results_dict["lpips_avg"] = lpips_avg
    print("\n[ITER {}] Evaluating {} set: PSNR {} SSIM {} LPIPS {}".format(iteration, name, psnr_avg, ssim_avg, lpips_avg))
    with open(os.path.join(model_path, name, "nvs_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(model_path, name, "nvs_results.json"))
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.build_bvh()      
        gaussians.env_map.update_pdf()
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
        
        bg_color = [1,1,1]# if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        map_path = os.path.join(dataset.model_path,
                     "point_cloud",
                     "iteration_" + str(scene.loaded_iter),
                     "point_cloud1.exr")
        gaussians.env_map = EnvLight(path=map_path,
                                     device='cuda', max_res=1024,
                                     activation=gaussians.env_map.activation_name).cuda()
        gaussians.env_map.set_transform(transform)

        #env_dict = gaussians.render_env_map()

        #print('!!! keys ', env_dict.keys())

        #grid = [
        #    env_dict["env1"].permute(2, 0, 1),
        #    env_dict["env2"].permute(2, 0, 1),
        #]
        #grid = make_grid(grid, nrow=1, padding=10)
        #save_image(grid, os.path.join('outputs/s2_dam_wall_4k_32x16_rot90/irgs_hook150_v3_transl_statictimestep1/test/',
        #                              f"debug_env.png"))

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)