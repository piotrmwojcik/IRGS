import json
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
from scene.dataset_readers import load_img_rgb
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def find_matching_file(folder, prefix_file):
    prefix = os.path.splitext(prefix_file)[0]  # 'r_0010' from 'r_0010.png'
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith('.png') and f != prefix_file:
            return f
    return None


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    gaussians = GaussianModel(3)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
    gaussians.build_bvh()
    
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
        "training": False,
        "relight": False,
        "base_color_scale": None,
        "material_only": True,
    }
    
    albedo_list = []
    albedo_gt_list = []
    #print('!!! len ', len(frames))
    for idx, frame in enumerate(tqdm(frames, leave=False)):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        match = find_matching_file(os.path.join(args.source_path, 'albedo'), frame["file_path"])

        albedo_path = match
        #print('!!!! ', albedo_path)
        scale_factor = 0.5
        ### RESOLUTION
        from PIL import Image

        subdir = os.environ.get("DATA_SUBDIR", "")
        gt_albedo_np = load_img_rgb(os.path.join(args.source_path, 'albedo', match))
        image_path = os.path.join(args.source_path, f'{subdir}/' + frame["file_path"].split("/")[-1] + ".png")
        image_rgba = load_img_rgb(image_path)
        mask = torch.from_numpy(image_rgba[..., 3:4]).permute(2, 0, 1).float().cuda()
        #print('mask !!!!!!', mask, mask.max().item(), mask.min().item())
        # Resize to 400x400 using bilinear interpolation
        import torch.nn.functional as F
        mask = mask.unsqueeze(0)
        # Interpolate to [1, 1, 400, 400]
        mask_resized = F.interpolate(mask, size=(400, 400), mode='bilinear', align_corners=False).unsqueeze(0)
        # Remove batch dimension: [1, 400, 400]
        #print('mask !!! ', mask.shape)
        #gt_albedo_np = srgb_to_rgb(gt_albedo_np)
        #print(gt_albedo_np)
        gt_albedo = F.interpolate(torch.from_numpy(gt_albedo_np).unsqueeze(0), size=(400, 400), mode='bilinear',
                                  align_corners=False).squeeze(0)
        gt_albedo /= 255.0  # normalize to [
        #gt_albedo_np = rgb_to_srgb(gt_albedo_np[..., :3])  # convert only RGB to linear
        #print('!!! albedo', gt_albedo_np.shape)
        #print(mask)
        #print('!!!! ', gt_albedo_np.shape, mask.permute(1, 2, 0).shape)
        #gt_albedo = (torch.from_numpy(gt_albedo).cuda() * mask.permute(1, 2, 0)).permute(2, 0, 1).float().cuda()

        H = mask.shape[1]
        W = mask.shape[2]
        fovy = focal2fov(fov2focal(fovx, W), H)

        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy,
                            image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

        with torch.no_grad():
            render_pkg = render_ir(viewpoint_camera=custom_cam, **render_kwargs)

        albedo_gt_list.append(srgb_to_rgb(gt_albedo.cuda()).permute(1, 2, 0)[mask[0] > 0])
        albedo_list.append(render_pkg['base_color_linear'].permute(1, 2, 0)[mask[0] > 0])
        
    albedo_gts = torch.cat(albedo_gt_list, dim=0)
    albedo_ours = torch.cat(albedo_list, dim=0)
    albedo_scale_json = {}
    albedo_scale_json["0"] = [1.0, 1.0, 1.0]

    albedo_scale_json["1"] = [(albedo_gts/albedo_ours.clamp_min(1e-6))[..., 0].median().item()] * 3
    albedo_scale_json["2"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).median(dim=0).values.tolist()
    ##print(torch.min(albedo_gts), torch.max(albedo_gts), torch.min(albedo_ours), torch.max(albedo_ours),
    #     albedo_ours.clamp_min(1e-6).median(dim=0).values, (albedo_ours < 1e-6).sum())
    albedo_scale_json["3"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).mean(dim=0).tolist()
    print("Albedo scales:\n", albedo_scale_json)
        
    with open(os.path.join(args.model_path, "albedo_scale.json"), "w") as f:
        print('model_path',args.model_path)
        json.dump(albedo_scale_json, f)