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

import os
import sys
import math
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json, cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pyexr
import imageio as imageio
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array # #
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            # #
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # #
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1],
            ])
        # #
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.exists(image_path):
            image_path = image_path.replace('.JPG', '.jpg')
        image = Image.open(image_path)

        mask_rgba = Image.open(
            image_path
                .replace("/images/", "/masks/", 1)
                .replace(".jpg", ".png")
        ).convert("RGB")  # Ensure 3-channel image

        # Convert to NumPy array
        mask_rgba_np = np.array(mask_rgba)
        mask_from_image = (mask_rgba_np[:, :, :3] == 255).all(axis=-1)

        print('!!! masking ', mask_from_image.shape)

        # Combine with existing mask (assuming it may come from alpha channel elsewhere)
        #if intr.model=="SIMPLE_RADIAL":
        #    image = cv2.undistort(np.array(image), K, np.array([intr.params[3], 0,0,0]))
        #    image = Image.fromarray(image.astype('uint8')).convert('RGB')
        # #
        real_im_scale = image.size[0] / width
        K[:2] *=  real_im_scale

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=mask_from_image)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # #
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print('Load Ply color and normals failed, random init')
        colors = np.random.rand(*positions.shape) / 255.0
        normals = np.random.rand(*positions.shape)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # #
    spc_ply_path = os.path.join(path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        fovx = contents.get("camera_angle_x", None)
        if fovx is None:
            w = contents["w"]
            fl_x = contents["fl_x"]
            fovx = 2 * math.atan(w / (2 * fl_x))
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            # if not idx % 10 == 0:
            #     continue
            file_path = frame["file_path"]
            if '.png' not in file_path:
                file_path = file_path + extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            ### NOTE !!!!!
            # Here R has been transposed, R = w2c.T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            subdir = os.environ.get("DATA_SUBDIR", "")
            image_path = os.path.join(path, subdir, os.path.basename(cam_name))
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            if norm_data.shape[-1] == 4:
                mask = norm_data[:, :, 3] > 0.5
            else:
                mask = None

            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # #
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            # #
            # For blender datasets, we consider its camera center offset is zero (ideal camera)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    
    if eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    else:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("Reading Test Transforms")
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_img_rgb(path):
    
    if path.endswith(".exr"):
        exr_file = pyexr.open(path)
        img = exr_file.get()
        img[..., 0:3] = rgb_to_srgb(img[..., 0:3])
        # img[..., 0:3] = rgb_to_srgb(img[..., 0:3], clip=False)
    else:
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    return img

def load_mask_bool(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask

def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            # if not idx % 10 == 0:
            #     continue
            # if idx > 1:
            #     break
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0
            
            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            arr = image[..., :3] * mask[..., None] + bg * (1 - mask[..., None])
            
            # H, W = image.shape[:2]
            # fo = fov2focal(fovx, W)
            # K = np.array([
            #     [fo, 0, W/2],
            #     [0, fo, H/2],
            #     [0, 0, 1],
            # ])
            
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1]))
    return cam_infos

def readSynthetic4RelightInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms2(path, transformsfile, white_background, 
                               extension=".png", benchmark_size = 512, debug=False):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            if os.path.exists(os.path.join(path, frame["file_path"] + '.png')):
                image_path = os.path.join(path, frame["file_path"] + '.png')
            else:
                image_path = os.path.join(path, frame["file_path"] + '.exr')
                
            mask_item = frame["file_path"].replace("test", "test_mask").replace("train", "train_mask")
            if os.path.exists(os.path.join(path, mask_item + '.png')):
                mask_path = os.path.join(path, mask_item + '.png')
            else:
                mask_path = os.path.join(path, mask_item + '.exr')
            
            image_name = Path(image_path).stem

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)
            image = cv2.resize(image, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (benchmark_size, benchmark_size), interpolation=cv2.INTER_AREA)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = image * mask[..., None] + bg * (1 - mask[..., None])

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1]))

            if debug and idx >= 5:
                break

    return cam_infos

def readStanfordORBInfo(path, white_background, eval, extension=".exr", benchmark_size = 512, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms2(path, "transforms_train.json", white_background, 
                                                 extension, benchmark_size, debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms2(path, "transforms_test.json", white_background, 
                                                    extension, benchmark_size, debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo,
    "StanfordORB": readStanfordORBInfo,
}