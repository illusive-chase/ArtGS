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
import copy
import json
import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import DeformModel, Scene
from utils.general_utils import safe_state, vis_depth
from utils.metrics import seed_everything


def generate_camera_poses(args, N=30):
    """
    Generate camera poses around the scene center on a circle.

    Parameters:
    - r: Radius of the circle.
    - theta: Elevation angle in degrees.
    - num_samples: Number of samples (camera positions) to generate.

    Returns:
    - poses: A list of camera poses (4x4 transformation matrices).
    """
    poses = []
    file = json.load(open('./arguments/cam_traj.json'))
    traj_info = file[args.dataset][args.subset][args.scene_name]
    radius, r_theta, r_phi = traj_info['radius'], traj_info['theta'], traj_info['phi']
    d_theta, d_phi = traj_info['d_theta'], traj_info['d_phi']

    thetas = np.linspace(r_theta[0] * np.pi, r_theta[1] * np.pi, N//2)
    thetas = np.concatenate([np.zeros(N//2) + r_theta[0] * np.pi, thetas]) + d_theta * np.pi
    azimuths = np.linspace(r_phi[0] * np.pi, r_phi[1] * np.pi, N//2)
    azimuths = np.concatenate([np.zeros(N//2) + r_phi[0] * np.pi, azimuths]) + d_phi * np.pi
    roty180 = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]]) if traj_info['roty180'] else np.eye(4)
    rotx90 = np.array([[1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]]) if traj_info['rotx90'] else np.eye(4)
    
    for theta, azimuth in zip(thetas, azimuths):
        # Convert spherical coordinates to Cartesian coordinates

        x = radius * np.cos(azimuth) * np.cos(theta)
        y = radius * np.sin(azimuth) * np.cos(theta)
        z = radius * np.sin(theta)

        # Camera position
        position = np.array([x, y, z])

        # Compute the forward direction (pointing towards the origin)
        forward = position / np.linalg.norm(position)

        # Compute the right and up vectors for the camera coordinate system
        up = np.array([0, 0, 1])
        if np.allclose(forward, up) or np.allclose(forward, -up):
            up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)

        # Normalize the vectors
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        # Construct the rotation matrix
        rotation_matrix = np.vstack([right, up, forward]).T

        # Construct the transformation matrix (4x4)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        transformation_matrix = roty180 @ rotx90.T @ transformation_matrix

        poses.append(transformation_matrix)
    return poses


def generate(views, args, N=30):
    new_views = []
    poses = generate_camera_poses(args, N)
    for i, pose in enumerate(poses):
        view = copy.deepcopy(views[0])
        view.fid = i / (len(poses) - 1)
        view.gt_alpha_mask = None
        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        view.reset_extrinsic(R, T)
        new_views.append(view)
    return new_views


def generate_video(imgs, video_name, fps=15, brighten=False):
    # imgs: list of img tensors [3, H, W]
    height, width = imgs[0].shape[1], imgs[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img in imgs:
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if brighten:
            img[:, :width//2] = cv2.convertScaleAbs(img[:, :width//2], alpha=0.8, beta=1)
        video.write(img)
    video.release()

def generate_video_ffmpeg(img_path, video_name, fps=15):
    os.system(f'rm {video_name}')
    os.system(f'ffmpeg -framerate {fps} -vsync 0 -i {img_path}' + '/%05d.png -c:v libx264 -crf 0 ' + video_name)


def video2gif(video_file):
    gif_file = video_file.replace('.mp4', '.gif').replace('video', 'gif')
    clip = VideoFileClip(video_file)
    clip.write_gif(gif_file)
    clip.close()


def render_set(args, name, iteration, views, gaussians, pipeline, background, deform, N_frames=60, inverse=False, recenter=False, outerpolation=False):
    model_path = args.model_path
    assert N_frames % 3 == 0
    offset = 0.5 if outerpolation else 0
    timels = np.concatenate([np.linspace(0 - offset, 1 + offset, N_frames//3//2), np.linspace(1 + offset, 0 - offset, N_frames//3 - N_frames//3//2)])
    timels = np.concatenate([timels, timels, timels])
    # timels = np.linspace(0, 1, N_frames)
    dx_list, dr_list = deform.deform.interpolate(gaussians, timels)
    assert len(dx_list) == len(views), (len(dx_list), len(views))
    
    save_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    if os.path.exists(save_dir):
        os.system(f'rm -r {save_dir}')
    makedirs(save_dir, exist_ok=True)

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgb")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    rgbd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgbd")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(rgbd_path, exist_ok=True)

    rgbs, depths = [], []
    vis_mask = None
    if recenter:
        gaussians._xyz -= gaussians._xyz.mean(0, keepdim=True)
    for idx, view in enumerate(tqdm(views)):
        d_xyz, d_rotation = dx_list[idx], dr_list[idx]
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)
        rgbs.append(torch.clamp(results["render"], 0.0, 1.0))
        depths.append(results['depth'])
    if inverse:
        rgbs = rgbs[:N_frames//2] + rgbs[:N_frames//2][::-1] + rgbs[N_frames//2:] + rgbs[N_frames//2:][::-1]
        depths = depths[:N_frames//2] + depths[:N_frames//2][::-1] + depths[N_frames//2:] + depths[N_frames//2:][::-1]
    rgbs = torch.stack(rgbs, 0)
    depths = torch.stack(depths, 0)

    rgbds = []
    for i in range(len(rgbs)):
        rgb = rgbs[i].cpu() # [3, H, W]
        torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(i) + ".png")) 
        depth = vis_depth(depths[i], os.path.join(depth_path, '{0:05d}'.format(i) + ".png")) # [3, H, W]
        rgbd = torchvision.utils.make_grid([rgb, depth], nrow=2, padding=0)
        torchvision.utils.save_image(rgbd, os.path.join(rgbd_path, '{0:05d}'.format(i) + ".png"))
        rgbds.append(rgbd)
    # save video
    scene_name = args.scene_name
    # generate_video_ffmpeg(rgbd_path, os.path.join('data/demo/video', f"{scene_name}.mp4"), fps=10)
    generate_video(rgbds, os.path.join(model_path, name, "ours_{}".format(iteration), f"{scene_name}.mp4"), fps=15)
    video2gif(os.path.join(model_path, name, "ours_{}".format(iteration), f"{scene_name}.mp4"))
    print(f"Saved video to {os.path.join(model_path, name, 'ours_{}'.format(iteration), f'{scene_name}.mp4')}")
    

def render_sets(args, dataset: ModelParams, iteration, pipeline: PipelineParams, N_frames=30):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        deform.update(20000)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cam_traj = scene.getTrainCameras()
        # cam_traj = scene.getTestCameras()
        # id = 0
        # cam_traj = cam_traj[id:id+1] * N_frames
        if args.inverse:
            N_frames = N_frames // 2
        print(N_frames)
        cam_traj = generate(cam_traj, args, N_frames)
        # cam_traj = cam_traj[len(cam_traj)-(N_frames // 2)-1:len(cam_traj)-(N_frames // 2)] * (N_frames // 2) + cam_traj[len(cam_traj)-(N_frames // 2):]
        render_set(args, "render_out" if args.outerpolation else "render", scene.loaded_iter, cam_traj, gaussians, pipeline, background, deform, N_frames, args.inverse, args.recenter, args.outerpolation)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--N_frames", default=30, type=int)
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--outerpolation", action="store_true")
    parser.add_argument("--recenter", action="store_true", help="Recenter the gs for real world demo")

    args = get_combined_args(parser)
    args.source_path = f'data/{args.dataset}/{args.subset}/{args.scene_name}'

    print("Rendering " + args.source_path + ' with '+ args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.N_frames)
