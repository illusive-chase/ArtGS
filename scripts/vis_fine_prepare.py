import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import DeformModel, Scene
from utils.general_utils import safe_state
from utils.metrics import seed_everything


def render_sets(args, dataset: ModelParams, iteration):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        deform.update(20000)

        gaussians = GaussianModel(dataset.sh_degree)
        Scene(dataset, gaussians, load_iteration=iteration)

        d_values_list = deform.step(gaussians, is_training=False)
        pred_joint_types = deform.deform.joint_types[1:]
        num_d_joints = len(pred_joint_types)
        part_pts = []
        for mask_id in range(-1, num_d_joints + 1):
            x = gaussians.get_xyz
            mask_part = d_values_list[0]['mask'] == mask_id
            part_pts.append(x[mask_part].cpu())
        torch.save(part_pts, os.path.join(args.model_path, 'part_pts.pkl'))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    args.source_path = f'data/{args.dataset}/{args.subset}/{args.scene_name}'

    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration)

    # dataset=artgs
    # subset=sapien
    # scenes=(table_31249)
    # model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    # python scripts/vis_fine_prepare.py \
    #     --dataset ${dataset} \
    #     --subset ${subset} \
    #     --scene_name ${scene} \
    #     --model_path ${model_path} \
    # python scripts/vis_fine_prepare.py --dataset artgs --subset sapien --scene_name table_31249 --model_path outputs/artgs/sapien/table_31249/artgs