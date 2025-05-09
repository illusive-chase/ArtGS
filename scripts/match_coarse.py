from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from rfstudio.data import DynamicDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages, IntensityImages, RGBAImages, Splats
from rfstudio.io import dump_float32_image
from rfstudio.model import GSplatter
from rfstudio.nn import Module
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.trainer import BaseTrainer
from rfstudio.ui import console
from rfstudio.utils.colormap import BinaryColorMap
from rfstudio.utils.decorator import lazy
from torch import Tensor, nn


@dataclass
class TestRender(Task):

    path: Path = Path('outputs/artgs/sapien/table_31249')
    name: str = 'coarse_gs'
    dataset: DynamicDataset = DynamicDataset(path=Path('data/artgs/sapien/table_31249'))
    output: Path = Path('outputs') / 'test_render'
        
    def run(self) -> None:
        self.dataset.to(self.device)
        base_path = self.path / self.name / 'point_cloud' / 'iteration_10000'
        state_0 = GSplatter(num_random=4, background_color='white').to(self.device)
        state_0.__setup__()
        state_0.set_max_sh_degree(0)
        state_0.gaussians = Splats.from_file(base_path / 'point_cloud_0.ply', device=self.device)
        state_1 = GSplatter(num_random=4, background_color='white').to(self.device)
        state_1.__setup__()
        state_1.set_max_sh_degree(0)
        state_1.gaussians = Splats.from_file(base_path / 'point_cloud_1.ply', device=self.device)
        cameras = self.dataset.get_inputs(split='test')[...]
        timestamps = self.dataset.get_meta(split='test')
        cameras = cameras[timestamps < 0.5]

        self.output.mkdir(exist_ok=True, parents=True)
        with console.progress('Rendering') as ptrack:
            for idx, camera in enumerate(ptrack(cameras.view(-1, 1))):
                rgb_0 = state_0.render_rgb(camera)
                rgb_1 = state_1.render_rgb(camera)
                dump_float32_image(self.output / f'{idx:04d}.png', torch.cat((rgb_0.item(), rgb_1.item()), dim=1))


@dataclass
class Matcher(Module):

    load: Path = ...
    name: str = 'coarse_gs'

    def __setup__(self) -> None:
        base_path = self.load / self.name / 'point_cloud' / 'iteration_10000'
        state_0 = GSplatter(num_random=4, background_color='white').to(self.device)
        state_0.__setup__()
        state_0.set_max_sh_degree(0)
        state_0.gaussians = Splats.from_file(base_path / 'point_cloud_0.ply', device=self.device)
        state_1 = GSplatter(num_random=4, background_color='white').to(self.device)
        state_1.__setup__()
        state_1.set_max_sh_degree(0)
        state_1.gaussians = Splats.from_file(base_path / 'point_cloud_1.ply', device=self.device)
        self.gs = [state_0, state_1]
        self.is_static_0 = nn.Parameter(torch.zeros(state_0.gaussians.shape[0]))
        self.is_static_1 = nn.Parameter(torch.zeros(state_1.gaussians.shape[0]))

    @lazy
    def update(self) -> None:
        for i in [0, 1]:
            self.gs[i].to(self.device)
            self.gs[i].gaussians = self.gs[i].gaussians.to(self.device)


@dataclass
class MatchTrainer(BaseTrainer):

    lr: float = 1e-3

    def setup(self, model: Matcher, dataset: DynamicDataset) -> ModuleOptimizers:
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict={
                'is_static': Optimizer(
                    category=torch.optim.Adam,
                    lr=self.lr,
                    modules=model,
                )
            },
        )

    def step(
        self,
        model: Matcher,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        
        model.update()
        
        depth_0, depth_1 = [], []
        is_static_0, is_static_1 = [], []

        model.gs[0].gaussians.replace_(colors=model.is_static_0.sigmoid().unsqueeze(-1).repeat(1, 3))
        model.gs[1].gaussians.replace_(colors=model.is_static_1.sigmoid().unsqueeze(-1).repeat(1, 3))
        with torch.no_grad():
            for camera in inputs.view(-1, 1):
                depth_0.append(model.gs[0].render_depth(camera).item())
                depth_1.append(model.gs[1].render_depth(camera).item())

                with torch.enable_grad():
                    is_static_0.append(model.gs[0].render_rgba(camera).item())
                    is_static_1.append(model.gs[1].render_rgba(camera).item())

            depth_0, depth_1 = torch.stack(depth_0), torch.stack(depth_1)

            d0 = depth_0[..., :1]
            a0 = depth_0[..., 1:]
            d1 = depth_1[..., :1]
            a1 = depth_1[..., 1:]
            union = (a0 > 0.5) & (a1 > 0.5)
            depth_diff = ((d0 - d1).abs() * 10 * union.float()).clamp(0, 1) # [B, H, W, 1]
            cov_diff = (a0 - a1).abs() # [B, H, W, 1]

        is_static_0 = torch.stack(is_static_0)
        is_static_1 = torch.stack(is_static_1)
        s0 = is_static_0[..., :1].clamp(0.01, 0.99)
        s1 = is_static_1[..., :1].clamp(0.01, 0.99)
        depth_loss = -torch.add(
            ((depth_diff * (1 - s0).log() + (1 - depth_diff) * s0.log()) * (a0 > 0.9).float()).mean(),
            ((depth_diff * (1 - s1).log() + (1 - depth_diff) * s1.log()) * (a1 > 0.9).float()).mean()
        )
        cov_loss = -torch.add(
            ((cov_diff * (1 - s0).log() + (1 - cov_diff) * s0.log()) * (a0 > 0.9).float()).mean(),
            ((cov_diff * (1 - s1).log() + (1 - cov_diff) * s1.log()) * (a1 > 0.9).float()).mean(),
        )

        metrics = {
            'depth_diff': depth_diff.mean(),
            'cov_diff': cov_diff.mean(),
            'depth_loss': depth_loss.detach(),
            'cov_loss': cov_loss.detach(),
        }

        image = None
        if visual:
            vis_depth = DepthImages(depth_0[0]).concat(DepthImages(depth_1[0])).visualize().resize_to(400, 400)
            vis_static = (
                IntensityImages([is_static_0[0][..., 2:], is_static_1[0][..., 2:]])
                    .visualize(BinaryColorMap('RdBu'))
                    .blend((1, 1, 1))
                    .resize_to(400, 400)
            )
            left = torch.cat((
                torch.cat((vis_depth[0].item(), vis_depth[1].item()), dim=1),
                torch.cat((vis_static[0].item(), vis_static[1].item()), dim=1),
            ), dim=0)
            right = gt_outputs.blend((1, 1, 1)).item()
            image = torch.cat((left, right), dim=1).clamp(0, 1)
        
        return depth_loss + cov_loss, metrics, image        


if __name__ == '__main__':
    TaskGroup(
        test=TestRender(cuda=0),
        match=TrainTask(
            cuda=0,
            dataset=DynamicDataset(path=Path('data/artgs/sapien/table_31249')),
            model=Matcher(load=Path('outputs/artgs/sapien/table_31249')),
            experiment=Experiment(name='match'),
            trainer=MatchTrainer(
                num_steps=1000,
                batch_size=8,
                num_steps_per_val=25,
                mixed_precision=False,
            ),
        ),
    ).run()
