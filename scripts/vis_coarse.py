from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import seaborn as sns
import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import Points, Splats
from rfstudio.visualization import Visualizer
from sklearn.cluster import SpectralClustering


@dataclass
class Script(Task):

    path: Path = Path('outputs/artgs/sapien/table_31249')
    name: str = 'coarse_gs'
    viser: Visualizer = Visualizer()

    def _load(self, npy_path: Path) -> Points:
        xyz = torch.from_numpy(np.load(npy_path)).float()
        return Points(positions=xyz)
    
    def _center_info(self) -> List[Tuple[Points, Points]]:
        base_path = self.path / self.name / 'point_cloud' / 'iteration_10000'
        canonical_static = np.load(base_path / 'xyz_static.npy')
        canonical_dynamic = np.load(base_path / 'xyz_dynamic.npy')
        num_slots = np.load(base_path / 'center_info.npy').shape[0]
        print("Finding centers by Spectral Clustering")
        if num_slots > 2:
            cluster = SpectralClustering(num_slots - 1, assign_labels='discretize', random_state=0)
            labels = cluster.fit_predict(canonical_dynamic)
            center_dynamic = np.array([canonical_dynamic[labels == i].mean(0) for i in range(num_slots - 1)])
            labels = np.concatenate([np.zeros(canonical_static.shape[0]), labels + 1])
            center = np.concatenate([canonical_static.mean(0, keepdims=True), center_dynamic])
        else:
            labels = np.concatenate([np.zeros(canonical_static.shape[0]), np.ones(canonical_dynamic.shape[0])])
            center = np.concatenate([canonical_static.mean(0, keepdims=True), canonical_dynamic.mean(0, keepdims=True)])
        lst = []
        x = torch.from_numpy(np.concatenate([canonical_static, canonical_dynamic])).float()
        center = torch.from_numpy(center).float()
        labels = torch.from_numpy(labels).long()
        pallete = torch.tensor(sns.color_palette("hls", num_slots)).float()
        for i in range(num_slots):
            vis_center = center[i] + torch.randn(1000, 3) * 0.05
            vis_pcd = x[labels == i, :]
            lst.append((
                Points(positions=vis_pcd, colors=pallete[i].expand_as(vis_pcd)),
                Points(positions=vis_center, colors=pallete[i].expand_as(vis_center) * 0.6),
            ))
        return lst
        
    def run(self) -> None:
        base_path = self.path / self.name / 'point_cloud' / 'iteration_10000'

        canonical_static = self._load(base_path / 'xyz_static.npy')
        canonical = Splats.from_file(base_path / 'point_cloud.ply').as_points().replace(colors=None)
        center_infos = self._center_info()
        
        vis = {
            'canonical': canonical[canonical_static.shape[0]:],
            'state_0/static': self._load(base_path / 'xyz_static_0.npy'),
            'state_0/dynamic': self._load(base_path / 'xyz_dynamic_0.npy'),
            'state_1/static': self._load(base_path / 'xyz_static_1.npy'),
            'state_1/dynamic': self._load(base_path / 'xyz_dynamic_1.npy'),
        }

        for idx, (pcd, center) in enumerate(center_infos):
            vis[f'part_{idx}/pcd'] = pcd
            vis[f'part_{idx}/center'] = center

        self.viser.show(**vis)


if __name__ == '__main__':
    Script(cuda=0).run()
