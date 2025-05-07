from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import seaborn as sns
import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import Points
from rfstudio.visualization import Visualizer


@dataclass
class Script(Task):

    path: Path = Path('outputs/artgs/sapien/table_31249')
    name: str = 'artgs'
    viser: Visualizer = Visualizer()
        
    def run(self) -> None:
        pts_path = self.path / self.name / 'part_pts.pkl'
        pts_lst = torch.load(pts_path)[1:]

        pallete = torch.tensor(sns.color_palette("hls", len(pts_lst))).float()
        vis = {}
        for idx, positions in enumerate(pts_lst):
            if positions.shape[0] == 0:
                continue
            pts = Points(positions=positions, colors=pallete[idx].expand_as(positions))
            vis[f'part_{idx}'] = pts

        self.viser.show(**vis)


if __name__ == '__main__':
    Script(cuda=0).run()
