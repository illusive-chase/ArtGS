from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import appdirs
import numpy as np
import requests
import torch
from rfstudio.data import SfMDataset
from rfstudio.data.dataparser import DPKUDataparser
from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras
from rfstudio.io import dump_float32_image
from rfstudio.utils.process import run_command
from tqdm import tqdm


def get_colmap_version(colmap_cmd: str = "colmap", default_version: str = "3.8") -> str:
    """
    Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command(f"{colmap_cmd} -h", verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            version = line.split(" ")[1]
            return version
    warnings.warn(f"Could not find COLMAP version. Using default {default_version}")
    return default_version


def get_vocab_tree() -> Path:
    """
    Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("rfstudio")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in tqdm(
                r.iter_content(chunk_size=1024),
                total=int(total_length + 1023) // 1024,
                desc="Downloading vocab tree...",
                ascii=True,
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename


@dataclass
class ImageReconstruction(Task):

    source: Path = ...

    target: Path = ...

    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""

    refine_intrinsics: bool = True
    """If True, do bundle adjustment to refine intrinsics."""

    colmap_cmd: str = 'colmap'

    verbose: bool = False

    def run(self) -> None:
        assert self.source.is_dir(), f"The argument 'source' must be a directory. {self.source} received instead."
        assert not self.target.exists(), f"The directory {self.target} has already existed."
        assert self.target.parent.exists(), f"The parent directory {self.target.parent} must exist."

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            colmap_version = get_colmap_version(self.colmap_cmd)
            colmap_database_path = tmpdir / "database.db"

            # Feature extraction
            feature_extractor_cmd = [
                f"{self.colmap_cmd} feature_extractor",
                f"--database_path {colmap_database_path}",
                f"--image_path {self.source}",
                "--ImageReader.single_camera 1",
                f"--SiftExtraction.use_gpu {1 if self.device_type == 'cuda' else 0}",
                f"--SiftExtraction.gpu_index {-1 if self.cuda is None else self.cuda}",
            ]
            feature_extractor_cmd = " ".join(feature_extractor_cmd)
            print("Running COLMAP feature extractor...")
            run_command(feature_extractor_cmd, verbose=self.verbose)
            print("Done.")

            # Feature matching
            feature_matcher_cmd = [
                f"{self.colmap_cmd} {self.matching_method}_matcher",
                f"--database_path {colmap_database_path}",
                f"--SiftMatching.use_gpu {1 if self.device_type == 'cuda' else 0}",
                f"--SiftMatching.gpu_index {-1 if self.cuda is None else self.cuda}",
            ]
            if self.matching_method == "vocab_tree":
                vocab_tree_filename = get_vocab_tree()
                feature_matcher_cmd.append(f'--VocabTreeMatching.vocab_tree_path "{vocab_tree_filename}"')
            feature_matcher_cmd = " ".join(feature_matcher_cmd)
            print('Running COLMAP feature matcher...')
            run_command(feature_matcher_cmd, verbose=self.verbose)
            print("Done.")

            # Bundle adjustment
            sparse_dir = tmpdir / "distort"
            sparse_dir.mkdir()
            mapper_cmd = [
                f"{self.colmap_cmd} mapper",
                f"--database_path {colmap_database_path}",
                f"--image_path {self.source}",
                f"--output_path {sparse_dir}",
                "--Mapper.multiple_models 0",
            ]
            if tuple(colmap_version.split('.')) >= ('3', '7'):
                mapper_cmd.append("--Mapper.ba_global_function_tolerance=1e-6")

            mapper_cmd = " ".join(mapper_cmd)

            print('Running COLMAP bundle adjustment... (This may take a while)')
            run_command(mapper_cmd, verbose=self.verbose)
            print("Done.")

            if self.refine_intrinsics:
                print('Refine intrinsics...')
                bundle_adjuster_cmd = [
                    f"{self.colmap_cmd} bundle_adjuster",
                    f"--input_path {sparse_dir / '0'}",
                    f"--output_path {sparse_dir / '0'}",
                    "--BundleAdjustment.refine_principal_point 1",
                ]
                run_command(" ".join(bundle_adjuster_cmd), verbose=self.verbose)
                print('Done.')

            shutil.copytree(self.source, self.target / 'images')
            shutil.move(sparse_dir, self.target / 'sparse')
            shutil.move(colmap_database_path, self.target / 'database.db')


@dataclass
class VideoReconstruction(Task):

    source: Path = ...

    target: Path = ...

    fps: float = ...

    refine_intrinsics: bool = True
    """If True, do bundle adjustment to refine intrinsics."""

    colmap_cmd: str = 'colmap'

    verbose: bool = False

    def run(self) -> None:
        assert self.source.is_file(), f"The argument 'source' must be a video file. {self.source} received instead."
        assert not self.target.exists(), f"The directory {self.target} has already existed."
        assert self.target.parent.exists(), f"The parent directory {self.target.parent} must exist."

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            image_path = tmpdir / "images"
            image_path.mkdir()

            ffmpeg_cmd = [
                "ffmpeg",
                f"-i {self.source}",
                "-vsync vfr",
                f"-vf fps={self.fps:.2f}",
                "-q:v 2",
                f'-f image2 {image_path / "%05d.jpg"}',
            ]
            ffmpeg_cmd = " ".join(ffmpeg_cmd)
            run_command(ffmpeg_cmd, verbose=self.verbose)

            colmap_version = get_colmap_version(self.colmap_cmd)
            colmap_database_path = tmpdir / "database.db"

            # Feature extraction
            feature_extractor_cmd = [
                f"{self.colmap_cmd} feature_extractor",
                f"--database_path {colmap_database_path}",
                f"--image_path {image_path}",
                "--ImageReader.single_camera 1",
                f"--SiftExtraction.use_gpu {1 if self.device_type == 'cuda' else 0}",
                f"--SiftExtraction.gpu_index {-1 if self.cuda is None else self.cuda}",
            ]
            feature_extractor_cmd = " ".join(feature_extractor_cmd)
            print("Running COLMAP feature extractor...")
            run_command(feature_extractor_cmd, verbose=self.verbose)
            print("Done.")

            # Feature matching
            feature_matcher_cmd = [
                f"{self.colmap_cmd} sequential_matcher",
                f"--database_path {colmap_database_path}",
                f"--SiftMatching.use_gpu {1 if self.device_type == 'cuda' else 0}",
                f"--SiftMatching.gpu_index {-1 if self.cuda is None else self.cuda}",
            ]
            feature_matcher_cmd = " ".join(feature_matcher_cmd)
            print('Running COLMAP feature matcher...')
            run_command(feature_matcher_cmd, verbose=self.verbose)
            print("Done.")

            # Bundle adjustment
            sparse_dir = tmpdir / "distort"
            sparse_dir.mkdir()
            mapper_cmd = [
                f"{self.colmap_cmd} mapper",
                f"--database_path {colmap_database_path}",
                f"--image_path {image_path}",
                f"--output_path {sparse_dir}",
                "--Mapper.multiple_models 0",
            ]
            if tuple(colmap_version.split('.')) >= ('3', '7'):
                mapper_cmd.append("--Mapper.ba_global_function_tolerance=1e-6")

            mapper_cmd = " ".join(mapper_cmd)

            print('Running COLMAP bundle adjustment... (This may take a while)')
            run_command(mapper_cmd, verbose=self.verbose)
            print("Done.")

            if self.refine_intrinsics:
                print('Refine intrinsics...')
                bundle_adjuster_cmd = [
                    f"{self.colmap_cmd} bundle_adjuster",
                    f"--input_path {sparse_dir / '0'}",
                    f"--output_path {sparse_dir / '0'}",
                    "--BundleAdjustment.refine_principal_point 1",
                ]
                run_command(" ".join(bundle_adjuster_cmd), verbose=self.verbose)
                print('Done.')

            shutil.move(sparse_dir, self.target / 'sparse')
            shutil.move(image_path, self.target / 'images')
            shutil.move(colmap_database_path, self.target / 'database.db')


@dataclass
class Script(Task):

    start: Path = ...
    end: Path = ...
    output: Path = ...
    fps: float = 3.0
    verbose: bool = True

    def run(self) -> None:
        (self.output / 'colmap').mkdir(exist_ok=True, parents=True)
        if not (self.output / 'colmap' / 'start').exists():
            VideoReconstruction(
                cuda=self.cuda,
                auto_breakpoint=False,
                traceback_pretty=False,
                source=self.start,
                target=self.output / 'colmap' / 'start',
                fps=self.fps,
                verbose=self.verbose,
            ).join()
        if not (self.output / 'colmap' / 'end').exists():
            VideoReconstruction(
                cuda=self.cuda,
                auto_breakpoint=False,
                traceback_pretty=False,
                source=self.end,
                target=self.output / 'colmap' / 'end',
                fps=self.fps,
                verbose=self.verbose,
            ).join()

        for state in ['start', 'end']:
            dataset = SfMDataset(self.output / 'colmap' / state, dataparser=DPKUDataparser(downsample=1))
            dataset.__setup__()
            dataset.to(self.device)

            inputs, gt_outputs = [], []
            for split in ['train', 'val', 'test']:
                inputs.append(dataset.get_inputs(split=split))
                gt_outputs += [img.item() for img in dataset.get_gt_outputs(split=split)]
            inputs = Cameras.cat(inputs, dim=0)
            gt_outputs = torch.stack(gt_outputs)
            assert inputs.ndim == 1
            focal_length = inputs.fx[0].item()
            image_width = inputs.width[0].item()
            image_height = inputs.height[0].item()
            assert (inputs.fx == focal_length).all() and (inputs.fy == focal_length).all()

            (self.output / state).mkdir(exist_ok=True)

            transforms = {
                'frames': [],
                'camera_angle_x': np.arctan2(image_width, 2 * focal_length) * 2,
                'camera_angle_y': np.arctan2(image_height, 2 * focal_length) * 2,
            }
            c2w = torch.cat((inputs.c2w, torch.tensor([0, 0, 0, 1]).to(inputs.c2w).expand(inputs.shape[0], 1, 4)), dim=1)
            c2w[:, :3, 3] *= 1.5
            (self.output / state / 'train' / 'rgba').mkdir(exist_ok=True, parents=True)
            (self.output / state / 'train' / 'depth').mkdir(exist_ok=True, parents=True)
            for idx, pose, image in zip(range(len(c2w)), c2w, gt_outputs.clamp(0, 1), strict=True):
                frame = {
                    'transform_matrix': pose.detach().cpu().numpy().tolist(),
                    'file_path': str(Path(state) / 'train' / 'rgba' / f'{idx:04d}'),
                    'time': 0.0 if state == 'start' else 1.0,
                }
                transforms['frames'].append(frame)
                dump_float32_image(self.output / state / 'train' / 'rgba' / f'{idx:04d}.png', image)
                dump_float32_image(self.output / state / 'train' / 'depth' / f'{idx:04d}.png', torch.zeros_like(image))
            with open(self.output / f"transforms_train_{state}.json", 'w') as f:
                json.dump(transforms, f, indent=4)


if __name__ == '__main__':
    Script(cuda=0).run()
