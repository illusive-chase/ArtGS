import shutil
import subprocess
from pathlib import Path

with open('selected_data.txt') as f:
    paths = [Path(x.rstrip()) for x in f.readlines() if x.rstrip().endswith('.png')]

Path('outputs').mkdir(exist_ok=True, parents=True)

for p in paths:
    if 'table_31249' not in str(p):
        continue
    dataset = p.parts[1]
    subset = p.parts[2]
    scene = p.parts[3]

    model_path = Path('outputs') / dataset / subset / scene

    if not (model_path / 'coarse_gs_prior').exists():
        shutil.copytree(model_path / 'coarse_gs', model_path / 'coarse_gs_prior')

    cmd = [
        'python', 'train_predict.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'pred'),
        '--eval',
        '--resolution', '8',
        '--iterations', '3000',
        '--densify_grad_threshold', '0.001',
        '--coarse_name', 'coarse_gs_prior',
        '--use_art_type_prior',
        '--joint_types', 's,p,r,p,r',
        '--random_bg_color',
    ]
    print('####################')
    print(' '.join(cmd))
    # subprocess.run(cmd)

    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'artgs_prior'),
        '--eval',
        '--resolution', '1',
        '--iterations', '20000',
        '--coarse_name', 'coarse_gs_prior',
        '--seed', '0',
        '--use_art_type_prior',
        '--random_bg_color',
        '--densify_grad_threshold', '0.001'
    ]
    print('####################')
    print(' '.join(cmd))
    # subprocess.run(cmd)


    cmd = [
        'python', 'render_video.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'artgs_prior'),
        '--resolution', '1',
        '--iteration', 'best',
        '--white_background',
        '--N_frames', '30',
        # '--outerpolation',
    ]
    print('####################')
    print(' '.join(cmd))
    subprocess.run(cmd)
