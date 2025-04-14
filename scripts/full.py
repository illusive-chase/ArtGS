import subprocess
from pathlib import Path

with open('selected_data.txt') as f:
    paths = [Path(x.rstrip()) for x in f.readlines() if x.rstrip().endswith('.png')]

Path('outputs').mkdir(exist_ok=True, parents=True)

for p in paths:
    dataset = p.parts[1]
    subset = p.parts[2]
    scene = p.parts[3]

    model_path = Path('outputs') / dataset / subset / scene

    
    cmd = [
        'python', 'train_coarse.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'coarse_gs'),
        '--resolution', '2',
        '--iterations', '10000',
        '--opacity_reg_weight', '0.1',
        '--random_bg_color',
    ]
    print('####################')
    print(' '.join(cmd))
    subprocess.run(cmd)

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
        '--coarse_name', 'coarse_gs',
        '--random_bg_color',
    ]
    print('####################')
    print(' '.join(cmd))
    subprocess.run(cmd)

    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'artgs'),
        '--eval',
        '--resolution', '1',
        '--iterations', '20000',
        '--coarse_name', 'coarse_gs',
        '--seed', '0',
        '--use_art_type_prior',
        '--random_bg_color',
        '--densify_grad_threshold', '0.001'
    ]
    print('####################')
    print(' '.join(cmd))
    subprocess.run(cmd)


    cmd = [
        'python', 'render_video.py',
        '--dataset', dataset,
        '--subset', subset,
        '--scene_name', scene,
        '--model_path', str(model_path / 'artgs'),
        '--resolution', '1',
        '--iteration', 'best',
        '--white_background',
        '--N_frames', '30',
        # '--outerpolation',
    ]
    print('####################')
    print(' '.join(cmd))
    subprocess.run(cmd)
