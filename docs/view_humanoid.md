# View Humanoid In Isaac Lab

This branch adds the first runnable Isaac Lab scene for the project: a single
built-in humanoid on a ground plane with basic lighting. It does not implement
soccer, rewards, observations, training, or a Direct RL environment yet.

## Run From An Isaac Lab Checkout

If you installed Isaac Lab from source, run this repository script through the
Isaac Lab launcher:

```powershell
cd C:\path\to\IsaacLab
.\isaaclab.bat -p C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_humanoid.py
```

On this Windows machine, using the existing Conda environment:

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"
conda activate env_isaaclab
cd C:\IsaacLab
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_humanoid.py" --device cuda:0 --rendering_mode performance --reset-interval 500
```

## Run From An Isaac Lab Python Environment

If your active Python environment already imports `isaaclab` and
`isaaclab_assets`, run:

```powershell
cd C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka
python .\scripts\view_humanoid.py
```

Useful options:

```powershell
python .\scripts\view_humanoid.py --num-humanoids 1
python .\scripts\view_humanoid.py --reset-interval 500
python .\scripts\view_humanoid.py --no-hold-joints
python .\scripts\view_humanoid.py --headless --max-steps 120
```

Close the Isaac Sim window or press `Ctrl+C` to stop.

## Troubleshooting

If Isaac Lab needs a Kit argument that starts with `--`, pass it with `=` so
argparse does not treat it as a new script option:

```powershell
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_humanoid.py" --kit_args=--clear-cache
```

For a short headless smoke test:

```powershell
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_humanoid.py" --headless --device cuda:0 --rendering_mode performance --max-steps 30
```
