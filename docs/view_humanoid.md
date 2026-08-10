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
```

Close the Isaac Sim window or press `Ctrl+C` to stop.
