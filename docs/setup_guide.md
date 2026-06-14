# Setup Guide

This project supports Windows 11 and Linux. The validated Windows stack is:

- Python 3.11 in Conda environment `env_isaaclab`
- PyTorch 2.7.0 with CUDA 12.8 wheels
- Isaac Sim 5.1.0
- Isaac Lab 2.3.2
- RSL-RL 5.0.1

Isaac Sim does not require a separately installed CUDA Toolkit or `nvcc`.
The NVIDIA driver must be recent enough for the CUDA 12.8 runtime bundled with
PyTorch and Isaac Sim.

## Windows 11

### 1. Check the machine

In PowerShell:

```powershell
nvidia-smi
```

NVIDIA lists 16 GB VRAM as the minimum for Isaac Sim 5.1. The RTX 4070 Laptop
GPU has 8 GB, so this repository uses conservative defaults and must be run
headless with a reduced environment count.

Enable Windows long paths from an elevated PowerShell, then restart the shell:

```powershell
Set-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled `
  -Type DWord `
  -Value 1
```

### 2. Create the Conda environment

Install Miniconda, open a new PowerShell, and initialize Conda if needed:

```powershell
& "$HOME\miniconda3\Scripts\conda.exe" init powershell
```

Create the project environment:

```powershell
conda env create -f environment-windows.yml
conda activate env_isaaclab
```

### 3. Install PyTorch and Isaac Sim

```powershell
python -m pip install `
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 `
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install "isaacsim[all,extscache]==5.1.0" `
  --extra-index-url https://pypi.nvidia.com
```

Accept NVIDIA's Omniverse EULA once:

```powershell
python -c "import isaacsim"
```

### 4. Install Isaac Lab

Use a short path to avoid Windows path-length problems:

```powershell
git clone --branch v2.3.2 --depth 1 `
  https://github.com/isaac-sim/IsaacLab.git C:\IsaacLab

C:\IsaacLab\isaaclab.bat --install rsl_rl
```

### 5. Install this repository

From the repository root:

```powershell
python -m pip install -r requirements-windows-isaacsim.txt
python -m pip install -e ".[rl,hpo,dev]"
```

The Windows requirements file prevents newer `tensordict`, `wandb`, and dev
tool releases from replacing versions required by Isaac Sim 5.1.

### 6. Verify CUDA and Isaac Sim

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/smoke_isaac_cuda.py --headless
pytest -q
```

The simulator smoke test must print:

```text
ISAAC_SIM_CUDA_STEP_OK
```

### 7. Train

Start small on an 8 GB GPU:

```powershell
python scripts/train_rl.py --headless --num_envs 64 --max_iterations 10
```

For a longer run, begin at 256 environments and monitor VRAM:

```powershell
python scripts/train_rl.py --headless --num_envs 256 --max_iterations 1000
nvidia-smi
```

Reduce `--num_envs` to 64-128 if Isaac Sim reports out-of-memory.

## Linux

Use Python 3.11 and follow the official Isaac Lab 2.3.2 pip installation guide.
Install this package into the same environment:

```bash
pip install -e ".[rl,hpo,dev]"
python scripts/smoke_isaac_cuda.py --headless
pytest -q
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` is missing | NVIDIA driver is missing | Install/update the Windows NVIDIA driver. |
| `import isaaclab` fails | Wrong Conda environment | Run `conda activate env_isaaclab`. |
| Access violation while importing `tensordict` | Incompatible latest release | Install `requirements-windows-isaacsim.txt`. |
| Isaac Sim shutdown hangs on Windows | Kit 5.1 cleanup issue | Use the repository launchers, which request fast cleanup. |
| CUDA out-of-memory | Too many parallel environments | Run headless and lower `--num_envs`. |
| `No module named pxr` | Isaac modules imported before Kit launch | Launch through `AppLauncher` or the provided scripts. |

See also `rl_quickstart.md` and `architecture.md`.
