# View The NAO In Isaac Sim

Phase 1 scene: a single SoftBank / Aldebaran NAO H25 V5.0 standing above a
ground plane, as a free-floating PhysX articulation. No soccer, no ball, no
rewards, no observations, no actions, no training.

See [`assets/licenses/README.md`](../assets/licenses/README.md) for the model's
provenance and licensing.

## One-time asset bootstrap

The NAO meshes are licensed CC BY-NC-ND 4.0 and upstream permits redistribution
only through an installer that obtains the user's explicit assent, so this
repository ships no mesh file. Fetch them once per checkout:

```powershell
cd C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka
python scripts\fetch_nao_meshes.py
```

The script prints the license and requires you to type `I ACCEPT` before it
downloads anything. Add `--accept-license` to skip the prompt in automation.

It downloads the official ROS Noetic `nao_meshes` package, verifies its SHA-256,
and extracts the geometry with the Python standard library. **No WSL, no Ubuntu,
no ROS and no 7-Zip are required** — it runs on native Windows.

## Run the viewer

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"
conda activate env_isaaclab
cd C:\IsaacLab
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_nao.py" --device cuda:0 --rendering_mode performance --kit_args=--/app/vulkan=false
```

The first run converts the URDF to USD (roughly 10 s) and caches the result in
`assets/generated/nao/`. Later runs reuse it.

Close the Isaac Sim window or press `Ctrl+C` to stop.

## Options

```powershell
# headless smoke test, exits on its own
.\isaaclab.bat -p "...\scripts\view_nao.py" --device cuda:0 --headless --max-steps 1

# force regeneration of the derived URDF and the USD
.\isaaclab.bat -p "...\scripts\view_nao.py" --device cuda:0 --rebuild-usd

# spawn higher, and settle for longer before reporting
.\isaaclab.bat -p "...\scripts\view_nao.py" --device cuda:0 --spawn-height 0.5 --settle-steps 600
```

| Flag | Meaning |
| --- | --- |
| `--spawn-height` | Root height in metres. Default 0.36. |
| `--rebuild-usd` | Regenerate the derived URDF and USD before loading. |
| `--settle-steps` | Physics steps used for the gravity check. Default 120. |
| `--max-steps` | Stop after N steps. 0 (default) runs until the window closes. |

## Expected behavior

1. Isaac Sim opens on a ground plane with a NAO hovering ~2.7 cm above it.
2. Diagnostics print: 43 bodies, 42 joints, 42 DOFs, root body `base_link`,
   root not fixed, total mass 5.3054 kg, and the full joint limit table.
3. Physics starts and the robot **collapses to the ground**. That is correct:
   there is no controller yet, and the fall is what validates the articulation,
   the collision geometry and gravity.

## Troubleshooting

**`NAO geometry is missing from this checkout`** — run the bootstrap above. The
viewer checks for meshes before launching Isaac Sim, so this fails in about a
second rather than after a slow startup.

**The GUI crashes during RTX/Vulkan startup** with an access violation in
`rtx.scenedb.plugin.dll` or `carb.scenerenderer-rtx.plugin.dll`. This is a
driver-level issue on this machine, unrelated to the NAO asset. Force
Direct3D 12:

```powershell
--rendering_mode performance --kit_args=--/app/vulkan=false
```

Headless runs are unaffected.

**Kit arguments starting with `--`** must be passed with `=` so argparse does
not read them as new script options:

```powershell
.\isaaclab.bat -p "...\scripts\view_nao.py" --kit_args=--clear-cache
```

**Warnings about unresolved `visuals` prims** for `LElbow`, `RElbow`,
`l_gripper` and `r_gripper` are harmless. Those four links carry inertia but no
visual or collision geometry upstream, so the importer emits a `visuals`
reference that resolves to nothing.

**`DEPRECATION WARNING: Merging bodies with inertia`** is emitted while the 36
fixed sensor frames are merged into their parent links. That is the intended
behavior; see `nao_usd.convert_nao_urdf_to_usd` for why merging is required.

## Run without the Isaac Lab launcher

If your active environment already imports `isaaclab`:

```powershell
cd C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka
python .\scripts\view_nao.py --headless --max-steps 1
```
