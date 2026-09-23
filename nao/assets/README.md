# NAO H25 V5.0 asset

Third-party robot description. See [`licenses/README.md`](licenses/README.md)
for full attribution, upstream commit SHAs and the licensing decisions that
govern this directory.

## Layout

```text
nao/assets/
|-- urdf/nao.urdf        # tracked   - verbatim upstream URDF, BSD 3-Clause
|-- meshes/V40/          # UNTRACKED - fetched geometry, CC BY-NC-ND 4.0
|-- texture/             # UNTRACKED - fetched texture, CC BY-NC-ND 4.0
`-- README.md            # this file
```

`meshes/` and `texture/` are intentionally **not** committed: upstream permits
redistribution of the NAO geometry only through an installer that obtains the
user's explicit assent. Populate them once per checkout with:

```powershell
python scripts\fetch_nao_meshes.py
```

## Notes on the URDF

* `<robot name="NaoH25V50">` — NAO H25, version 5.0.
* 79 links, 78 joints: 25 independent actuated joints (the "25" in H25),
  17 `mimic` joints (`RHipYawPitch` follows `LHipYawPitch`; the finger and
  thumb joints follow `LHand` / `RHand`), and 36 fixed joints that carry
  sensor frames (cameras, sonars, FSRs, bumpers, IMU, tactile).
* Every mesh reference carries `scale="0.1 0.1 0.1"`. The upstream geometry is
  authored in decimetres; this factor converts it to metres. **Do not remove
  it** — without it the robot renders ten times too large.
* Visual geometry is `.dae`, collision geometry is the matching `*_0.10.stl`.
* The mesh files live under `meshes/V40/` even for the V5.0 model: upstream
  reuses the V4.0 geometry for the V5.0 description.
