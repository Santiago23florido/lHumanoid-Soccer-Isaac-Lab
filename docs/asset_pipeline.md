# Asset Pipeline

## Robot — NAO H25 V5.0 (done, Phase 1)

The robot pipeline is implemented. It runs entirely from the vendored URDF, with
no manual step in the Isaac Sim GUI:

1. **Source and license.** NAO H25 V5.0 URDF from `ros-naoqi/nao_robot`, BSD
   3-Clause, vendored verbatim at `assets/robots/nao/urdf/nao.urdf`. Geometry
   from `ros-naoqi/nao_meshes`, CC BY-NC-ND 4.0, fetched locally and never
   committed. See `third_party/nao/README.md`.
2. **Bootstrap geometry.** `scripts/fetch_nao_meshes.py` downloads and verifies
   the meshes after explicit license acceptance.
3. **Derive an Isaac-resolvable URDF.** `nao_usd.build_derived_urdf` rewrites
   the ROS `package://` mesh references to repository-relative paths, drops
   `<gazebo>` and `<transmission>`, and works around the `.` in the
   `<Link>_0.10.stl` filenames that USD rejects as a prim name. No physical
   quantity is altered.
4. **Convert to USD.** `nao_usd.convert_nao_urdf_to_usd` drives the Isaac Lab
   `UrdfConverter` (`isaacsim.asset.importer.urdf`) into `assets/generated/nao/`.
5. **Verify in simulation.** `scripts/view_nao.py` reports bodies, joints, DOFs,
   masses, limits and the articulation root read back from PhysX, and checks
   scale, left/right symmetry, uprightness and gravity.
6. **Expose a reusable config.** `NAO_CFG` in `humanoid_soccer_lab/assets/nao.py`.

Still to do for RL: a joint-order map matching the policy action vector, and an
actuator model with real gains (drives are currently zero).

## Field

1. Define the field dimensions and goal geometry.
2. Keep collision geometry simple for training speed.
3. Add visual materials only after the physics scene is stable.

## Ball

1. Start with a sphere primitive config for fast iteration.
2. Calibrate mass, friction, restitution, and damping.
3. Replace with a USD asset only if visuals or custom collision are needed.
