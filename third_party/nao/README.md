# Third-party NAO model sources

This directory records the provenance and licensing of the third-party
SoftBank Robotics / Aldebaran **NAO H25 V5.0** model data used by this project.

**Nothing in this repository is an original NAO model.** The kinematic
description and the 3D geometry are the work of the upstream authors listed
below and remain under their original licenses. This project only adds Isaac
Sim / Isaac Lab integration code.

---

## 1. Kinematic description — `ros-naoqi/nao_robot`

| Field | Value |
| --- | --- |
| Upstream project | <https://github.com/ros-naoqi/nao_robot> |
| Upstream commit SHA | `67476469a1371b00b17538eb6ea336367ece7d44` |
| Commit date | 2018-10-31 |
| Upstream package | `nao_description` (version 0.5.15) |
| Original authors | Armin Hornung, Stefan Osswald |
| Upstream maintainers | Séverin Lemaignan, Vincent Rabaud |
| Copyright | Copyright (c) 2009-2013, A. Hornung, University of Freiburg |
| License | BSD 3-Clause — see [`LICENSE.nao_robot.txt`](LICENSE.nao_robot.txt) |

### Files copied from `nao_robot`

Exactly **one** file was copied, **verbatim and unmodified**:

| Upstream path | Path in this repository |
| --- | --- |
| `nao_description/urdf/naoV50_generated_urdf/nao.urdf` | [`assets/robots/nao/urdf/nao.urdf`](../../assets/robots/nao/urdf/nao.urdf) |

* SHA-256 of the copied file: `50da7a565da17664...` (full digest verified by
  `tests/test_nao_assets.py`).
* The file declares `<robot name="NaoH25V50">`, i.e. the NAO H25 version 5.0
  model, which is the version this project targets.
* No other file from `nao_robot` is present in this repository. In particular
  `nao_apps`, `nao_bringup`, the `nao_robot` metapackage, ROS launch files,
  Gazebo integration, xacro sources, scripts and upstream git history are
  **not** vendored. This remains an Isaac Lab project, not a ROS project.

### Modifications

The vendored `nao.urdf` is **not modified**.

For Isaac Sim a *derived* URDF is generated at build time into
`assets/generated/nao/nao_isaac.urdf` (an untracked build artifact — see
section 4). The derivation is performed by
[`source/humanoid_soccer_lab/humanoid_soccer_lab/assets/nao_usd.py`](../../source/humanoid_soccer_lab/humanoid_soccer_lab/assets/nao_usd.py)
and does exactly two things:

1. Rewrites `package://nao_meshes/meshes/...` mesh references to
   repository-relative filesystem paths, so that Isaac Sim resolves the
   geometry without any ROS runtime, `ROS_PACKAGE_PATH`, or catkin workspace.
2. Drops the `<gazebo>` and `<transmission>` elements, which describe Gazebo
   plugins and ROS control transmissions that have no meaning in Isaac Sim.

The derived file keeps the original XML comment header and gains an explicit
BSD attribution banner. **No link, joint, inertial, limit, axis, visual,
collision or mesh `scale` value is altered** — in particular the upstream
`scale="0.1 0.1 0.1"` on every mesh is preserved, because the NAO `.dae`/`.stl`
geometry is authored in decimetres and that factor is what converts it to
metres.

BSD 3-Clause permits this modification; the copyright notice and license text
are retained here and in the generated file.

---

## 2. 3D geometry — `ros-naoqi/nao_meshes`

| Field | Value |
| --- | --- |
| Upstream project | <https://github.com/ros-naoqi/nao_meshes> |
| Upstream commit SHA | `7c5b9f3a880fe38552459214157dcf1969812f4c` |
| Commit date | 2025-04-03 |
| Upstream package version | 0.1.13 |
| Original authors | Vincent Rabaud, Mikael Arguedas |
| Upstream maintainers | Maxime Busy, Surya Ambrose |
| Geometry copyright | Aldebaran Robotics / SoftBank Robotics |
| License | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International — see [`LICENSE.nao_meshes.txt`](LICENSE.nao_meshes.txt) |

Related upstream repository holding the official binary installers:

| Field | Value |
| --- | --- |
| Upstream project | <https://github.com/ros-naoqi/nao_meshes_installer> |
| Upstream commit SHA | `f2bf5251b232551f992c73740b128684ba3fe328` |

### Files copied from `nao_meshes`

**None.** No mesh file is vendored in this repository. See section 3 for the
reasoning and section 5 for how the meshes are obtained.

The only file taken from that repository is its license text, reproduced
verbatim as [`LICENSE.nao_meshes.txt`](LICENSE.nao_meshes.txt), for attribution
purposes.

---

## 3. Licensing decisions — stated explicitly, not assumed

The `nao_meshes` license file and README impose a redistribution condition that
is stricter than plain CC BY-NC-ND. Quoting the upstream `LICENSE` file:

> The meshes are licensed under the following license, and can be redistributed,
> but in their installer form.

and the upstream `README`:

> Aldebaran allows you to redistribute those meshes as long as they are given
> through an installer that asks the user to click/enter "ok/yes".
>
> Justification: in French law, only clicking/entering text is recognized as a
> virtual signature; the LICENSE file is therefore not sufficient.

Three consequences were evaluated and acted upon:

### Decision 3.1 — NAO mesh files are **not** committed to this repository

Committing `.dae` / `.stl` / texture files would redistribute the geometry
outside of an installer that obtains the user's explicit assent, which upstream
explicitly does not permit. Therefore the meshes are **fetched by the user, on
their own machine**, by [`scripts/fetch_nao_meshes.py`](../../scripts/fetch_nao_meshes.py),
which reproduces the license and requires the user to type an explicit
acceptance before anything is downloaded.

`assets/robots/nao/meshes/` and `assets/robots/nao/texture/` are therefore
listed in `.gitignore`.

### Decision 3.2 — the generated USD is **not** committed to this repository

Isaac Sim cannot instantiate a URDF directly; it converts it to USD. The
resulting USD **embeds the NAO geometry**, so it is a format-converted
reproduction of the licensed meshes.

Under CC BY-NC-ND 4.0 §2(a)(1)(B) the licensee may *"produce and reproduce, but
not Share, Adapted Material for NonCommercial purposes only"*. Generating the
USD locally for non-commercial simulation is therefore permitted; publishing it
in a public git repository would be "Sharing" and is not.

The conversion output therefore goes to `assets/generated/nao/`, which is
**untracked** (`.gitignore`) and regenerated on demand from the user's own
locally fetched meshes. This is the reason the project uses a generated
directory at all, rather than checking in a ready-made USD.

### Decision 3.3 — the derived URDF is a build artifact

The derived URDF contains only BSD-licensed content, so committing it would be
permissible. It is nevertheless generated into the same untracked
`assets/generated/nao/` directory because it embeds absolute paths to the
user's locally fetched mesh tree and would otherwise go stale. Keeping it
generated avoids duplicating the upstream URDF in two tracked copies.

### Non-commercial scope

CC BY-NC-ND 4.0 restricts use of the NAO geometry to **non-commercial**
purposes. Any commercial use of this project would require removing the NAO
meshes or obtaining a separate license from SoftBank Robotics / Aldebaran.

### Not legal advice

The above is a good-faith engineering reading of the upstream license terms,
documented so the decision is reviewable rather than silent. It is not legal
advice.

---

## 4. Where each group of files lives, and under which license

| Path | Origin | License |
| --- | --- | --- |
| `assets/robots/nao/urdf/nao.urdf` | `nao_robot` @ `6747646`, verbatim | BSD 3-Clause (`LICENSE.nao_robot.txt`) |
| `assets/robots/nao/meshes/**`, `assets/robots/nao/texture/**` | `nao_meshes` @ 0.1.13, fetched locally, **untracked** | CC BY-NC-ND 4.0 (`LICENSE.nao_meshes.txt`) |
| `assets/generated/nao/nao_isaac.urdf` | derived from the BSD URDF, **untracked** | BSD 3-Clause |
| `assets/generated/nao/**/*.usd*` | derived from BSD URDF + CC BY-NC-ND meshes, **untracked** | CC BY-NC-ND 4.0 (Adapted Material, do not redistribute) |
| everything else in this repository | this project | see top-level `LICENSE` |

The top-level project `LICENSE` applies **only** to this project's own code. It
does **not** apply to, and does not relicense, any upstream NAO asset.

---

## 5. How the meshes are obtained

`scripts/fetch_nao_meshes.py` downloads the official ROS Noetic binary package
of `nao_meshes` and extracts only the mesh and texture payload:

| Field | Value |
| --- | --- |
| Source | `http://packages.ros.org/ros/ubuntu/pool/main/r/ros-noetic-nao-meshes/` |
| Package | `ros-noetic-nao-meshes_0.1.13-0focal.20250426.003811_amd64.deb` |
| SHA-256 | `8149c70bd13a7cc89938f3c57c23d0da9f7ea9683aefabbfad7f6860f49a0b63` |
| Publisher | ROS build farm, maintainer Maxime Busy \<mbusy@softbankrobotics.com\> |

The `.deb` is a plain `ar` archive containing `data.tar.xz`, both of which are
parsed with the Python standard library, so the fetch works on **native
Windows** with no WSL, no ROS, no catkin and no third-party tooling.

The upstream `.run` installers in `ros-naoqi/nao_meshes_installer` are Linux
x86-64 ELF binaries and cannot execute on native Windows; the ROS `.deb` is the
same 0.1.13 mesh payload published by the same maintainer. Because that route
does not carry the installer's own click-through dialog, the fetch script
implements the assent step itself: it prints the CC BY-NC-ND 4.0 license and
refuses to download anything until the user types `I ACCEPT`.

Files extracted (78 geometry files + 1 texture):

```text
assets/robots/nao/meshes/V40/*.dae      # 39 visual meshes
assets/robots/nao/meshes/V40/*_0.10.stl # 39 collision meshes
assets/robots/nao/texture/textureNAO.png
```

The `texture/` directory sits beside `meshes/` because the upstream `.dae`
files reference the texture as `../../texture/textureNAO.png`; that layout is
required for the material to resolve.
