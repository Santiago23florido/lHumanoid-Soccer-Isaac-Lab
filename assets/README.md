# Assets

This directory is reserved for simulation assets that will be imported into
Isaac Lab.

Asset groups:

- `robots/nao/`: NAO H25 V5.0 description. Tracked URDF plus an untracked,
  locally fetched mesh tree. See [`robots/nao/README.md`](robots/nao/README.md)
  and [`../third_party/nao/README.md`](../third_party/nao/README.md).
- `generated/`: build artifacts produced from the sources above (derived URDF,
  converted USD). Untracked and reproducible; never edit by hand.
- `robots/humanoid/`: placeholder for any future humanoid other than NAO.
- `fields/`: soccer field USD, goal geometry, boundary markings.
- `balls/`: soccer ball USD or primitive config with mass, friction, restitution.

Large binary assets should be tracked with Git LFS before they are committed.
Third-party assets whose license forbids redistribution must stay untracked and
be fetched by a script instead.
