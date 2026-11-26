"""PACT: a physically anchored, multi-body token latent.

One encoder and one decoder serve every body. A body enters only through a
descriptor made of Buckingham-Pi groups of its joints and links
(:mod:`.descriptors`), so dynamically similar bodies have the same descriptor.
The latent is a set of entity tokens (centroid, contacts, free) whose anchored
parts read out the dimensionless skill state ``z*`` (:mod:`.tokens`).

Everything here is torch or NumPy only: no Isaac Lab import.
"""
