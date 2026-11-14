"""What runs on the real NAO: the tracker, its estimator and the safety layer.

Everything a policy reads on the robot is computed here, from the robot's own
signals: joint encoders, the inertial unit and the foot pressure sensors. The
simulator's training pipeline (:mod:`humanoid_transfer.nao.sim2real`) computes
the same inputs with a torch transcription of this code, and tests hold the two
equal, so a policy never sees in training a quantity the robot cannot measure.

This package depends only on the Python standard library, NumPy and its own
modules (``tests/test_import_boundaries.py`` enforces it). The folder can then be
copied to the robot on its own, without the extension, Isaac Lab or torch.
"""
