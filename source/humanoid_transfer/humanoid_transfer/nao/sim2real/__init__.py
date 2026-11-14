"""The real NAO's imperfections, modelled in the simulator.

The tracker trains against an ideal robot: drives with no delay and no torque
limit that depends on speed, sensors read without bias, and an estimator that
reads the simulator's own link states. This package adds what separates that
robot from the real one:

* an actuator model (delay, torque-speed saturation, reflected rotor inertia,
  friction), with parameters read from a robot card and refitted from data
  recorded on the robot;
* sensor models (inertial unit offset and bias, encoder quantisation and
  calibration offset, finite-difference joint velocity, foot pressure sensors);
* the robot-side estimator in torch, a transcription of
  :mod:`humanoid_transfer.nao.deploy` that tests hold equal to it;
* domain randomisation over the parameters the robot card leaves uncertain.

Every option is off by default, so the published tasks are unchanged.
"""
