"""Command-line tools for the robot's sessions.

Run from the folder that holds ``deploy`` (on the robot, or on a laptop through
``relay.py``), e.g. ``python3 -m deploy.tools.check_sensors --lola``:

* :mod:`.check_sensors` -- the robot's message against what the runtime expects:
  keys and lengths, IMU axes against the accelerometer, gyroscope bias and noise,
  force-sensor totals against the robot's mass, the joints at rest;
* :mod:`.measure_delay` -- the latency from a target to the joint's response, by
  cross-correlation, on a joint swinging free (robot in its harness);
* :mod:`.sysid_record` -- steps and chirps per leg joint, recorded for
  ``nao/scripts/fit_actuators.py``;
* :mod:`.run_policy` -- a run of the tracker (:class:`~..session.Session`).
"""
