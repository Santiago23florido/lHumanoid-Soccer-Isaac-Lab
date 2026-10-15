"""The learned side of the transfer: a latent space shared by every body.

The one subpackage of :mod:`transfer` that uses torch. It never imports Isaac
Lab -- a test enforces it -- so the models train and are tested on rollouts
without a simulator, and the simulator only appears in the scripts that
capture those rollouts and run the result closed-loop.

``model``
    Per-body encoder and decoder, and the shared skill policy and latent
    dynamics, with the anchored and free parts of the latent kept apart.
``losses``
    Imitation, anchoring to the analytic skill state, latent dynamics and the
    bottleneck on the free part.
``dataset``
    Teacher rollouts as within-episode transitions.
``train``
    Offline training and the evaluation that asks whether the skill really
    flows through the shared space.
``memory``
    A shared memory that reads only the history of the skill state, on a grid
    in body time, so the body cannot reach the shared space through it.
``limbs``
    A decoder split by limb, each head reading its own share of ``u``.
``sequences``
    Rollouts as streams whose history windows never cross a reset.
``probes``
    Linear and MLP probes over wide, lazily gathered inputs.
``train_memory``
    The memory model's two-stage training and its evaluation.
``events``
    Each contact's next change and next landing, read off a rollout's ``z*``.
``plan`` / ``train_plan``
    A shared contact planner (hazard and landing) and a policy committed to
    it -- calibrated, and not used by the policy: the result that led to the
    contract.
``skill_torch``
    ``z*`` in torch, held to the NumPy definition, for environments that
    reward a body by the state it reaches.
``contract`` / ``train_contract``
    The interface fixed by physics instead of learned: a generator per task
    that samples the next ``z*`` from the history of ``ẑ*`` and the command,
    and its training, gate and chained-gait measurement. Bodies follow it in
    :mod:`humanoid_transfer.transfer.tasks.tracking`.
"""
