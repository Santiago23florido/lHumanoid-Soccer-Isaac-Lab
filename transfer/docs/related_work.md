# Related work: what the transfer stack takes from it

This review was compiled on 2026-10-15 to design the next stage: a tracker (decoder) with physics in the model, a
latent that carries balance, and a learned part of the latent shared by the G1 and the NAO. It covers about 60
works. The ones below are those the design takes something from, each with what it is taken for.

- Venues and claims were checked against the arXiv or proceedings page.
- "(preprint)" marks 2026 works not yet peer-reviewed. Verify them before citing them in a paper.
- "(uncertain)" marks a venue that could not be confirmed.

## 1. Trackers of planned references, and what makes them generalize

| work | what it shows | taken for |
| --- | --- | --- |
| Jenelten, He, Farshidian, Hutter. *DTC: Deep Tracking Control.* Science Robotics 2024. arXiv:2309.15462 | Feet and contacts make the interface (see the row note below) | A plan read as contacts and forces; physics-rich plan quantities in the critic; plans corrupted in training |
| Ciebielski, Burgio, Khadiv. *Contact-conditioned learning of multi-gait locomotion policies.* arXiv:2408.00776 (uncertain); Omar, Khadiv. *Learning to Act Through Contact.* L4DC 2026, arXiv:2510.03599 | Conditioning on the next contacts (where, when) generalizes out of distribution far better than velocity + gait + phase; redundant channels make it worse | A balance block of contacts and forces; no redundant inputs to the actor |
| Luo et al. *SONIC.* arXiv:2511.07820 | Scale (data, model) plus adaptive, failure-capped sampling | Adaptive sampling of the plan family |
| Liao et al. *BeyondMimic.* arXiv:2508.08241; Chen et al. *GMT.* arXiv:2506.14770 | Failure-weighted sampling was what made long motions converge; mixture of experts | Failure-weighted replay |
| Li, Stanger-Jones, Heim, Kim. *FLD: Fourier Latent Dynamics.* ICLR 2024. arXiv:2402.13820 | ALP-GMM over a motion latent beats fixed sampling; a fallback switches to safe execution when the target is risky | Learning-progress curriculum; competence-gated plan projection |
| Luo et al. *PHC.* ICCV 2023; *PULSE.* ICLR 2024 | Progressive capacity for hard clips; a variational bottleneck bounds a learned skill latent | A bottleneck on the learned latent |

Row notes:
- *DTC.* An RL policy tracks the footholds and contact phases of a model-based planner. It transfers zero-shot to a planner it never saw, because footholds do not depend on the optimizer. It degrades on foot patterns it never practised. The actor is deliberately not given the planned base pose.

## 2. Physics in the controller: analytic priors with a learned residual

| work | what it shows | taken for |
| --- | --- | --- |
| Jeon, Lee, Hong, Kim. *Residual MPC.* arXiv:2510.12717 | An RL residual added to a GPU-batched MPC tracks a wider command range than either alone, and handles unseen gaits and uneven ground zero-shot | The residual over a physical prior |
| Tay, Chang, Gu. *HybridMimic.* arXiv:2603.06775 (preprint) | The policy outputs contact states and centroidal targets; a centroidal optimization gives feed-forward torques that stay feasible under shift | Feed-forward torque from the plan's contact forces |
| Duan et al. *Learning task space actions for bipedal locomotion.* ICRA 2021. arXiv:2011.04741 | Task-space actions plus analytic inverse dynamics on Cassie | The action in task space, joints by a model |
| Liu, Zhao et al. *Opt2Skill.* arXiv:2409.20514 | Torque-aware, dynamically feasible references plus an RL residual track contact forces better | References that carry forces |
| Xie, Da, Babich, Garg, van de Panne. *GLiDE.* WAFR 2022. arXiv:2104.09771 | RL in centroidal space with a QP below transfers to stepping stones | Centroidal quantities as the learned interface |
| Li, Wu, Esteban, Yang, Drgoňa, Ames. *MPC-guided RL for humanoids.* arXiv:2606.05687 (preprint) | Batched centroidal MPC used in training only, its CoM / momentum / GRF predictions as rewards and critic inputs | Physics in reward and critic |
| Poddar et al. *Embedding classical balance control principles in RL.* arXiv:2603.08619 (preprint) | Capture point, CoM and centroidal momentum in a privileged critic and reward; without them stand-up fails | DCM and momentum in the critic |
| Chane-Sane et al. *CaT: Constraints as Terminations.* IROS 2024. arXiv:2403.18765 | Constraint violations as a termination probability inside PPO, with no reward tuning | Torque and joint-limit constraints |
| Morton, Mohnot, Pavone. *ConstrainedMimic.* CoRL 2026. arXiv:2606.00374 | An operational-space + CBF layer under an RL tracker | A runtime safety layer (later) |
| Romero, Song, Scaramuzza. *Actor-Critic MPC.* ICRA 2024; Amos et al. *Differentiable MPC.* NeurIPS 2018 | A differentiable MPC inside the actor improves out-of-distribution behaviour | Not taken now: no batched differentiable QP in this environment |
| Lutter, Ritter, Peters. *DeLaN.* ICLR 2019 | Euler–Lagrange structure in a network extrapolates better than black-box dynamics | Not in the actor; a candidate for a learned feasibility model |

## 3. Symmetry, smoothness and the deep-learning core

| work | what it shows | taken for |
| --- | --- | --- |
| Mittal, Rudin, Klemm, Allshire, Hutter. *Symmetry considerations for learning task symmetric robot policies.* ICRA 2024 | Mirror augmentation and mirror loss in PPO | Already used: mirror loss; Reynolds symmetrization at deploy |
| Su et al. *Leveraging symmetry in RL-based legged locomotion control.* IROS 2024; Nie et al. *SE-Policy.* AAAI 2026 | A strictly equivariant actor beats augmentation; on a G1, up to 40 % better tracking | Kept; strict equivariance is a later option |
| Ordoñez-Apraez et al. *Morphological symmetries in robotics.* IJRR 2025 | Theory and tooling for morphological symmetries | Basis of the measured joint mirror |
| Chen et al. *Lipschitz-constrained policies.* arXiv:2410.11825 (uncertain) | A gradient penalty on the policy's Lipschitz constant replaces smoothness rewards | Bounded sensitivity to unseen plan inputs |
| Sferrazza, Huang, Liu, Lee, Abbeel. *Body Transformer.* CoRL 2024. arXiv:2408.06316 | Attention masked by the body graph beats MLPs in RL and imitation | Entity tokens |

## 4. Curricula and coverage

| work | what it shows | taken for |
| --- | --- | --- |
| Portelas, Colas, Hofmann, Oudeyer. *ALP-GMM.* CoRL 2019. arXiv:1910.07224 | Sampling environment parameters by absolute learning progress beats random sampling and avoids unlearnable regions | Sampling the plan family by learning progress |
| Jiang, Grefenstette, Rocktäschel. *Prioritized Level Replay.* ICML 2021. arXiv:2010.03934 | Replaying procedural levels by TD-error magnitude improves test generalization | Score each plan cell by its GAE magnitude |
| Margolis et al. *Rapid locomotion via RL.* RSS 2022 / IJRR 2024. arXiv:2205.02824 | A grid curriculum expands the command box where the policy succeeds | Learned ranges: expand a range's edge where it is mastered |
| Margolis, Agrawal. *Walk These Ways.* CoRL 2022. arXiv:2212.03238 | One policy over an explicit gait family | The family's coordinates are gait parameters |
| Kakade, Langford 2002; Munos 2003 | Concentrability bounds on the loss under a shifted state distribution | The coverage theory of §8.1 |

## 5. Shared latents across embodiments

| work | what it shows | taken for |
| --- | --- | --- |
| Do et al. *CrossBFM.* arXiv:2609.38087 (preprint) | See the row note below | Freeze the G1 side; align with paired data; physical normalization first |
| Luo et al. *SONIC.* arXiv:2511.07820 | Separate encoders into one quantized token, aligned by reconstruction + token-matching + cycle losses on retargeted pairs | Alignment losses |
| Wang, Bhatt, Wang, Atanasov. arXiv:2406.01968 | A new robot's encoders trained with adversarial, cycle and latent-dynamics losses, without reward | Latent-dynamics consistency |
| Zhang, Xiao, Efros, Pinto, Wang. *Dynamics cycle-consistency.* ICLR 2021. arXiv:2012.09811 | Correspondence learned from unpaired data through each domain's dynamics | A regularizer when pairs are scarce |
| Yan, Lee. *ImitationNet* (Humanoids 2023) and arXiv:2601.15419 (preprint) | Contrastive alignment with a soft, physics-based similarity kernel; latents per body part | A contrastive term; per-entity latents |
| Zhang, Gu, Zhang. *X-WBC.* arXiv:2609.15213 (preprint) | MSE alignment leaves the embodiment's identity in the tokens | Why a body classifier constrains the learned part |
| Bohlinger et al. *URMA.* CoRL 2024. arXiv:2409.06366 | Per-joint tokens with descriptions, attention-pooled. Its 16 robots include the NAO. Unseen quadrupeds transfer zero-shot, unseen humanoids do not | Entity tokens anchored to physics, not raw joints |
| Xue et al. *XHugWBC.* arXiv:2602.05791 (uncertain) | Physics-consistent morphology randomization (pseudo-inertia), graph-masked then global attention; zero-shot to 7 humanoids | Pseudo-inertia randomization; the hybrid mask |
| Aberman et al. *Skeleton-aware networks.* SIGGRAPH 2020 | Pooling skeletons to a common primal skeleton aligns different bodies by construction | Functional entities (trunk, feet) |
| Gupta, Devin, Liu, Abbeel, Levine. *Invariant feature spaces.* ICLR 2017 | Invariant features from time-aligned proxy skills on two bodies transfer new skills | The paired-data setting itself |
| Oh et al. *PolyMorph.* arXiv:2610.19615 (preprint); Ai et al. *Embodiment scaling laws.* CoRL 2025 | Generalization holds only within the morphology range trained; humanoids are the hardest | Expect interpolation, not extrapolation |

Row notes:
- *CrossBFM.* It freezes the G1's behaviour latent and regresses other humanoids' encoders onto it, using motions retargeted on the same timeline. Recovery is 89 % on the robot most like the others and 12.8 % on the smallest. The authors note that cosine distance does not capture control performance.

## 6. Dimensionless transfer

| work | what it shows | taken for |
| --- | --- | --- |
| Liu, Zhao. *Scale2Real.* NeurIPS 2026 workshop (preprint) | Scale-normalized observations transfer across geometrically scaled H1s (return 0.07 → 1.03) | Support for z* |
| Charvet, Stein, Murray-Smith. *Dimensionless MDPs.* arXiv:2504.10006; Girard 2024 | Buckingham-π states and actions are robust to context shifts | Actions in body units too |
| Oke et al. *Allometric scaling laws for bipedal robots.* arXiv:2603.22560 (preprint) | Real bipeds are not dynamically similar: mass ∝ L², torque ∝ mL | Capability ratios as descriptors, not as latent |
| Shafiee, Bellegarda, Ijspeert. *ManyQuadrupeds.* ICRA 2024 | One policy from 2 to 200 kg; only the pattern-scaling layer differs | A shared plan plus a per-body layer |

## What the design takes, in one place

1. **The decoder.** It keeps a deep-learning core (temporal plan encoder, residual trunk, privileged critic, mirror loss) and carries physics in three places:
   - *In the representation:* heads predicting per-foot CoP, ground force, DCM and falls.
   - *In the action:* a prior from inverse kinematics with the body's Jacobians, plus feed-forward contact forces and gravity, with an RL residual (Residual MPC, HybridMimic, Duan).
   - *In the critic and reward:* the balance block in the critic and in one extra reward term, plus constraints as terminations (DTC, MPC-guided RL, CaT).
2. **The latent's balance block.** Per-foot CoP and ground force, and the DCM offset, are all dimensionless (contact-conditioned policies; DTC).
3. **Generalization without training everywhere.**
   - Ranges are learned: expanded where mastered, sampled by learning progress (Margolis, PLR, ALP-GMM).
   - Pseudo-inertia randomization; plan corruption; a Lipschitz bound.
   - Competence-gated projection at deploy (FLD).
4. **A learned latent part ζ.**
   - *Encoder:* entity tokens with attention, anchored to z*. Its bottleneck predicts balance and falls.
   - *Alignment:* paired procedural data that both bodies execute, with latent-dynamics consistency.
   - *Constraint:* a body classifier at chance level (CrossBFM, SONIC, T. Wang, X-WBC, URMA).
