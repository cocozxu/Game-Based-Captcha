# Trace-the-Tunnel — Checkpoint 1: Evaluation Plan

## Project Goal

Can an adversarial system, in our case the Claude Code agent, produce trajectories through a constrained tunnel game that are distinguishable from those produced by human players?

The broader question is whether we can distinguish LLM agents from humans in fine-grained, continuous motor tasks at the level of the physical movement pattern used to complete the task. This not only measures for motor behavior but also how LLM agents might reason about human behavior and mimic it in this game.

## What We Are Measuring

The game records every mouse event (x, y, timestamp) as a player drags through a curved tunnel. Both humans and agents play the same fixed pool of 10 tunnels. A trajectory is either a success (reached the end without leaving the tunnel) or a failure. We ran 10+ samples for both humans and agents.

We define "distinguishable" operationally: a classifier trained on labeled trajectories should perform better than chance (AUC > 0.5) when trying to separate human from agent. The evals below are the concrete machinery for answering that question — they should be able to give a definitive yes or no regardless of which agent variant is being tested.

## Current Datasets

| Source | Trajectories | Completed | Completion rate |
|--------|-------------|-----------|----------------|
| `human` | 187 | 172 | 92% |
| `visual agent` | 123 | 122 | 99% |

The agent used so far (`visual agent`) computes the tunnel centerline, injects perpendicular Gaussian noise, and replays the resulting waypoints via Playwright.

---

## Experiments and Evals

### Eval 1 — Feature Summary

**Question:** What scalar properties of a trajectory differ between humans and agents?

We extract 25 features from each trace organized into six groups. The justification for each group is below.

- **Kinematics — speed, acceleration, jerk.** Human arm movements follow the "minimum jerk" principle (Flash & Hogan, 1985): the central nervous system selects trajectories that minimize the integral of squared jerk, producing smooth bell-shaped speed profiles. An agent generating discrete waypoints and interpolating between them does not optimize for this, so its jerk distribution will be structurally different. Speed and acceleration statistics are among the most commonly used features in the mouse dynamics literature for separating humans from bots (Zheng et al., 2011; Shen et al., 2013).

- **Curvature.** The Steering Law (Accot & Zhai, 1997) describes how humans move through constrained tunnel-like paths: movement time scales with the integral of path curvature divided by tunnel width. The law predicts that humans slow down at high-curvature segments and speed up on straight stretches, producing a distinctive curvature-speed coupling. An agent following pre-computed waypoints does not need to obey this law, so its curvature profile will be flatter and decoupled from its speed.

- **Path geometry.** Fitts' Law (1954) predicts that humans trade off movement speed against spatial accuracy in pointing tasks, which manifests as characteristic undershooting and re-approach patterns. Path efficiency (straight-line distance / actual path length) and direction-change count both capture how directly a player navigates. A scripted path that solves the geometry optimally will tend to be more efficient and have fewer sharp direction changes than a human trajectory that reflects continuous real-time re-aiming.

- **Centerline deviation.** This is a game-specific feature directly motivated by the Steering Law. In tunnel navigation tasks, the Steering Law predicts that humans maintain a non-zero, variable offset from the centerline as they continuously correct for motor noise. A fixed-offset or noise-injected synthetic path produces a different deviation distribution — not because it deviates more, but because the *pattern* of deviation (autocorrelation, distribution shape, coupling to local tunnel curvature) differs from the continuous-correction process a human uses.

- **Tremor-band power (8–12 Hz).** Physiological hand tremor in healthy adults occurs at 8–12 Hz (Elble & Koller, 1990) and is measurable in mouse cursor trajectories. It is an established feature in behavioral biometrics (Shen et al., 2013) precisely because it is a passive, involuntary signal that is difficult to suppress or fake. An agent that generates smooth Bézier-interpolated waypoints will produce no power in this band at all.

**Success criterion:** At least 15 of 25 features show a statistically significant difference between human and agent (Mann-Whitney U, BH-corrected q < 0.05), with at least one feature in each of the six groups above reaching significance. This confirms that the feature set is capturing real behavioral differences and is not dominated by one narrow signal.

**Current result:** 22 of 25 features are significant at q < 0.05, spanning all six groups. The largest effect sizes are in centerline deviation (Cliff's δ ≈ 0.98), curvature variability (δ = −0.73), and speed variability (δ = −0.68). The agent is faster on average (median 921 ms vs 1354 ms for humans) and markedly more uniform in speed. The success criterion is met.


### Eval 2 — ML Classifier

**Question:** Can a standard classifier reliably tell human from agent given those features?

Three classifiers (Logistic Regression, Random Forest, Gradient Boosting) are trained and evaluated with stratified 5-fold cross-validation on the 294 successful trajectories. Using three classifiers of different complexity guards against a result that depends on a specific model family.

**Success criterion:** At least one classifier achieves AUC ≥ 0.75 and balanced accuracy ≥ 0.70. This threshold is well above chance (0.5) but does not demand perfection — it means the feature set carries enough signal to reliably separate the two groups even under cross-validation. A result below this threshold would indicate the agent's trajectories are already difficult to distinguish from humans in feature space.

**Current result:** All three classifiers far exceed the criterion. Random Forest achieves AUC = 1.000 and accuracy = 99.7%. Logistic Regression reaches AUC = 0.999. The classifier makes only 1–3 errors across 294 trajectories.

Top feature importances (Random Forest):

| Feature | Importance |
|---------|-----------|
| centerline_dev_mean | 0.229 |
| centerline_dev_std | 0.221 |
| dt_mean | 0.110 |
| speed_std | 0.079 |

These four features alone account for ~64% of total importance. Centerline deviation dominates, which tells us that the agent's spatial behavior inside the tunnel is its most visible tell.

### Eval 3 — State-Space Density

**Question:** Do human and agent trajectories occupy the same regions of (position, speed, curvature) space at the level of individual events?

Rather than summarizing each trajectory into scalars, this eval treats every mouse event as a point in a 4D state space and compares the per-event distributions between the two groups. KL divergence measures how different the distributions are; occupancy IoU measures what fraction of the state space is visited by both sources. This captures distributional differences that per-trajectory statistics miss — for example, whether the agent visits the same spatial regions of the tunnel at the same speeds as humans.

**Success criterion:** KL divergence > 0.1 and occupancy IoU < 0.90 in at least two of the four 2D projections (x-y, speed-curvature, x-speed, y-speed). Values in this range confirm that the distributions are meaningfully different and that the eval can discriminate between sources. A result where all KL values are near zero and IoU is near 1.0 would indicate the per-event state distributions are indistinguishable.

**Current result:**

| Projection | KL divergence | Occupancy IoU |
|-----------|--------------|--------------|
| x-y (spatial) | 5.715 | 0.376 |
| y-speed | 0.783 | 0.527 |
| speed-curvature | 0.617 | 0.448 |
| x-speed | 0.566 | 0.589 |

All four projections exceed the success criterion. The spatial KL of 5.72 is especially striking — despite playing the same tunnels, the agent occupies a fundamentally different spatial distribution within them. The spatial IoU of 0.376 means only 37.6% of spatial bins are shared between the two groups.


### Eval 4 — Raw Sequence Classifier

**Question:** Is the distinction learnable from the raw event sequence without any hand-crafted features?

A small 1D-CNN is trained end-to-end on sequences of (x_norm, y_norm, speed, acceleration, curvature) with stratified 5-fold cross-validation. This tests whether temporal ordering carries information beyond summary statistics, and whether a detector with no domain knowledge can still separate the two groups. If hand-crafted features in Eval 2 were somehow cherry-picked to favor separability, this eval provides an independent check.

**Success criterion:** Raw-sequence AUC ≥ 0.75. A result above this threshold confirms that the temporal structure of the trajectories is inherently different between sources, independent of any particular feature engineering choice.

**Current result:** AUC = 0.985, accuracy = 90.5%, F1 = 0.873. The CNN separates human from agent without any hand-crafted features, confirming the signal is present in the raw sequence and is not an artifact of the feature selection in Eval 1. The success criterion is met.


## Summary

| Eval | Success criterion | Current result | Met? |
|------|-----------------|----------------|------|
| Feature tests (# significant) | ≥ 15 / 25 features, all groups | 22 / 25, all groups | Yes |
| ML classifier AUC | ≥ 0.75 | 1.000 (RF), 0.999 (LR) | Yes |
| State-space KL (≥ 2 projections) | KL > 0.1, IoU < 0.90 | Min KL = 0.57, Max IoU = 0.59 | Yes |
| Raw sequence AUC | ≥ 0.75 | 0.985 | Yes |

All four evals confirm the same conclusion: the `visual agent`'s trajectories are clearly distinguishable from human trajectories. The eval infrastructure is in place and can produce a definitive answer as new agent variants are introduced.

## References

Accot, J., & Zhai, S. (1997). Beyond Fitts' law: Models for trajectory-based HCI tasks. In *Proceedings of the ACM SIGCHI Conference on Human Factors in Computing Systems (CHI '97)* (pp. 295–302). ACM.

Elble, R. J., & Koller, W. C. (1990). *Tremor*. Johns Hopkins University Press.

Fitts, P. M. (1954). The information capacity of the human motor system in controlling the amplitude of movement. *Journal of Experimental Psychology*, 47(6), 381–391.

Flash, T., & Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688–1703.

Gamboa, H., & Fred, A. (2004). A behavioral biometric system based on human-computer interaction. In *Proceedings of SPIE*, 5404, 381–392.

Pusara, M., & Brodley, C. E. (2004). User re-authentication via mouse movements. In *Proceedings of the 2004 ACM Workshop on Visualization and Data Mining for Computer Security (VizSEC/DMSEC '04)* (pp. 1–8). ACM.

Shen, C., Cai, Z., Guan, X., Du, Y., & Maxion, R. A. (2013). User authentication through mouse dynamics. *IEEE Transactions on Information Forensics and Security*, 8(1), 16–30.

Zheng, N., Paloski, A., & Wang, H. (2011). An efficient user verification system via mouse movements. In *Proceedings of the 18th ACM Conference on Computer and Communications Security (CCS '11)* (pp. 139–150). ACM.
