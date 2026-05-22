# Checkpoint 2

## Summary

We implemented and iterated on the second game family described in Checkpoint 1 — a class of **pure reasoning games** in which the player must watch four objects, reason about their behavior, and identify the one that follows a different rule. This is a deliberate contrast to Trace-the-Tunnel: where Checkpoint 1 tested *motor control* (dragging along a continuous path), Checkpoint 2 tests *logical reasoning under observation* (deduce from what you see, then commit to a single decision).

Unlike Trace-the-Tunnel, where agent trajectories were trivially distinguishable from human ones (AUC = 1.0 on all four evaluations), the reasoning game family poses a harder challenge: **the Claude Code agent solves every version with accuracy and click count comparable to a human, differing only in latency.** This is a meaningful intermediate result — it tells us the agent reasons about these games in a way that is qualitatively similar to how humans do, and points us toward the behavioral signals that remain to be analyzed.

## Recap: Where We Left Checkpoint 1

| Eval | Criterion | Result |
|------|-----------|--------|
| Feature tests | ≥ 15 / 25 significant, all groups | 22 / 25 ✓ |
| ML classifier AUC | ≥ 0.75 | 1.000 (RF) ✓ |
| State-space KL | KL > 0.1, IoU < 0.90 in ≥ 2 projections | Min KL = 0.57, Max IoU = 0.59 ✓ |
| Raw CNN AUC | ≥ 0.75 | 0.985 ✓ |

For Trace-the-Tunnel, all four evals confirmed the agent's trajectories were far outside the human distribution. The dominant tell was centerline deviation: the agent hugged the tunnel centerline with near-zero variance, a behavior that no human produced. The underlying reason is structural — a human's answer to "drag through a tunnel" is an embodied motor program that accumulates noise, corrects in real time, and couples speed to curvature. An agent's answer is to solve the geometry and replay waypoints. These are categorically different processes.

The planned H2 game (originally "Hover-to-Find") required too much deliberate effort per round to be viable as a CAPTCHA. We pivoted to a game family that is quick for humans but still forces *causal reasoning*: the player must infer an unobserved physical rule from observed motion.

---

## Why Reasoning Games? Theoretical Motivation

The shift from motor control (Checkpoint 1) to reasoning (Checkpoint 2) is motivated by a specific hypothesis about what makes human cognition hard to replicate at the behavioral level.

### Implicit vs. explicit cognition

Humans solve many perceptual and social tasks through processes that are fast, automatic, and largely pre-verbal — what dual-process theory (Kahneman, 2011) calls System 1. An agent that reasons in natural language operates almost entirely in a deliberate, explicit mode (System 2). A game that requires *implicit* reasoning — pattern recognition that happens below the threshold of articulation — may therefore create an asymmetry that games demanding explicit logical inference do not.

### Causal reasoning about physical rules

The first dimension we targeted is **intuitive physics** — the ability to rapidly infer unobserved causal properties from observed motion. Humans are fast and accurate at this: Baillargeon (1987) demonstrated that infants as young as 3.5 months form expectations about physical laws from brief observations. Smith & Vul (2013) showed that human predictions about physical dynamics are well-modeled by approximate probabilistic simulation — a process that runs quickly and does not require articulated rules. When a ball floats instead of falling, a human immediately perceives something is wrong, without needing to enumerate physics laws.

This is distinct from an LLM agent's approach, which must observe a screenshot, describe what it sees in language, compare that description against its knowledge of physics, and then identify the violation. The process is the same in outcome but different in structure: implicit vs. explicit, simultaneous vs. sequential, perceptual vs. inferential.

### Social and emotional attribution

A second, stronger dimension is **animacy perception** and **social attribution** — the human tendency to automatically perceive intention, emotion, and social roles in moving abstract shapes. Heider & Simmel (1944) showed that people watching two triangles and a circle move around a box spontaneously describe a social narrative ("the big triangle is chasing the small one, which is scared"). This attribution is involuntary and immediate — observers cannot easily suppress it. Scholl & Tremoulet (2000) reviewed extensive evidence that animacy detection is a dedicated perceptual module, triggered by specific motion cues (self-propulsion, contingency, goal-directedness), not a post-hoc inference.

Johansson (1973) established that humans recognize biological motion from just twelve point-lights attached to body joints, within 200 ms, even when the display is otherwise meaningless noise. This percept is specific to biological motion: scrambled point-lights produce no such recognition. The implication is that humans have a specialized neural pathway for detecting the motion signatures of living, intentional agents — a pathway that did not evolve for abstract stimuli but generalizes to them.

The key claim for our project: **an LLM agent must reason *about* social and emotional cues, while a human *perceives* them directly.** If the task requires tracking a "shy" entity — one that avoids others and hesitates — a human identifies it through immediate social pattern matching; an agent must explicitly compute which blob shows avoidance behavior and which shows hesitation, then maintain that assignment over time. These are not equivalent cognitive operations, even when they produce the same label.

### What this implies for CAPTCHA design

The behavioral signature we are looking for is not accuracy (both populations can answer correctly) but the *process trace*: how long the player looked before deciding, which objects they examined, whether their cursor anticipated movement rather than following it, and how quickly they re-identified a target after an interruption. These process features are where implicit and explicit cognition diverge, even when the final decision converges.

This is consistent with the failure mode we observed: the agent matches human accuracy and click count in the Physics Room (v6) because the rule violations are salient enough to be reasoned about explicitly. The games that most strongly favor a human-vs-agent distinction should be those where the cue is subtle enough that it requires implicit, holistic pattern recognition — specifically, social and emotional cues encoded in continuous motion trajectories.


## The New Game: Reasoning About Motion Rules

### Core mechanic

Four objects move simultaneously on a canvas. Three follow the same underlying rule; one follows a different rule. The player watches for as long as they wish, forms a hypothesis about which object violates the shared rule, and then clicks it to commit. There is no penalty for watching longer, and no penalty for a missed click on empty space.

The game is a reasoning task, not a motor task: the challenge is to correctly infer which rule is being broken, not to execute a precise physical movement. Both humans and agents face the same epistemic problem — they can only observe the objects' behavior, not inspect the underlying parameters.

This design satisfies the three constraints from Checkpoint 1:

1. **No reaction time.** The game is not timed. Latency between perception and action is a property of the agent's MCP loop, not of its reasoning, so we exclude it from all evaluations.
2. **Sequential decisions.** The player makes at least one and typically several observations before committing. We record the full hover path and every click.
3. **Viable CAPTCHA.** Median human completion time across all versions is under 5 seconds. The UI is a single canvas element.

### Version history

We developed six versions, each varying the type of rule difference and the amount of information available per observation:

| Version | Title | Objects | Rule difference | How the player observes |
|---------|-------|---------|-----------------|-------------------------|
| v1 | Spot the Different One | Irregular polygons | Roll vs. slide | Click shape to trigger a preview sweep |
| v2 | Spot the Different One | Squares | Roll vs. slide | Click shape to trigger a preview sweep |
| v3 | Spot the Different One | Circles | Roll vs. slide | Click shape to trigger a preview sweep |
| v5 | Race Watcher | Horizontal racers | Motion profile (ease-in, ease-out, pause, reverse) | Watch a live animation; no reveals needed |
| v6 | Physics Room | Bouncing balls | Physics rule (floater, heavy, hyper, drifter) | Watch a live physics simulation |
| — | Keep Following *(excluded)* | Animated blobs | Social trait (shyness) | Continuous cursor tracking for 9 s |

**v1–v3 (Roll-or-Slide).** Three shapes slide back and forth along a fixed axis; one rolls (it rotates as it moves). The three shape variants test whether visual texture (the spinning polygon) is necessary for the player to distinguish rolling from sliding, or whether the motion pattern alone suffices. All three versions share the same mechanic: the player clicks any shape to trigger a single animated preview sweep, watches the motion, and forms a judgment. To prevent an agent from solving the task by coordinate comparison — finding the outlier as the shape whose sweep distance is furthest from the mean — each shape's amplitude is independently jittered by ±5%. This is below the human perceptual threshold for amplitude difference, so humans are unaffected, but it means no two shapes share a canonical trajectory. The agent must reason about the *type* of motion (rotation present or absent) rather than just comparing peak displacement values.

**v5 (Race Watcher).** Abstracts away shape. Four colored dots race along horizontal tracks. The rule difference is kinematic: ease-in (slow start, fast finish), ease-out (fast start, slow finish), pause (rush then hold then finish), or reverse (overshoot then backtrack). The player watches the live animation and clicks when ready.

**v6 (Physics Room).** Four balls undergo full Newtonian physics (gravity, elasticity, wall collisions, drag). One ball's physics departs from the shared baseline in one of four ways: *floater* (zero gravity), *heavy* (3× gravity, near-inelastic), *hyper* (near-perfect elasticity), or *drifter* (constant lateral force). Players watch the live simulation and click when they have made up their mind.

**Keep Following (implemented, excluded from data collection).** This game takes a different approach: rather than inferring a physical rule, the player must attribute a *social trait* to one of four abstract blobs. One blob is "shy" — it actively avoids proximity to the other three and randomly hesitates mid-movement. The player's task is to identify the shy blob by watching the group, then continuously keep their cursor near it for nine seconds. A mid-round interruption (a pulsing button the player must click) resets the visual scene and tests whether the player can re-identify the shy blob from scratch.

The game is theoretically the most promising for human-vs-agent discrimination, precisely because it requires social attribution rather than physical reasoning. Per the Heider-Simmel framework, humans perceive shyness in abstract moving shapes *directly* — the behavioral signature (avoidance, hesitation) is recognized as a social trait without deliberate analysis. An agent must instead explicitly observe blob positions across frames, compute which blob's velocity vector points away from its neighbors, identify the hesitation pauses, and then issue continuous cursor-positioning commands — all in a single session. This is a fundamentally different cognitive operation.

We excluded Keep Following from the current data collection for two reasons. First, it conflates two separate skills — social attribution (identifying the shy blob) and motor tracking (keeping the cursor near it) — making it difficult to attribute any human-agent difference to one specific capability. Second, the 9-second continuous tracking session is longer than a viable CAPTCHA interaction. Both issues are solvable by design iteration: separating the identification step from the tracking step, or capping the tracking phase at 3–4 seconds. We include it here as the strongest candidate for future work.

---

## Current Results

For most versions, the data is thin by design. v2 and v5 were piloted and abandoned quickly because the rule difference is visually unambiguous — every player, human or agent, answers in a single click with no deliberation. There is no process to compare. v6 (Physics Room) is the most-played version with 16 human sessions and several agent sessions, but the data it produces is similarly flat: both populations watch for a few seconds and click once. The recorded hover events look rich in volume but on inspection the cursor is mostly parked at the edge of the canvas while the player watches the simulation — not actively scanning between balls. The outcome data (correct or not, time to click) confirms that floater and drifter are easy for everyone while hyper and heavy produce more errors, but this is visible just by playing the game and does not require a dataset to establish.

---

## What This Tells Us and What Comes Next

The result from Checkpoint 2 is a negative one, and it is informative. For Trace-the-Tunnel, the agent's process was categorically different from a human's and the difference was measurable immediately. For the reasoning game family, the agent has no shortcut: it observes the same animation and makes the same judgment a human would, producing the same outcome with the same click economy. The games are working as reasoning games — both populations reason correctly — but that also means they are not yet useful as CAPTCHAs, because there is no behavioral residual to measure beyond the click itself.

The clearest path forward is to design a version that forces more visible intermediate decisions — short enough for a CAPTCHA context but structured so that each observation is a logged data point, not just the final click. The agent's decision sequence is likely to be more systematic and less reactive than a human's, and that difference should be measurable even with a small number of sessions.

---

## References

Accot, J., & Zhai, S. (1997). Beyond Fitts' law: Models for trajectory-based HCI tasks. *CHI '97*, 295–302.

Elble, R. J., & Koller, W. C. (1990). *Tremor*. Johns Hopkins University Press.

Fitts, P. M. (1954). The information capacity of the human motor system in controlling the amplitude of movement. *Journal of Experimental Psychology*, 47(6), 381–391.

Flash, T., & Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688–1703.

Shen, C., Cai, Z., Guan, X., Du, Y., & Maxion, R. A. (2013). User authentication through mouse dynamics. *IEEE Transactions on Information Forensics and Security*, 8(1), 16–30.

Baillargeon, R. (1987). Object permanence in 3.5- and 4.5-month-old infants. *Developmental Psychology*, 23(5), 655–664.

Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"? *Cognition*, 21(1), 37–46.

Heider, F., & Simmel, M. (1944). An experimental study of apparent behavior. *American Journal of Psychology*, 57(2), 243–259.

Johansson, G. (1973). Visual perception of biological motion and a model for its analysis. *Perception & Psychophysics*, 14(2), 201–211.

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Scholl, B. J., & Tremoulet, P. D. (2000). Perceptual causality and animacy. *Trends in Cognitive Sciences*, 4(8), 299–309.

Smith, K. A., & Vul, E. (2013). Sources of uncertainty in intuitive physics. *Topics in Cognitive Science*, 5(1), 185–199.
