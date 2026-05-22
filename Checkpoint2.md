# Checkpoint 2

## Summary

Checkpoint 1 proposed a second game family — "Hover-to-Find" — to test reasoning under uncertainty. We first tried to push this further with a **Wason-style rule-discovery game**: the player saw one ✓ and one ✗ example of a hidden rule, constructed up to three test triples from a 9-piece palette to probe the rule with ✓/✗ feedback, then committed one final triple. The intent was to use the player's hypothesis-revision trajectory itself as the behavioral signal.

We abandoned this route. The probe-and-refine loop turned out too cognitively heavy for a CAPTCHA interaction — holding a hypothesis, designing a discriminating test, interpreting feedback, and updating, all before committing, makes the median session run far past CAPTCHA-acceptable length. We pivoted to a family of **single-commit reasoning games**: the player watches four objects, silently reasons about which one breaks a shared rule, and commits with one click. The reasoning load is preserved; the interaction stays short.

Across seven versions of this single-commit family, **the Claude Code agent solves every variant with accuracy and click count comparable to a human, differing only in latency.** This is a contrast to Trace-the-Tunnel (Checkpoint 1), where agent and human trajectories were trivially distinguishable (AUC = 1.0). Within scope, single-commit reasoning games do not produce a behavioral gap on outcome-level metrics — the gap, if any, lives in the *structure* of how the agent samples and computes between observations.

---

## Why reasoning games

The shift from motor control (Checkpoint 1) to reasoning is motivated by a specific question: what kinds of cognition stay hard for an LLM agent at the behavioral level, even when it produces the right final answer? Two perceptual systems suggest where the gap might appear.

- **Intuitive physics.** Humans rapidly infer causal properties from observed motion (Baillargeon 1987; Smith & Vul 2013). When a ball floats instead of falling, a human perceives something is wrong without enumerating physics laws.
- **Animacy and social attribution.** Humans automatically perceive intention and social roles in moving abstract shapes (Heider & Simmel 1944; Scholl & Tremoulet 2000; Johansson 1973). These percepts are fast, automatic, and pre-verbal — what dual-process theory (Kahneman 2011) calls System 1. An agent operates almost entirely in System 2.

From these we derived two hypotheses:

- **H1 — motion-pattern perception is hard for screenshot-driven agents.** Claude Code does not perceive video; it captures discrete screenshots and reasons over each still in language. To recover motion at all, it must capture multiple stills, hold them in working context, and explicitly compute frame-to-frame change. A human just watches.
- **H2 — implicit social attribution is hard to articulate.** Humans perceive "shy", "leader", "lazy" directly from motion. An agent has no such percept; it must translate the scene into explicit computation and map the result onto a personality label.

**On latency as a metric.** We deliberately exclude reasoning latency from the evaluation. Latency would trivially separate populations, but the gap is dominated by MCP tool-call overhead and token generation — a hardware- and harness-dependent signal, not a reasoning signal. A CAPTCHA whose discriminator is "the agent is slow" is a moving target. The same exclusion applied to Checkpoint 1.

---

## Game family: Reasoning About Motion Rules

### Core mechanic

Four objects move on a canvas. Three follow a shared rule; one follows a different rule (except in v7, see below). The player watches for as long as they wish, forms a hypothesis, and clicks the outlier to commit. There is no time pressure and no penalty for missed clicks on empty space. Median human completion time across versions is under 5 seconds.

**Mouse trajectories are not a primary metric for this family.** In practice both populations barely move the cursor — they watch the animation and click their chosen option. Trajectory analysis was the point of Checkpoint 1; re-using it here would just rehash that result.

### Version history

| Version | Title | Objects | Rule difference | How the player observes |
|---------|-------|---------|-----------------|-------------------------|
| v1 | Spot the Different One | Irregular polygons | Roll vs. slide | Click shape to trigger preview sweep |
| v2 | Spot the Different One | Squares | Roll vs. slide | Click shape to trigger preview sweep |
| v3 | Spot the Different One | Circles | Roll vs. slide | Click shape to trigger preview sweep |
| v5 | Race Watcher | Horizontal racers | Kinematic profile (ease-in, ease-out, pause, reverse) | Watch live animation |
| v6 | Physics Room | Bouncing balls | Physics rule (floater, heavy, hyper, drifter) | Watch live physics sim |
| v7 | Pick the Personality | Colored dots | Behavioral personality (leader, follower, hyper, lazy) | Watch 10 s live animation; click named personality |

**v1–v3 (Roll-or-Slide).** Three shapes slide along an axis; one rolls (rotates as it moves). Shape variants test whether visual texture is necessary to distinguish rolling from sliding. To prevent an agent from solving by coordinate comparison, each shape's amplitude is jittered by ±5% — below the human perceptual threshold, but enough that no two shapes share a canonical trajectory.

**v5 (Race Watcher).** Four dots race along horizontal tracks; one has a different kinematic profile.

**v6 (Physics Room).** Four balls under Newtonian physics; one's parameters depart from the baseline (zero gravity, 3× gravity, near-perfect elasticity, or constant lateral drift).

**v7 (Pick the Personality).** Four dots, each with a *distinct* personality: leader heads straight up, follower springs after the leader, hyper bursts left-right, lazy drifts slowly upward. The player is told one personality (e.g., "Find the FOLLOWER") and clicks the matching dot.

v7 was motivated by a structural concern about v1–v6: every prior version had three identical objects and one outlier, so an agent could in principle solve them *without understanding the prompt at all* — just by finding the object that behaves differently from the other three. v7 closes that shortcut. Every dot behaves differently, so the player must actually map the named label to its motion signature.

---

## Results

**v2 and v5** were piloted and dropped — the rule difference is visually unambiguous, every player answers in one click, and there is no process to compare.

**v6** is the most-played version (16 human sessions, several agent sessions). Outcomes are flat: both populations watch for a few seconds and click once. Floater and drifter are easy for everyone; hyper and heavy produce more errors. The recorded hover events look rich in volume but the cursor is mostly parked while the player watches — no active scanning between balls.

**v7** is where we ran the agent most carefully to test H2 (label-to-motion mapping). Three findings:

1. **Accuracy depends on what the prompt reveals.** Given only the target label ("Find the LAZY one"), the agent makes meaningful mistakes — often confusing semantically adjacent personalities (picking lazy when asked for follower, since both have low forward motion). Given all four definitions upfront, accuracy is much higher. The agent reasons better by elimination than by anchoring on one label.

2. **Accuracy approaches 100% across rounds.** Once the agent has seen each personality in motion at least once, the label-to-motion mapping stabilizes. First-round mistakes are calibration, not capability.

3. **The agent solves it numerically, not perceptually.** The agent's own report describes its method: capture ~12 s of canvas pixels at 80 ms intervals, find each dot's centroid per frame, derive per-dot velocity `(dx, dy)` and pairwise lagged cross-correlations `corr(A_vel(t), B_vel(t+lag))`. It scores each personality against a numerical signature:
   - **leader** — highest leadership score (motion precedes others at +lag).
   - **follower** — highest single-pair lagged correlation at lag ≥ 1.
   - **hyper** — highest total path length.
   - **lazy** — lowest total path length.

   The agent reported one calibration error: it initially defined follower as "minimum leadership score" (which conflates lazy with follower), then switched to "max single-pair lagged correlation" after getting a round wrong. It also initially equated hyper with high-variance speed, which pointed to the follower's catch-up jumps; the corrected definition (max total path length) is what it has used since.

**On H2.** The hypothesis holds *partially*. The "no prior intuition for what 'lazy' looks like" prediction is supported by finding (1): with the label in isolation, the agent confuses semantically adjacent personalities. But the gap closes (finding 2) once the agent can compare and contrast — it builds its own numerical signature for each label and disambiguates reliably. The agent does not need to *understand* "lazy" semantically; it only has to map the label to a computable motion statistic. H2 survives as a **cold-start phenomenon, not a persistent capability gap**.

---

## What this tells us and what comes next

For Trace-the-Tunnel, the agent's process was categorically different from a human's and the difference was measurable immediately. For the single-commit reasoning family, the agent has no shortcut to bypass *the answer*, but it has a strong shortcut to bypass *the perception*: capture a frame stack, compute centroids, velocities, and cross-correlations, and reduce the task to numerical comparison. v7 makes this especially visible because the agent's own report enumerates the formulas.

Both populations produce comparable click economies, but they reason in qualitatively different modes — a human watches and pattern-matches, the agent samples and computes. That difference is not in the click, which is why it does not show up in outcome data, but it should be visible in the *timing and structure* of what the agent does between observations: multi-second capture pauses, no cursor activity during analysis, decisions arriving all at once rather than building up. The next step is a version that forces more visible intermediate decisions — short enough for a CAPTCHA, but structured so each observation is a logged data point and the sample-and-compute loop has nowhere to hide.

---

## References

Baillargeon, R. (1987). Object permanence in 3.5- and 4.5-month-old infants. *Developmental Psychology*, 23(5), 655–664.

Heider, F., & Simmel, M. (1944). An experimental study of apparent behavior. *American Journal of Psychology*, 57(2), 243–259.

Johansson, G. (1973). Visual perception of biological motion and a model for its analysis. *Perception & Psychophysics*, 14(2), 201–211.

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Scholl, B. J., & Tremoulet, P. D. (2000). Perceptual causality and animacy. *Trends in Cognitive Sciences*, 4(8), 299–309.

Smith, K. A., & Vul, E. (2013). Sources of uncertainty in intuitive physics. *Topics in Cognitive Science*, 5(1), 185–199.
