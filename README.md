# **Designing a Deployable Game-Based CAPTCHA Under Adversarial Pressure**

## **Background and Motivation**

Classical CAPTCHAs have effectively collapsed under multimodal AI. Automated traffic surpassed human traffic in 2024, and modern solver systems achieve \>95% success rates, with some reCAPTCHA bypasses approaching 99.8%. Even safeguards at the agent level are fragile: LLM agents can be induced to solve CAPTCHAs through prompt injection. Any CAPTCHA framed as static recognition has been overtaken by vision-language models.

Two main defenses have emerged. Behavioral biometrics analyzes interaction patterns (e.g., mouse movement), but learned imitators achieving high bypass rates. Adversarial puzzle design, explored in benchmarks like OpenCaptchaWorld and NextGen-CAPTCHAs, constructs tasks targeting model weaknesses (e.g., spatial or temporal reasoning). These can temporarily restore separation, but outcome-based metrics collapse as models improve. Meanwhile, interactive benchmarks such as ARC-AGI-3 show a much larger human–AI gap—but are far too slow for deployment.

A consistent limitation across these approaches is that verification is based on outcomes. However, empirical evidence suggests the more robust distinction lies in *how* a task is solved: humans explore, revise, and recover from mistakes in ways AI systems do not reliably replicate.

## **Core Idea**

This project proposes a CAPTCHA system that evaluates **how a task is solved, rather than whether it is solved**.

Users interact with short (≤10s), procedurally generated micro-games. The system extracts two complementary signals from each session. The first is the **interaction trajectory**, capturing low-level behavior such as timing, movement, and action sequences. The second is the **procedural trajectory**, capturing how the user explores the game’s state space, including which states are visited, how strategies evolve, and how the user responds to unexpected outcomes.

The key hypothesis is that while interaction signals alone are vulnerable to imitation, and outcome-based verification collapses as models improve, the *combination* of interaction and procedural trajectories captures a more durable distinction. Human users exhibit structured exploration, rapid adaptation, and hypothesis-driven behavior, whereas current AI agents tend to follow brittle plans, exhibit low exploration entropy, and fail to update effectively in response to feedback.

## **Design Constraints**

The system is shaped by five constraints:

* **Latency:** Tasks must be completed in under \~10 seconds.  
* **Statelessness:** Each challenge must independently verify the user.  
* **Adversarial robustness:** Attackers can deploy LLM agents, imitation models, and large-scale parallelism.  
* **Cost:** Verification must remain cheap.  
* **Accessibility:** Interaction cannot rely solely on fine motor precision or visual acuity.

These constraints force the design toward simple tasks with rich behavioral signals, rather than complex puzzles.

## **Game Design**

We propose the following set of micro-game families, each targeting known AI weaknesses.

* **Rule Discovery Tasks:** Simple environments with hidden mechanics require users to probe and revise hypotheses. Humans explore and adapt quickly; AI agents tend to overcommit to initial assumptions.  
* **Physical Interaction Tasks:** Lightweight physics-based challenges expose gaps in intuitive causal reasoning and fine-grained control.  
* **Temporal Consistency Tasks:** Users identify subtle inconsistencies in short animations, requiring cross-frame reasoning that remains difficult for multimodal models.

These tasks should be easy to complete but generate interaction traces that reveal differences in the process.

## **Environment Design**

The system follows a standard pipeline:

1. **Generator** creates diverse, non-replayable challenges.  
2. **Interaction Logger** records full action sequences (timing, movement, state transitions).  
3. **Feature Extraction** computes signals such as:  
   * Exploration entropy  
   * Error recovery latency  
   * Trajectory curvature  
   * Strategy revision frequency  
4. **Classifier** estimates the likelihood of human-like behavior.

## **Research Questions and Evaluation**

We organize the project around four questions, each paired with the experiment that would answer it. We aim to collect trajectories from 8–12 participants (team plus peer volunteers) and run two frontier LLM agents (one OpenAI, one Anthropic) through OpenCaptchaWorld on the same procedurally generated instances. The small sample limits population-level claims but is sufficient to detect large effect sizes; we report bootstrapped confidence intervals throughout.

**RQ1 — Does trajectory information add separation beyond outcome alone?** We train a binary classifier (logistic regression, gradient boosting) on human-vs-agent sessions and compare its AUC against outcome-only features.   
**RQ2 — Are interaction and procedural signals complementary?** A feature-group ablation: interaction-only, procedural-only, and combined classifiers. A non-trivial gain from combination justifies the two-signal design; a null result tells us which signal carries the real information and simplifies the system.  
**RQ3 — Is the gap robust to a basic imitation attack?** We train a supervised imitator on our human trajectories, pair it with an LLM solver, and report both its classifier-evasion rate and its pass rate. High evasion at the cost of solving means the system holds; cheap evasion without solving cost localizes which features need replacement.  
**RQ4 — Which game family produces the largest gap?** Per-family classifier AUC, pass@1 gap, and median human completion time. Identifies the minimal viable family set and exposes whether different families exploit outcome vs. trajectory axes differently.  
**Cost as a first-class metric.** We report API dollars and reasoning tokens per agent attempt against seconds per human attempt across all four studies.

**Success criteria.** The project succeeds if (1) human pass@1 ≥ 95% with median completion \<10s, (2) trajectory-classifier AUC ≥ 0.85 against frontier baselines, (3) RQ2 shows non-trivial complementarity, and (4) the imitator in RQ3 cannot evade the classifier and solve at unmodified-solver cost. 
