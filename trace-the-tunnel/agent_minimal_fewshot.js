// agent_minimal_fewshot.js
//
// Few-shot human-mimicking trace algorithm for trace-the-tunnel.
//
// Design (derived from the 3 example screenshots in experiments/examples/):
//   Humans do NOT follow the wavy centerline. They produce a much smoother
//   path that "cuts corners" — riding the inside of every bend, only
//   straying from straight when a hard tunnel turn forces them to.
//
// Algorithm:
//   1. Read dense centerline + tunnel half-width from the live game.
//   2. Heavy Gaussian smoothing on the centerline (sigma scales with width).
//      This is the "corner-cut" — a low-pass filter on the wiggle.
//   3. Boundary safety: any smoothed point that would land >TUNNEL_W - margin
//      from the centerline is pulled back toward the nearest centerline pt.
//   4. Re-sample by arc length (≈4 px steps) to match human spacing.
//   5. Add micro-jitter (~0.5 px std) — humans aren't perfectly steady.
//   6. Dispatch real DOM mouse events on the canvas with realistic delays
//      (≈8–10 ms gaps, slight ease-in/ease-out), ~1.2–1.4 s total.
//
// Usage in the page (after window.__tunnelGame is loaded):
//   await window.__fewshot.playTunnel(tunnelId)

(() => {
  // ---- Geometry helpers ---------------------------------------------------

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  function gaussianKernel(sigma) {
    const radius = Math.max(1, Math.ceil(sigma * 3));
    const kernel = [];
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const w = Math.exp(-(i * i) / (2 * sigma * sigma));
      kernel.push(w);
      sum += w;
    }
    return kernel.map((w) => w / sum);
  }

  function smoothPath(points, sigma) {
    const kernel = gaussianKernel(sigma);
    const radius = (kernel.length - 1) / 2;
    const out = [];
    for (let i = 0; i < points.length; i++) {
      let sx = 0, sy = 0;
      for (let k = 0; k < kernel.length; k++) {
        const j = Math.min(points.length - 1, Math.max(0, i + k - radius));
        sx += points[j].x * kernel[k];
        sy += points[j].y * kernel[k];
      }
      out.push({ x: sx, y: sy });
    }
    // Pin endpoints — do NOT let smoothing drag start/end off the dots.
    out[0] = { x: points[0].x, y: points[0].y };
    out[out.length - 1] = { x: points[points.length - 1].x, y: points[points.length - 1].y };
    return out;
  }

  function nearestOnCenterline(p, centerline) {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < centerline.length; i++) {
      const d = dist(p, centerline[i]);
      if (d < bestD) { bestD = d; best = i; }
    }
    return { idx: best, point: centerline[best], distance: bestD };
  }

  // Pull any point that strays past the safe band back toward the centerline.
  function clampToTunnel(path, centerline, tunnelHalfWidth, margin) {
    const safe = tunnelHalfWidth - margin;
    return path.map((p) => {
      const n = nearestOnCenterline(p, centerline);
      if (n.distance <= safe) return p;
      const t = safe / n.distance;
      return {
        x: n.point.x + (p.x - n.point.x) * t,
        y: n.point.y + (p.y - n.point.y) * t,
      };
    });
  }

  function resampleByArcLength(path, stepPx) {
    if (path.length < 2) return path.slice();
    const out = [path[0]];
    let carry = 0;
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1], b = path[i];
      const seg = dist(a, b);
      let traveled = -carry;
      while (traveled + stepPx <= seg) {
        traveled += stepPx;
        const t = traveled / seg;
        out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
      }
      carry = seg - traveled;
    }
    const last = path[path.length - 1];
    if (dist(out[out.length - 1], last) > 0.5) out.push(last);
    return out;
  }

  function addJitter(path, stdPx) {
    // Skip first and last so we still hit the dots.
    const out = path.map((p, i) => {
      if (i === 0 || i === path.length - 1) return { ...p };
      // Box-Muller for Gaussian noise
      const u1 = Math.random() || 1e-9, u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const nx = r * Math.cos(2 * Math.PI * u2) * stdPx;
      const ny = r * Math.sin(2 * Math.PI * u2) * stdPx;
      return { x: p.x + nx, y: p.y + ny };
    });
    return out;
  }

  // Ease-in/ease-out cosine schedule: humans accelerate then decelerate.
  function easeSchedule(n, totalMs) {
    const delays = new Array(n);
    let prev = 0;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // Cosine ease maps t in [0,1] to a non-uniform progress, leaving
      // small steps near the ends and bigger spans (=> more time per step
      // when waypoints are uniform-arclen) in the middle.
      const eased = 0.5 - 0.5 * Math.cos(Math.PI * t);
      const ms = eased * totalMs;
      delays[i] = Math.max(0, ms - prev);
      prev = ms;
    }
    return delays;
  }

  // ---- Path generation ----------------------------------------------------

  function buildHumanPath(centerline, tunnelHalfWidth, opts = {}) {
    const sigma = opts.sigma ?? Math.max(8, tunnelHalfWidth * 0.55);
    const margin = opts.margin ?? 10;
    const stepPx = opts.stepPx ?? 4.0;
    const jitterPx = opts.jitterPx ?? 0.5;

    let path = smoothPath(centerline, sigma);
    path = clampToTunnel(path, centerline, tunnelHalfWidth, margin);
    path = resampleByArcLength(path, stepPx);
    path = addJitter(path, jitterPx);
    return path;
  }

  // ---- Event dispatch -----------------------------------------------------

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function canvasToClient(canvas, p) {
    const rect = canvas.getBoundingClientRect();
    return {
      clientX: rect.left + (p.x / 600) * rect.width,
      clientY: rect.top + (p.y / 350) * rect.height,
    };
  }

  function fire(canvas, type, p, button = 0, buttons = 1) {
    const c = canvasToClient(canvas, p);
    const ev = new MouseEvent(type, {
      bubbles: true,
      cancelable: true,
      view: window,
      button,
      buttons,
      clientX: c.clientX,
      clientY: c.clientY,
    });
    canvas.dispatchEvent(ev);
  }

  async function replay(path, totalMs) {
    const canvas = document.getElementById("game");
    const delays = easeSchedule(path.length, totalMs);
    fire(canvas, "mousedown", path[0]);
    for (let i = 1; i < path.length; i++) {
      await sleep(delays[i]);
      // Game ends as soon as we touch the red dot on a mousemove, so just
      // keep dispatching mousemoves the whole way.
      fire(canvas, "mousemove", path[i]);
      // If the game already finished (success or fail), stop.
      const st = window.__tunnelGame.getState();
      if (st === "done_success" || st === "done_fail") return st;
    }
    fire(canvas, "mouseup", path[path.length - 1]);
    return window.__tunnelGame.getState();
  }

  // ---- Public entry -------------------------------------------------------

  async function playTunnel(tunnelId, opts = {}) {
    const g = window.__tunnelGame;
    g.loadTunnel(tunnelId);
    // Wait one frame so the new tunnel is rendered + state is "ready".
    await sleep(60);

    const start = g.getStartPos();
    const end = g.getEndPos();
    const halfW = g.getTunnelWidth();
    const centerline = g.getCenterline();

    // Anchor the path to the actual start/end dots so mousedown registers.
    const anchored = [start, ...centerline.slice(1, -1), end];

    const totalMs = opts.totalMs ?? (1200 + Math.random() * 200);
    const path = buildHumanPath(anchored, halfW, opts);

    const finalState = await replay(path, totalMs);
    return {
      tunnelId,
      state: finalState,
      events: g.getSessionData().events.length,
      pathPoints: path.length,
      durationMs: totalMs,
    };
  }

  window.__fewshot = {
    playTunnel,
    buildHumanPath,
    smoothPath,
    clampToTunnel,
    resampleByArcLength,
  };
})();
