(function () {
  "use strict";

  // --- Tunnel generation ---------------------------------------------------
  // The tunnel is a chain of cubic Bezier segments for tighter, more frequent curves.
  // segments[] = [{p0,p1,p2,p3}, ...] — each is one cubic Bezier.

  const CANVAS_W = 600;
  const CANVAS_H = 350;
  const TUNNEL_WIDTH = 38; // half-width from centerline (narrower = harder)
  const DOT_RADIUS = 14;
  const NUM_SEGMENTS = 4; // number of Bezier segments chained together
  const SAMPLES_PER_SEG = 80; // centerline samples per segment

  let tunnelSeed = null;
  let segments = []; // [{p0,p1,p2,p3}, ...]
  let controlPoints = []; // flat list of all unique knot/control points (for JSON export)
  let centerlinePts = []; // sampled points along the full path

  // Simple seeded PRNG (mulberry32)
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generateTunnel(seed) {
    const rng = mulberry32(seed);
    const margin = 50;
    const usableH = CANVAS_H - 2 * margin;

    // Generate NUM_SEGMENTS+1 knot points spread left-to-right
    const knots = [];
    for (let i = 0; i <= NUM_SEGMENTS; i++) {
      const frac = i / NUM_SEGMENTS;
      knots.push({
        x: margin + frac * (CANVAS_W - 2 * margin),
        y: margin + rng() * usableH,
      });
    }

    // Build cubic Bezier segments with C1 continuity
    // For each segment, generate control handles that alternate up/down for S-curves
    segments = [];
    let prevHandle = null; // outgoing handle of previous segment (for continuity)

    for (let i = 0; i < NUM_SEGMENTS; i++) {
      const a = knots[i];
      const b = knots[i + 1];
      const mx = (a.x + b.x) / 2;
      const spanY = usableH * 0.7;

      let cp1;
      if (prevHandle) {
        // Mirror the previous outgoing handle for C1 continuity
        cp1 = { x: 2 * a.x - prevHandle.x, y: 2 * a.y - prevHandle.y };
      } else {
        // First segment: free handle, push up or down strongly
        const dir = (rng() < 0.5) ? -1 : 1;
        cp1 = {
          x: a.x + (b.x - a.x) * 0.3,
          y: a.y + dir * (0.3 + rng() * 0.5) * spanY,
        };
      }

      // Second control point: push opposite direction for S-shape
      const dir2 = (cp1.y < (a.y + b.y) / 2) ? 1 : -1;
      const cp2 = {
        x: a.x + (b.x - a.x) * 0.7,
        y: b.y + dir2 * (0.3 + rng() * 0.5) * spanY,
      };

      // Clamp control points to canvas bounds
      cp1.y = Math.max(margin * 0.3, Math.min(CANVAS_H - margin * 0.3, cp1.y));
      cp2.y = Math.max(margin * 0.3, Math.min(CANVAS_H - margin * 0.3, cp2.y));

      segments.push({ p0: a, p1: cp1, p2: cp2, p3: b });
      prevHandle = cp2; // store for next segment's continuity
    }

    // Build flat controlPoints array for export
    controlPoints = [];
    for (const seg of segments) {
      controlPoints.push(seg.p0, seg.p1, seg.p2, seg.p3);
    }

    // Sample centerline
    centerlinePts = [];
    for (const seg of segments) {
      for (let i = 0; i <= SAMPLES_PER_SEG; i++) {
        // Skip first point of subsequent segments to avoid duplicates
        if (centerlinePts.length > 0 && i === 0) continue;
        const t = i / SAMPLES_PER_SEG;
        centerlinePts.push(cubicBezier(seg, t));
      }
    }
  }

  function cubicBezier(seg, t) {
    const { p0, p1, p2, p3 } = seg;
    const u = 1 - t;
    return {
      x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
      y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
    };
  }

  function distToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function distToCenterline(px, py) {
    let minD = Infinity;
    for (let i = 0; i < centerlinePts.length - 1; i++) {
      const a = centerlinePts[i], b = centerlinePts[i + 1];
      const d = distToSegment(px, py, a.x, a.y, b.x, b.y);
      if (d < minD) minD = d;
    }
    return minD;
  }

  function isInsideTunnel(px, py) {
    return distToCenterline(px, py) <= TUNNEL_WIDTH;
  }

  // --- Tunnel pool (fetched from server) ------------------------------------

  let tunnelPool = [];   // [{tunnel_id, seed}, ...]
  let tunnelIndex = 0;   // which tunnel we're currently on
  let currentTunnelId = null;

  async function fetchTunnelPool() {
    try {
      const resp = await fetch("/api/tunnels");
      const data = await resp.json();
      tunnelPool = data.tunnels || [];
      console.log(`[tunnel] loaded ${tunnelPool.length} tunnels from server`);
    } catch (e) {
      console.warn("[tunnel] server unavailable, using random seeds");
      // fallback: generate some random tunnels locally
      for (let i = 0; i < 30; i++) {
        tunnelPool.push({ tunnel_id: i, seed: Math.floor(Math.random() * 2147483647) });
      }
    }
  }

  // --- Session / trajectory state ------------------------------------------

  let sessionId = crypto.randomUUID();
  let events = [];
  let state = "ready"; // ready | tracing | done_success | done_fail
  let failReason = null; // null | "out_of_bounds" | "released_early"
  let boundaryViolations = 0;
  let tracePoints = []; // [{x,y,inside}] for rendering the trail

  const startPos = () => segments[0].p0;
  const endPos = () => segments[segments.length - 1].p3;

  // --- Canvas setup --------------------------------------------------------

  const canvas = document.getElementById("game-canvas");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status-msg");
  const sessionDisplay = document.getElementById("session-display");

  function updateSessionDisplay() {
    const first3 = events.slice(0, 3).map(e =>
      `  {x:${e.x.toFixed(1)}, y:${e.y.toFixed(1)}, t:${e.timestamp.toFixed(1)}, type:${e.event_type}}`
    ).join("\n");
    const last3 = events.slice(-3).map(e =>
      `  {x:${e.x.toFixed(1)}, y:${e.y.toFixed(1)}, t:${e.timestamp.toFixed(1)}, type:${e.event_type}}`
    ).join("\n");

    sessionDisplay.textContent =
      `session_id: ${sessionId}\ntunnel_id: ${currentTunnelId ?? "?"} / ${tunnelPool.length}  seed: ${tunnelSeed}\nevents: ${events.length}\n--- first events ---\n${first3 || "  (none)"}\n--- last events ---\n${last3 || "  (none)"}`;
  }

  // --- Drawing -------------------------------------------------------------

  function draw() {
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // Helper: trace the full multi-segment path
    function tracePath() {
      ctx.moveTo(segments[0].p0.x, segments[0].p0.y);
      for (const seg of segments) {
        ctx.bezierCurveTo(seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y, seg.p3.x, seg.p3.y);
      }
    }

    // Draw tunnel fill
    ctx.save();
    ctx.lineWidth = TUNNEL_WIDTH * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(100, 120, 160, 0.25)";
    ctx.beginPath();
    tracePath();
    ctx.stroke();
    ctx.restore();

    // Draw tunnel border
    ctx.save();
    ctx.lineWidth = TUNNEL_WIDTH * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(80, 100, 140, 0.45)";
    ctx.beginPath();
    tracePath();
    ctx.stroke();
    ctx.restore();

    // Draw user trail
    if (tracePoints.length > 1) {
      for (let i = 1; i < tracePoints.length; i++) {
        const prev = tracePoints[i - 1];
        const cur = tracePoints[i];
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(cur.x, cur.y);
        ctx.lineWidth = 3;
        ctx.strokeStyle = cur.inside ? "rgba(100, 200, 255, 0.8)" : "rgba(255, 80, 80, 0.8)";
        ctx.stroke();
      }
    }

    // Draw start dot (green)
    const s = startPos();
    ctx.beginPath();
    ctx.arc(s.x, s.y, DOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = state === "ready" ? "#4caf50" : "#2e7d32";
    ctx.fill();

    // Draw end dot (red)
    const e = endPos();
    ctx.beginPath();
    ctx.arc(e.x, e.y, DOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = "#ef5350";
    ctx.fill();
  }

  // --- Event handling ------------------------------------------------------

  function getCanvasPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY,
    };
  }

  function recordEvent(type, pos) {
    const inside = isInsideTunnel(pos.x, pos.y);
    const evt = {
      x: pos.x,
      y: pos.y,
      timestamp: performance.now(),
      event_type: type,
      inside_tunnel: inside,
    };
    events.push(evt);

    if (type === "mousemove") {
      tracePoints.push({ x: pos.x, y: pos.y, inside });
      if (!inside) boundaryViolations++;
    }

    updateSessionDisplay();
  }

  function hitsDot(pos, dot) {
    return Math.hypot(pos.x - dot.x, pos.y - dot.y) <= DOT_RADIUS + 4;
  }

  canvas.addEventListener("mousedown", (e) => {
    if (state !== "ready") return;
    const pos = getCanvasPos(e);
    if (!hitsDot(pos, startPos())) return;

    state = "tracing";
    statusEl.textContent = "Tracing... stay inside the tunnel!";
    statusEl.className = "status tracing";
    recordEvent("mousedown", pos);
    tracePoints.push({ x: pos.x, y: pos.y, inside: true });
    draw();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (state !== "tracing") return;
    const pos = getCanvasPos(e);
    recordEvent("mousemove", pos);

    // Check if cursor left the tunnel → immediate fail, auto-retry same tunnel
    if (!isInsideTunnel(pos.x, pos.y)) {
      failReason = "out_of_bounds";
      autoSave();
      retryCurrentTunnel("Failed! You left the tunnel. Retrying...");
      return;
    }

    // Check if reached end dot
    if (hitsDot(pos, endPos())) {
      recordEvent("mouseup", pos);
      state = "done_success";
      failReason = null;
      statusEl.textContent = "Success! You reached the red dot.";
      statusEl.className = "status success";
      autoSave();
    }
    draw();
  });

  canvas.addEventListener("mouseup", (e) => {
    if (state !== "tracing") return;
    const pos = getCanvasPos(e);
    recordEvent("mouseup", pos);
    failReason = "released_early";
    autoSave();
    retryCurrentTunnel("Released too early. Retrying...");
  });

  function retryCurrentTunnel(msg) {
    state = "done_fail";
    statusEl.textContent = msg;
    statusEl.className = "status fail";
    draw();
    // Auto-retry after a short delay
    setTimeout(() => {
      sessionId = crypto.randomUUID();
      events = [];
      tracePoints = [];
      boundaryViolations = 0;
      failReason = null;
      state = "ready";
      generateTunnel(tunnelSeed);
      statusEl.textContent = `Ready. Tunnel ${currentTunnelId ?? "?"} of ${tunnelPool.length}. Press the green dot to begin.`;
      statusEl.className = "status ready";
      updateSessionDisplay();
      draw();
    }, 800);
  }

  // Prevent context menu on canvas
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // --- Auto-save trajectory ------------------------------------------------

  function buildSessionJSON() {
    const startTime = events.length > 0 ? events[0].timestamp : 0;
    const endTime = events.length > 0 ? events[events.length - 1].timestamp : 0;
    return {
      session_id: sessionId,
      tunnel_id: currentTunnelId,
      tunnel_seed: tunnelSeed,
      control_points: controlPoints,
      tunnel_width: TUNNEL_WIDTH,
      canvas_size: { width: CANVAS_W, height: CANVAS_H },
      source: "human",
      completed: state === "done_success",
      fail_reason: failReason,
      boundary_violations: boundaryViolations,
      start_time: startTime,
      end_time: endTime,
      duration_ms: endTime - startTime,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      events: events,
    };
  }

  function autoSave() {
    const data = buildSessionJSON();
    fetch("/api/save_trajectory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).catch(() => {}); // silent fail if server not running
  }

  // --- Controls ------------------------------------------------------------

  document.getElementById("btn-reset").addEventListener("click", () => {
    reset();
  });

  document.getElementById("btn-download").addEventListener("click", () => {
    const data = buildSessionJSON();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trajectory_${sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  function reset() {
    sessionId = crypto.randomUUID();
    events = [];
    tracePoints = [];
    boundaryViolations = 0;
    failReason = null;
    state = "ready";

    // Pick next tunnel from the pool (cycles through)
    if (tunnelPool.length > 0) {
      const entry = tunnelPool[tunnelIndex % tunnelPool.length];
      currentTunnelId = entry.tunnel_id;
      tunnelSeed = entry.seed;
      tunnelIndex++;
    } else {
      currentTunnelId = null;
      tunnelSeed = Math.floor(Math.random() * 2147483647);
    }

    generateTunnel(tunnelSeed);
    statusEl.textContent = `Ready. Tunnel ${currentTunnelId ?? "?"} of ${tunnelPool.length}. Press the green dot to begin.`;
    statusEl.className = "status ready";
    updateSessionDisplay();
    draw();
  }

  // --- Expose internals for Playwright agent access ------------------------
  // The agent can read these to know where to drag.
  window.__tunnelGame = {
    getSegments: () => segments,
    getControlPoints: () => controlPoints,
    getCenterline: () => centerlinePts,
    getTunnelWidth: () => TUNNEL_WIDTH,
    getStartPos: () => startPos(),
    getEndPos: () => endPos(),
    getState: () => state,
    getSessionData: () => buildSessionJSON(),
    getCanvasRect: () => canvas.getBoundingClientRect(),
    getTunnelId: () => currentTunnelId,
    getTunnelIndex: () => tunnelIndex,
    getTunnelPool: () => tunnelPool,
    loadTunnel: (id) => {
      // Allow agent/caller to jump to a specific tunnel by id
      const entry = tunnelPool.find(t => t.tunnel_id === id);
      if (entry) {
        currentTunnelId = entry.tunnel_id;
        tunnelSeed = entry.seed;
        sessionId = crypto.randomUUID();
        events = [];
        tracePoints = [];
        boundaryViolations = 0;
        failReason = null;
        state = "ready";
        generateTunnel(tunnelSeed);
        draw();
        return true;
      }
      return false;
    },
  };

  // --- Init ----------------------------------------------------------------
  fetchTunnelPool().then(() => reset());
})();
