// trace-the-tunnel solver — install onto window.__solver
// Inject by reading this file and passing its contents to the page
// inside an IIFE wrapper via browser_evaluate.
// Public surface (after injection):
//   window.__solver.runOnce(tunnelId, params)
//   window.__solver.runMany(tunnelId, N, baseSeed)
//   window.__solver.runAll(perTunnel, baseSeed)
//   window.__solver.sampleParams(seed)
//   window.__solver.saveSession()
(() => {
  const S = (window.__solver = window.__solver || {});

  S.sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  S.makeRng = (seed) => {
    let t = seed >>> 0;
    return () => {
      t |= 0; t = (t + 0x6d2b79f5) | 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  };

  S.readField = function () {
    const canvas = document.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const data = ctx.getImageData(0, 0, W, H).data;
    const tunnel = new Uint8Array(W * H);
    let gx = 0, gy = 0, gn = 0, rx = 0, ry = 0, rn = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const isTunnel = Math.abs(r - 80) < 40 && Math.abs(g - 100) < 40 && Math.abs(b - 144) < 50 && b > r;
        const isGreen = g > 130 && r < 120 && b < 130;
        const isRed = r > 170 && g < 130 && b < 130;
        if (isTunnel || isGreen || isRed) tunnel[y * W + x] = 1;
        if (isGreen) { gx += x; gy += y; gn++; }
        if (isRed) { rx += x; ry += y; rn++; }
      }
    }
    if (!gn || !rn) throw new Error('no green/red dot found');
    return {
      W, H, tunnel,
      green: [Math.round(gx / gn), Math.round(gy / gn)],
      red: [Math.round(rx / rn), Math.round(ry / rn)],
    };
  };

  S.erode = function (mask, W, H, radius) {
    const out = new Uint8Array(W * H), r2 = radius * radius;
    for (let y = radius; y < H - radius; y++) {
      for (let x = radius; x < W - radius; x++) {
        if (!mask[y * W + x]) continue;
        let ok = true;
        for (let dy = -radius; dy <= radius && ok; dy++) {
          for (let dx = -radius; dx <= radius && ok; dx++) {
            if (dx * dx + dy * dy > r2) continue;
            if (!mask[(y + dy) * W + (x + dx)]) ok = false;
          }
        }
        if (ok) out[y * W + x] = 1;
      }
    }
    return out;
  };

  S.bfsConnected = function (mask, W, H, sIdx, tIdx) {
    const vis = new Uint8Array(W * H);
    const q = [sIdx]; vis[sIdx] = 1; let head = 0;
    while (head < q.length) {
      const idx = q[head++]; if (idx === tIdx) return true;
      const x = idx % W, y = (idx / W) | 0;
      if (x > 0 && !vis[idx - 1] && mask[idx - 1]) { vis[idx - 1] = 1; q.push(idx - 1); }
      if (x < W - 1 && !vis[idx + 1] && mask[idx + 1]) { vis[idx + 1] = 1; q.push(idx + 1); }
      if (y > 0 && !vis[idx - W] && mask[idx - W]) { vis[idx - W] = 1; q.push(idx - W); }
      if (y < H - 1 && !vis[idx + W] && mask[idx + W]) { vis[idx + W] = 1; q.push(idx + W); }
    }
    return false;
  };

  S.nearestInMask = function (x0, y0, target, fallback, W, H) {
    const vis = new Uint8Array(W * H);
    const q = [y0 * W + x0]; vis[q[0]] = 1; let head = 0;
    while (head < q.length) {
      const idx = q[head++]; if (target[idx]) return idx;
      const x = idx % W, y = (idx / W) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const ni = ny * W + nx;
        if (!vis[ni] && fallback[ni]) { vis[ni] = 1; q.push(ni); }
      }
    }
    return -1;
  };

  S.bestEroded = function (field) {
    const { W, H, tunnel, green, red } = field;
    for (const radius of [8, 7, 6, 5, 4, 3, 2, 1]) {
      const e = S.erode(tunnel, W, H, radius);
      const eS = S.nearestInMask(green[0], green[1], e, tunnel, W, H);
      const eT = S.nearestInMask(red[0], red[1], e, tunnel, W, H);
      if (eS < 0 || eT < 0) continue;
      if (S.bfsConnected(e, W, H, eS, eT)) return { mask: e, sIdx: eS, tIdx: eT, radius };
    }
    return null;
  };

  S.bfsShortestPath = function (mask, W, H, sIdx, tIdx, rng) {
    const parent = new Int32Array(W * H).fill(-1);
    parent[sIdx] = sIdx;
    const q = [sIdx]; let head = 0; let found = false;
    const ALL = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];
    while (head < q.length) {
      const idx = q[head++]; if (idx === tIdx) { found = true; break; }
      const x = idx % W, y = (idx / W) | 0;
      const ns = ALL.slice();
      for (let i = ns.length - 1; i > 0; i--) {
        const j = (rng() * (i + 1)) | 0;
        const t = ns[i]; ns[i] = ns[j]; ns[j] = t;
      }
      for (const [dx, dy] of ns) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const ni = ny * W + nx;
        if (parent[ni] === -1 && mask[ni]) { parent[ni] = idx; q.push(ni); }
      }
    }
    if (!found) return null;
    const path = [];
    let cur = tIdx;
    while (cur !== sIdx) { path.push([cur % W, (cur / W) | 0]); cur = parent[cur]; }
    path.push([sIdx % W, (sIdx / W) | 0]);
    path.reverse();
    return path;
  };

  S.makeSwerve = function (rng, amplitude, nHarmonics, totalLen) {
    const harm = [];
    for (let i = 0; i < nHarmonics; i++) {
      harm.push({
        amp: (rng() * 0.5 + 0.5) * amplitude / nHarmonics,
        freq: 0.5 + rng() * 3.0,
        phase: rng() * Math.PI * 2,
      });
    }
    return (s) => {
      const u = (s / Math.max(1, totalLen)) * Math.PI * 2;
      let v = 0;
      for (const h of harm) v += h.amp * Math.sin(h.freq * u + h.phase);
      return v;
    };
  };

  S.computeRandomizedPath = function (field, eroded, opts) {
    const { W, H, green, red } = field;
    const rng = opts.rng;
    const raw = S.bfsShortestPath(eroded.mask, W, H, eroded.sIdx, eroded.tIdx, rng);
    if (!raw) return null;
    raw.unshift([green[0], green[1]]);
    raw.push([red[0], red[1]]);

    const step = opts.decimateStep;
    const decimated = [];
    for (let i = 0; i < raw.length; i += step) decimated.push(raw[i]);
    if (decimated[decimated.length - 1] !== raw[raw.length - 1]) decimated.push(raw[raw.length - 1]);

    const N = decimated.length;
    const arc = new Float32Array(N);
    for (let i = 1; i < N; i++) {
      const dx = decimated[i][0] - decimated[i - 1][0];
      const dy = decimated[i][1] - decimated[i - 1][1];
      arc[i] = arc[i - 1] + Math.hypot(dx, dy);
    }
    const totalLen = arc[N - 1];
    const swerve = S.makeSwerve(rng, opts.lateralAmp, 2 + ((rng() * 3) | 0), totalLen);

    function isInside(mask, x, y) {
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= W || yi >= H) return false;
      return !!mask[yi * W + xi];
    }

    const swerved = [];
    for (let i = 0; i < N; i++) {
      if (i === 0 || i === N - 1) { swerved.push(decimated[i]); continue; }
      const a = decimated[Math.max(0, i - 1)], b = decimated[Math.min(N - 1, i + 1)];
      let tx = b[0] - a[0], ty = b[1] - a[1];
      const tn = Math.hypot(tx, ty) || 1;
      tx /= tn; ty /= tn;
      const nx = -ty, ny = tx;
      let off = swerve(arc[i]);
      let p = [decimated[i][0] + off * nx, decimated[i][1] + off * ny];
      let tries = 0;
      while (!isInside(eroded.mask, p[0], p[1]) && tries < 10) {
        off *= 0.5; tries++;
        p = [decimated[i][0] + off * nx, decimated[i][1] + off * ny];
      }
      if (!isInside(eroded.mask, p[0], p[1])) p = [decimated[i][0], decimated[i][1]];
      swerved.push(p);
    }

    const k = opts.smoothK;
    const smoothed = [];
    for (let i = 0; i < swerved.length; i++) {
      if (i === 0 || i === swerved.length - 1 || k === 0) { smoothed.push(swerved[i]); continue; }
      let sx = 0, sy = 0, n = 0;
      for (let j = Math.max(0, i - k); j <= Math.min(swerved.length - 1, i + k); j++) {
        sx += swerved[j][0]; sy += swerved[j][1]; n++;
      }
      const sm = [sx / n, sy / n];
      smoothed.push(isInside(eroded.mask, sm[0], sm[1]) ? sm : swerved[i]);
    }

    const jitterAmp = opts.jitterAmp;
    const final = [];
    for (let i = 0; i < smoothed.length; i++) {
      if (i === 0 || i === smoothed.length - 1 || jitterAmp === 0) { final.push(smoothed[i]); continue; }
      const ang = rng() * Math.PI * 2;
      const r = rng() * jitterAmp;
      let p = [smoothed[i][0] + r * Math.cos(ang), smoothed[i][1] + r * Math.sin(ang)];
      if (!isInside(eroded.mask, p[0], p[1])) p = smoothed[i];
      final.push(p);
    }

    return final;
  };

  S.dispatch = function (type, x, y, opts = {}) {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    canvas.dispatchEvent(new MouseEvent(type, {
      bubbles: true, cancelable: true,
      clientX: rect.left + x, clientY: rect.top + y,
      button: 0, buttons: opts.buttons ?? (type === 'mouseup' ? 0 : 1),
    }));
  };

  S.playPath = async function (path, opts) {
    const stepDelayMs = opts.stepDelayMs ?? 4;
    const jitterDelay = opts.jitterDelay ?? 0;
    const D = S.dispatch;
    D('mousedown', path[0][0], path[0][1]);
    await S.sleep(8);
    for (let i = 1; i < path.length; i++) {
      const state = window.__tunnelGame.getState();
      if (state === 'done_fail' || state === 'done_success') break;
      D('mousemove', path[i][0], path[i][1]);
      const d = stepDelayMs + (jitterDelay ? opts.rng() * jitterDelay : 0);
      if (d > 0) await S.sleep(d);
    }
    D('mouseup', path[path.length - 1][0], path[path.length - 1][1]);
    await S.sleep(50);
    return { state: window.__tunnelGame.getState() };
  };

  S.saveSession = async function () {
    const data = window.__tunnelGame.getSessionData();
    data.source = 'visual';
    const resp = await fetch('/api/save_trajectory', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return await resp.json();
  };

  S.runOnce = async function (tid, params) {
    window.__tunnelGame.loadTunnel(tid);
    let waited = 0;
    while (window.__tunnelGame.getState() !== 'ready' && waited < 3000) {
      await S.sleep(100); waited += 100;
    }
    const field = S.readField();
    const eroded = S.bestEroded(field);
    if (!eroded) return { error: 'no eroded mask', params };
    const path = S.computeRandomizedPath(field, eroded, params);
    if (!path) return { error: 'no path', params };
    const playOpts = { stepDelayMs: params.stepDelayMs, jitterDelay: params.jitterDelay, rng: params.rng };
    const result = await S.playPath(path, playOpts);
    let saveResult = null;
    if (result.state === 'done_success') saveResult = await S.saveSession();
    const eventCount = window.__tunnelGame.getSessionData().events.length;
    return {
      tunnelId: tid, state: result.state, save: saveResult,
      pathLen: path.length, eventCount, radius: eroded.radius,
      params: {
        seed: params.seed, decimateStep: params.decimateStep, smoothK: params.smoothK,
        lateralAmp: params.lateralAmp, jitterAmp: params.jitterAmp,
        stepDelayMs: params.stepDelayMs, jitterDelay: params.jitterDelay,
      },
    };
  };

  S.sampleParams = function (seed) {
    const rng = S.makeRng(seed);
    const decimateStep = 2 + ((rng() * 4) | 0);
    const smoothK = (rng() * 4) | 0;
    const lateralAmp = rng() * 3.0;
    const jitterAmp = rng() * 1.5;
    const stepDelayMs = 3 + ((rng() * 7) | 0);
    const jitterDelay = rng() * 4;
    return { seed, rng, decimateStep, smoothK, lateralAmp, jitterAmp, stepDelayMs, jitterDelay };
  };

  S.runMany = async function (tid, N, baseSeed = 1) {
    const out = [];
    for (let i = 0; i < N; i++) {
      let saved = null, attempts = 0;
      const seed = baseSeed + i * 97 + tid * 7919;
      while (!saved && attempts < 4) {
        const params = S.sampleParams(seed + attempts * 31);
        const r = await S.runOnce(tid, params);
        attempts++;
        if (r.state === 'done_success') { saved = r; break; }
      }
      out.push(saved || { tunnelId: tid, slot: i, state: 'fail_after_retries' });
    }
    return out;
  };

  S.runAll = async function (perTunnel, baseSeed = 1) {
    const out = {};
    for (let tid = 0; tid <= 9; tid++) out[tid] = await S.runMany(tid, perTunnel, baseSeed);
    return out;
  };

  return 'solver-installed';
})();
