// trace-the-tunnel warp solver — install onto window.__warp
//
// Strategy: for each live tunnel T (with control_points CP_live),
// sample a completed human trajectory whose tunnel_id != T from the
// server-enforced bank. Reconstruct both source and live centerlines
// from their cubic Bezier control_points (50 samples / segment, matching
// analysis/features.py:compute_centerline_deviation). For each source
// event compute its arc-length fraction f along the source centerline
// and its signed lateral offset d, then look up the live centerline at
// the same f, and emit point = p_live(f) + d * normal(tangent_live(f)).
// If a warped point lies outside the live tunnel (distance > width),
// project it back to the live centerline. Dispatch the warped events
// via the same in-page setTimeout/MouseEvent mechanism as replay_v1b
// so only the content source (warped from a different tunnel) differs.
//
// Public surface (after injection):
//   window.__warp.runOnce(tunnelId, seed?)
//   window.__warp.runMany(tunnelId, N, baseSeed?)
//   window.__warp.runAll(perTunnel, baseSeed?)
(() => {
  const W = (window.__warp = window.__warp || {});

  W.SAMPLES_PER_SEG = 50;        // matches analysis/features.py
  W.PROJECT_IF_OUTSIDE = true;   // simple clip: if warped point outside, project to centerline

  W.sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  W.makeRng = (seed) => {
    let t = seed >>> 0;
    return () => {
      t |= 0; t = (t + 0x6d2b79f5) | 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  };

  W.dispatch = function (type, x, y, opts = {}) {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    canvas.dispatchEvent(new MouseEvent(type, {
      bubbles: true, cancelable: true,
      clientX: rect.left + x, clientY: rect.top + y,
      button: 0, buttons: opts.buttons ?? (type === 'mouseup' ? 0 : 1),
    }));
  };

  W.fetchBank = async function (tid) {
    const resp = await fetch(`/api/human_bank/${tid}`);
    if (!resp.ok) throw new Error(`bank fetch failed for tunnel ${tid}: ${resp.status}`);
    return await resp.json();
  };

  W.fetchTraceFull = async function (sid) {
    const resp = await fetch(`/api/human_trace_full/${sid}`);
    if (!resp.ok) throw new Error(`trace fetch failed for session ${sid}: ${resp.status}`);
    return await resp.json();
  };

  // --- Centerline reconstruction ------------------------------------------

  W.buildCenterline = function (controlPoints) {
    // control_points is a flat list: 4 per cubic Bezier segment, in order.
    // Matches analysis/features.py shape; SAMPLES_PER_SEG points per segment.
    const pts = [];
    const numSegs = (controlPoints.length / 4) | 0;
    for (let s = 0; s < numSegs; s++) {
      const p0 = controlPoints[s * 4];
      const p1 = controlPoints[s * 4 + 1];
      const p2 = controlPoints[s * 4 + 2];
      const p3 = controlPoints[s * 4 + 3];
      for (let i = 0; i < W.SAMPLES_PER_SEG; i++) {
        const t = i / (W.SAMPLES_PER_SEG - 1);
        const u = 1 - t;
        const x = u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x;
        const y = u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y;
        pts.push({ x, y });
      }
    }
    // Cumulative arc-length
    const cum = new Float64Array(pts.length);
    cum[0] = 0;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i].x - pts[i - 1].x;
      const dy = pts[i].y - pts[i - 1].y;
      cum[i] = cum[i - 1] + Math.hypot(dx, dy);
    }
    const total = cum[cum.length - 1] || 1;
    const frac = new Float64Array(pts.length);
    for (let i = 0; i < pts.length; i++) frac[i] = cum[i] / total;
    return { pts, cum, frac, total };
  };

  // Project (px,py) onto the polyline. Returns:
  //   { f: arc-length fraction in [0,1], d: signed perpendicular offset,
  //     px_on, py_on: closest centerline point, idx: segment index }
  // Sign of d: positive if (px,py) is on the left of the tangent direction
  // (cross product (tangent) x (vec to point) > 0).
  //
  // Optional prevIdx/fwd/bwd restrict the search window. Not used by the
  // current warpEvents (which derives f and d directly from sampled
  // source-centerline-at-f instead of a closest-point projection), but
  // retained for diagnostic use — closest-point projection on a folded
  // centerline can flip branches at U-bends and is the bug the new
  // algorithm avoids by construction.
  W.projectToCenterline = function (px, py, cl, prevIdx, fwd, bwd) {
    const { pts, cum, total } = cl;
    let lo, hi;
    if (typeof prevIdx === 'number' && prevIdx >= 0) {
      const f_ = (typeof fwd === 'number') ? fwd : 8;
      const b_ = (typeof bwd === 'number') ? bwd : 4;
      lo = Math.max(0, prevIdx - b_);
      hi = Math.min(pts.length - 1, prevIdx + f_);
    } else {
      lo = 0; hi = pts.length - 1;
    }
    let bestD2 = Infinity, bestIdx = lo, bestT = 0, bestPx = pts[lo].x, bestPy = pts[lo].y;
    for (let i = lo; i < hi; i++) {
      const ax = pts[i].x, ay = pts[i].y;
      const bx = pts[i + 1].x, by = pts[i + 1].y;
      const dx = bx - ax, dy = by - ay;
      const lenSq = dx * dx + dy * dy;
      let t;
      if (lenSq === 0) {
        t = 0;
      } else {
        t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
        if (t < 0) t = 0; else if (t > 1) t = 1;
      }
      const cx = ax + t * dx;
      const cy = ay + t * dy;
      const ddx = px - cx, ddy = py - cy;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestIdx = i;
        bestT = t;
        bestPx = cx;
        bestPy = cy;
      }
    }
    // Arc-length at the projected location
    const arc = cum[bestIdx] + bestT * (cum[bestIdx + 1] - cum[bestIdx]);
    const f = arc / total;
    // Signed perpendicular offset via cross product with the segment tangent
    const ax = pts[bestIdx].x, ay = pts[bestIdx].y;
    const bx = pts[bestIdx + 1].x, by = pts[bestIdx + 1].y;
    let tdx = bx - ax, tdy = by - ay;
    const tnorm = Math.hypot(tdx, tdy) || 1;
    tdx /= tnorm; tdy /= tnorm;
    const vx = px - bestPx, vy = py - bestPy;
    // cross(tangent, v) = tdx*vy - tdy*vx ; sign defines side
    const cross = tdx * vy - tdy * vx;
    const dist = Math.sqrt(bestD2);
    const d = cross >= 0 ? dist : -dist;
    return { f, d, px_on: bestPx, py_on: bestPy, idx: bestIdx };
  };

  // Look up (point, tangent) on a centerline at arc-length fraction f.
  W.sampleAtFraction = function (f, cl) {
    const { pts, frac } = cl;
    if (f <= 0) {
      const tdx = pts[1].x - pts[0].x;
      const tdy = pts[1].y - pts[0].y;
      const n = Math.hypot(tdx, tdy) || 1;
      return { x: pts[0].x, y: pts[0].y, tx: tdx / n, ty: tdy / n };
    }
    if (f >= 1) {
      const L = pts.length;
      const tdx = pts[L - 1].x - pts[L - 2].x;
      const tdy = pts[L - 1].y - pts[L - 2].y;
      const n = Math.hypot(tdx, tdy) || 1;
      return { x: pts[L - 1].x, y: pts[L - 1].y, tx: tdx / n, ty: tdy / n };
    }
    // Binary search for the index whose frac[idx] <= f < frac[idx+1]
    let lo = 0, hi = frac.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (frac[mid] <= f) lo = mid; else hi = mid;
    }
    const denom = frac[hi] - frac[lo] || 1e-9;
    const u = (f - frac[lo]) / denom;
    const x = pts[lo].x + u * (pts[hi].x - pts[lo].x);
    const y = pts[lo].y + u * (pts[hi].y - pts[lo].y);
    // Tangent from the local segment direction
    let tdx = pts[hi].x - pts[lo].x;
    let tdy = pts[hi].y - pts[lo].y;
    const n = Math.hypot(tdx, tdy) || 1;
    return { x, y, tx: tdx / n, ty: tdy / n };
  };

  // --- Warp a full event list ---------------------------------------------

  W.warpEvents = function (events, srcCP, liveCP, tunnelWidth) {
    const clSrc = W.buildCenterline(srcCP);
    const clLive = W.buildCenterline(liveCP);
    const out = new Array(events.length);
    let nOutside = 0, nProjected = 0;

    // Live red-anchor: the last control point. The game registers completion
    // by hit-test against this dot (radius DOT_RADIUS+4=18 in game.js), so
    // the trajectory's last few events must actually land on (or in) the
    // live red dot — otherwise the play never completes.
    const liveRed = liveCP[liveCP.length - 1];

    // (H1 fix) Compute the per-event arc-length fraction from the CUMULATIVE
    // EVENT-PATH LENGTH, not from a closest-point projection onto the source
    // centerline. The projection-based f is non-monotonic when the source
    // wobbles near a U-bend (centerline curves back on itself), causing f to
    // jump forward 5-9% in one event and the warped point to teleport 80+ px
    // along the live tunnel. Cumulative event-path length is monotonic by
    // construction and tracks the source's temporal pacing through the tunnel.
    let totalEv = 0;
    const cumEv = new Float64Array(events.length);
    cumEv[0] = 0;
    for (let i = 1; i < events.length; i++) {
      const dx = events[i].x - events[i - 1].x;
      const dy = events[i].y - events[i - 1].y;
      totalEv += Math.hypot(dx, dy);
      cumEv[i] = totalEv;
    }
    if (totalEv <= 0) totalEv = 1;

    // (H4 fix) Snap tail to live red anchor by arc-length fraction (last 3%),
    // not by source-red hit-radius. The old hit-radius test depended on the
    // source path actually getting within 18px of its own red — when warp
    // branch-flips placed the last few warped events tens of px past the live
    // red anchor and then jumped back ("overshoot then loop back" artifact),
    // there was nothing to suppress it. Snap-by-f catches every case.
    const SNAP_F = 0.97;

    // (H2 fix) The lateral offset d is computed as the signed perpendicular
    // distance from the source event to the SOURCE CENTERLINE POINT AT f,
    // NOT from a global closest-point projection. Pairing event-path-f with
    // source-centerline-at-f is geometrically consistent (the same f indexes
    // both the source position and the live position used for mapping), so
    // d cannot blow up when the source projection lags the event-path. It
    // also dodges the closest-point ambiguity at U-bends entirely because
    // there is no closest-point search.
    let snapFrom = -1;
    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const f = cumEv[i] / totalEv;
      let wx, wy;
      let d = 0;
      let outside = false;
      if (f >= SNAP_F) {
        // (H4) Tail-snap: every remaining event lands on the live red anchor.
        wx = liveRed.x;
        wy = liveRed.y;
        if (snapFrom < 0) snapFrom = i;
      } else {
        // Source centerline position + tangent at f.
        const ss = W.sampleAtFraction(f, clSrc);
        // Source normal (left of tangent): rotate tangent +90deg.
        const snx = -ss.ty, sny = ss.tx;
        // Signed lateral offset of the event from the source-centerline-at-f.
        d = (ev.x - ss.x) * snx + (ev.y - ss.y) * sny;
        // Live centerline position + tangent at the SAME f.
        const ls = W.sampleAtFraction(f, clLive);
        const lnx = -ls.ty, lny = ls.tx;
        wx = ls.x + d * lnx;
        wy = ls.y + d * lny;
        outside = Math.abs(d) > tunnelWidth;
        if (outside) nOutside++;
        if (W.PROJECT_IF_OUTSIDE && outside) {
          // Shrink offset to (tunnelWidth - margin) so warped point is inside
          // the live tunnel by a small safety margin.
          const margin = 2;
          const shrunk = Math.sign(d) * Math.max(0, tunnelWidth - margin);
          wx = ls.x + shrunk * lnx;
          wy = ls.y + shrunk * lny;
          nProjected++;
        }
      }
      out[i] = {
        x: wx,
        y: wy,
        timestamp: ev.timestamp,
        event_type: ev.event_type,
      };
    }
    // Force the final event onto the red anchor regardless of type, so the
    // game's completion check on mouseup definitely passes.
    if (out.length > 0) {
      out[out.length - 1].x = liveRed.x;
      out[out.length - 1].y = liveRed.y;
    }
    return { events: out, n_outside: nOutside, n_projected: nProjected, snap_from: snapFrom };
  };

  // --- Playback ------------------------------------------------------------

  W.playEvents = async function (events) {
    if (!events || events.length === 0) return;
    const downIdx = events.findIndex((e) => e.event_type === 'mousedown');
    let upIdx = -1;
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].event_type === 'mouseup') { upIdx = i; break; }
    }
    const start = downIdx >= 0 ? downIdx : 0;
    const end = upIdx >= 0 ? upIdx : events.length - 1;
    const slice = events.slice(start, end + 1);

    const first = slice[0];
    W.dispatch(first.event_type, first.x, first.y);
    let last = first.timestamp;

    for (let i = 1; i < slice.length; i++) {
      const e = slice[i];
      const dt = Math.max(0, e.timestamp - last);
      last = e.timestamp;
      if (dt > 0) await W.sleep(dt);
      if (e.event_type === 'mousemove' || e.event_type === 'mousedown' || e.event_type === 'mouseup') {
        W.dispatch(e.event_type, e.x, e.y);
      }
      const state = window.__tunnelGame.getState();
      if (state === 'done_success' || state === 'done_fail') break;
    }
  };

  W.saveSession = async function () {
    const data = window.__tunnelGame.getSessionData();
    data.source = 'warp_v1';
    const resp = await fetch('/api/save_trajectory', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return await resp.json();
  };

  // --- Source selection (cross-tunnel constraint) --------------------------
  // The server's /api/human_bank/<tid> is already filtered by the allowlist.
  // We pick a source whose tunnel_id != live tunnel_id by enumerating banks
  // for every other tid and pooling their session_ids.
  W.pickSource = async function (liveTid, rng) {
    const pool = [];
    for (let tid = 0; tid <= 9; tid++) {
      if (tid === liveTid) continue;
      try {
        const bank = await W.fetchBank(tid);
        for (const entry of bank) {
          if (entry && entry.session_id && entry.tunnel_id !== liveTid) {
            pool.push({ session_id: entry.session_id, tunnel_id: entry.tunnel_id });
          }
        }
      } catch (e) {
        // skip on per-tid fetch error; continue building the pool
      }
    }
    if (pool.length === 0) return null;
    return pool[(rng() * pool.length) | 0];
  };

  W.runOnce = async function (tid, seed = Date.now() & 0x7fffffff) {
    window.__tunnelGame.loadTunnel(tid);
    let waited = 0;
    while (window.__tunnelGame.getState() !== 'ready' && waited < 3000) {
      await W.sleep(100); waited += 100;
    }
    const liveCP = window.__tunnelGame.getControlPoints();
    const tunnelWidth = window.__tunnelGame.getTunnelWidth();
    const rng = W.makeRng(seed);

    const pick = await W.pickSource(tid, rng);
    if (!pick) {
      return { error: `no cross-tunnel source available for tunnel ${tid}`, tunnelId: tid };
    }
    let full;
    try {
      full = await W.fetchTraceFull(pick.session_id);
    } catch (e) {
      return { error: String(e), tunnelId: tid };
    }
    if (!full || !full.control_points || !full.events) {
      return { error: 'malformed source trace', tunnelId: tid };
    }
    if (full.tunnel_id === tid) {
      return { error: 'source tunnel_id matches live (cross-tunnel violated)', tunnelId: tid };
    }

    const warped = W.warpEvents(full.events, full.control_points, liveCP, tunnelWidth);
    await W.playEvents(warped.events);
    await W.sleep(80);

    const state = window.__tunnelGame.getState();
    let saveResult = null;
    if (state === 'done_success') saveResult = await W.saveSession();
    const liveEvents = window.__tunnelGame.getSessionData().events.length;
    return {
      tunnelId: tid,
      state,
      save: saveResult,
      source_session: full.session_id,
      source_tunnel: full.tunnel_id,
      source_event_count: full.events.length,
      live_event_count: liveEvents,
      n_outside: warped.n_outside,
      n_projected: warped.n_projected,
    };
  };

  W.runMany = async function (tid, N, baseSeed = 1) {
    const out = [];
    for (let i = 0; i < N; i++) {
      let saved = null, attempts = 0, errored = false;
      const seed = baseSeed + i * 97 + tid * 7919;
      while (!saved && attempts < 4) {
        const r = await W.runOnce(tid, seed + attempts * 31);
        attempts++;
        if (r.error) { errored = true; break; }
        if (r.state === 'done_success') { saved = r; break; }
      }
      if (saved) out.push(saved);
      else if (errored) {
        out.push({ tunnelId: tid, slot: i, state: 'skipped_error' });
        break;
      } else {
        out.push({ tunnelId: tid, slot: i, state: 'fail_after_retries' });
      }
    }
    return out;
  };

  W.runAll = async function (perTunnel = 2, baseSeed = 1) {
    const out = {};
    for (let tid = 0; tid <= 9; tid++) out[tid] = await W.runMany(tid, perTunnel, baseSeed);
    return out;
  };

  return 'warp-installed';
})();
