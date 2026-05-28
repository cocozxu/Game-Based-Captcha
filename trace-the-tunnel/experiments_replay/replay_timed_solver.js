// trace-the-tunnel mechanism-silencing replay solver — install onto window.__replay
//
// Same source-selection logic as replay_solver.js: sample a real human trace
// from /api/human_bank/<tid> and dispatch its events back into the canvas.
//
// The ONLY difference from replay_solver.js is the dispatch schedule. The
// original uses `await setTimeout(dt)` between events, which has a 4ms minimum
// and ~1ms scheduling jitter — that distorts the recorded dt distribution
// (replay_v1b dt_mean=9.46 vs human 8.47; dt_std=1.81 vs human 0.94) and is
// what the mechanism head learned to detect.
//
// This variant uses a busy-wait targeting absolute timestamps:
//
//   - For each source event i, compute target_i = origin + (src.t[i] - src.t[0])
//   - `while (performance.now() < target_i) { /* spin */ }`
//   - synchronously canvas.dispatchEvent(...) — the captcha's listener fires
//     in the same tick and records performance.now() with sub-ms offset from
//     target_i.
//
// Cost: the main thread is blocked for the duration of one trace (~1.3s).
// That's fine for an automated agent.
//
// Saves to data/replay_timed/ when the server is in --experiment replay_timed.
(() => {
  const R = (window.__replay = window.__replay || {});

  R.sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  R.makeRng = (seed) => {
    let t = seed >>> 0;
    return () => {
      t |= 0; t = (t + 0x6d2b79f5) | 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  };

  R.dispatch = function (type, x, y, opts = {}) {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    canvas.dispatchEvent(new MouseEvent(type, {
      bubbles: true, cancelable: true,
      clientX: rect.left + x, clientY: rect.top + y,
      button: 0, buttons: opts.buttons ?? (type === 'mouseup' ? 0 : 1),
    }));
  };

  R.fetchBank = async function (tid) {
    const resp = await fetch(`/api/human_bank/${tid}`);
    if (!resp.ok) throw new Error(`bank fetch failed for tunnel ${tid}: ${resp.status}`);
    return await resp.json();
  };

  // -------- MECHANISM-SILENCING DISPATCHER --------
  // Synchronous busy-wait on absolute target timestamps, derived from the
  // source trace's own (src.t - src.t[0]) offsets. The recorded
  // performance.now() inside the captcha listener should match each
  // target_i to within sub-ms, so the recorded dt distribution mirrors the
  // human source.
  R.playEventsTimed = function (events) {
    if (!events || events.length === 0) return;

    const downIdx = events.findIndex((e) => e.event_type === 'mousedown');
    let upIdx = -1;
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].event_type === 'mouseup') { upIdx = i; break; }
    }
    const start = downIdx >= 0 ? downIdx : 0;
    const end = upIdx >= 0 ? upIdx : events.length - 1;
    const slice = events.slice(start, end + 1);

    const srcOrigin = slice[0].timestamp;
    const dispatchOrigin = performance.now();

    for (let i = 0; i < slice.length; i++) {
      const e = slice[i];
      const target = dispatchOrigin + (e.timestamp - srcOrigin);
      // Spin until we hit the target. Tight loop intentional — the cost of
      // mechanism-clean dispatch is blocking the event loop for ~1.3s.
      while (performance.now() < target) { /* spin */ }
      if (e.event_type === 'mousemove' || e.event_type === 'mousedown' || e.event_type === 'mouseup') {
        R.dispatch(e.event_type, e.x, e.y);
      }
      const state = window.__tunnelGame.getState();
      if (state === 'done_success' || state === 'done_fail') break;
    }
  };

  R.saveSession = async function () {
    const data = window.__tunnelGame.getSessionData();
    data.source = 'replay_timed';
    const resp = await fetch('/api/save_trajectory', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return await resp.json();
  };

  R.runOnce = async function (tid, seed = Date.now() & 0x7fffffff) {
    window.__tunnelGame.loadTunnel(tid);
    let waited = 0;
    while (window.__tunnelGame.getState() !== 'ready' && waited < 3000) {
      await R.sleep(100); waited += 100;
    }
    const bank = await R.fetchBank(tid);
    if (!bank || bank.length === 0) return { error: `empty bank for tunnel ${tid}`, tunnelId: tid };
    const rng = R.makeRng(seed);
    const choice = bank[(rng() * bank.length) | 0];
    R.playEventsTimed(choice.events);
    // Brief yield so the captcha's fire-and-forget autoSave can dispatch its
    // POST and the state machine can settle before we read it.
    await R.sleep(40);
    const state = window.__tunnelGame.getState();
    let saveResult = null;
    if (state === 'done_success') saveResult = await R.saveSession();
    const liveEvents = window.__tunnelGame.getSessionData().events.length;
    return {
      tunnelId: tid, state, save: saveResult,
      source_session: choice.session_id,
      source_event_count: choice.events.length,
      live_event_count: liveEvents,
      bank_size: bank.length,
    };
  };

  R.runMany = async function (tid, N, baseSeed = 1) {
    const out = [];
    for (let i = 0; i < N; i++) {
      let saved = null, attempts = 0, emptyBank = false;
      const seed = baseSeed + i * 97 + tid * 7919;
      while (!saved && attempts < 4) {
        const r = await R.runOnce(tid, seed + attempts * 31);
        attempts++;
        if (r.error && r.error.startsWith('empty bank')) {
          emptyBank = true;
          break;
        }
        if (r.state === 'done_success') { saved = r; break; }
      }
      if (saved) out.push(saved);
      else if (emptyBank) {
        out.push({ tunnelId: tid, slot: i, state: 'skipped_empty_bank' });
        break;
      } else {
        out.push({ tunnelId: tid, slot: i, state: 'fail_after_retries' });
      }
    }
    return out;
  };

  R.runAll = async function (perTunnel = 2, baseSeed = 1) {
    const out = {};
    for (let tid = 0; tid <= 9; tid++) out[tid] = await R.runMany(tid, perTunnel, baseSeed);
    return out;
  };

  return 'replay-timed-installed';
})();
