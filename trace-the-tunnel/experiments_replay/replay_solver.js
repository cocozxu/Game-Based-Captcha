// trace-the-tunnel replay solver — install onto window.__replay
//
// Strategy: for each live tunnel, fetch the server's bank of completed
// human trajectories for that exact tunnel_id, pick one at random, and
// dispatch its mouse events back into the canvas at their original
// inter-event timings. Because tunnels are deterministic from a fixed
// seed pool (see tunnel_config.json), the source human trace and the
// live tunnel share identical geometry, so no spatial warping is needed.
//
// Public surface (after injection):
//   window.__replay.runOnce(tunnelId, seed?)
//   window.__replay.runMany(tunnelId, N, baseSeed?)
//   window.__replay.runAll(perTunnel, baseSeed?)
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

  R.playEvents = async function (events) {
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
    R.dispatch(first.event_type, first.x, first.y);
    let last = first.timestamp;

    for (let i = 1; i < slice.length; i++) {
      const e = slice[i];
      const dt = Math.max(0, e.timestamp - last);
      last = e.timestamp;
      if (dt > 0) await R.sleep(dt);
      if (e.event_type === 'mousemove' || e.event_type === 'mousedown' || e.event_type === 'mouseup') {
        R.dispatch(e.event_type, e.x, e.y);
      }
      const state = window.__tunnelGame.getState();
      if (state === 'done_success' || state === 'done_fail') break;
    }
  };

  R.saveSession = async function () {
    const data = window.__tunnelGame.getSessionData();
    data.source = 'replay';
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
    await R.playEvents(choice.events);
    await R.sleep(80);
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
        break;  // no point trying the remaining slots; this tunnel has no source
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

  return 'replay-installed';
})();
