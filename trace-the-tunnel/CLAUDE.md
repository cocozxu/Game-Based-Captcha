# Trace-the-Tunnel — Agent Player

You are playing a browser game called "Trace-the-Tunnel" to collect trajectory data for research.
The game is running at http://localhost:5050.

## Your Task

1. Navigate to the game
2. Load the specific tunnel you're told to play
3. Read the tunnel geometry from the page
4. Trace from the green dot to the red dot, staying inside the tunnel
5. Re-save the trajectory with the correct source label
6. Confirm success

## How the Game Works

- A curved tunnel is drawn on a 600x350 canvas
- Green dot = start (left side), Red dot = end (right side)
- You must mousedown on the green dot, drag through the tunnel, and reach the red dot
- If you leave the tunnel, the game ends immediately as a failure
- The game records every mouse event (x, y, timestamp) as your trajectory
- On completion (success or failure), the game auto-POSTs the trajectory
- There are 10 tunnels (IDs 0-9) in a fixed pool — both humans and agents play the same set

## Reading Tunnel Geometry

Use browser_evaluate to get the centerline and canvas rect:

```js
(() => {
  const g = window.__tunnelGame;
  const rect = g.getCanvasRect();
  const centerline = g.getCenterline();
  const sampled = centerline.filter((_, i) => i % 10 === 0);
  return {
    tunnelId: g.getTunnelId(),
    start: g.getStartPos(),
    end: g.getEndPos(),
    tunnelWidth: g.getTunnelWidth(),
    canvasRect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
    canvasSize: { width: 600, height: 350 },
    centerlineSampled: sampled,
    state: g.getState(),
  };
})()
```

## Coordinate Conversion

Canvas coords to screen coords:
```
screenX = canvasRect.left + (canvasPoint.x / 600) * canvasRect.width
screenY = canvasRect.top  + (canvasPoint.y / 350) * canvasRect.height
```

## Loading a Specific Tunnel

```js
window.__tunnelGame.loadTunnel(TUNNEL_ID)  // 0-9
```

## Re-saving with Correct Source

After the game ends (success or failure), re-save with your label:
```js
(async () => {
  const data = window.__tunnelGame.getSessionData();
  data.source = 'agent';  // or 'agent_humanlike'
  const resp = await fetch('/api/save_trajectory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return { ok: resp.ok, session_id: data.session_id, tunnel_id: data.tunnel_id, events: data.events.length, completed: data.completed };
})()
```

---

## Mode: Naive Agent

You have access to the tunnel centerline and Playwright mouse controls.
Figure out how to trace through the tunnel using browser_run_code_unsafe.

Try to move the way a real human would — think about how a person actually
drags a mouse through a curving tunnel. Do your best to produce a trajectory
that looks natural. It's up to you how to achieve this.

---

## Mode: Human-Like Agent

Same as naive, but you also have access to `human_cursor.py` — a Python tool
that can transform a path into one with more natural characteristics.

Usage via Bash:
```bash
source .venv/bin/activate && python human_cursor.py --centerline '<JSON_ARRAY_OF_XY>' --noise 2.5 --overshoots 2 --duration 900 --points 150
```

Input: JSON array of [x,y] points (get from centerline).
Output: JSON array of {x, y, delay_ms} waypoints to replay via Playwright.

Parameters you can vary:
- `--noise` : perpendicular jitter std dev in pixels (default 2.0, range 1.0-4.0)
- `--overshoots` : number of overshoot/correction points (default 2, range 0-4)
- `--duration` : total trace time in ms (default 800, range 500-1500)
- `--points` : output waypoint count (default 150, range 80-200)

---

## Important Notes

- The server must be running: `source .venv/bin/activate && python server.py`
- Check game state after tracing: `window.__tunnelGame.getState()` — either `"done_success"` or `"done_fail"`
- The game auto-retries on failure (resets the same tunnel after 800ms). **If you fail, you must try again until you succeed.** Adjust your approach — use more points, less noise, smaller steps, etc.
- Only move to the next tunnel after a successful trace (`"done_success"`)
- Trajectories (both failures and the final success) are saved to `data/<source>/<session_id>.json`
