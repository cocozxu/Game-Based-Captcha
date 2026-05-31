import numpy as np

def generate(tunnel_spec: dict, seed: int) -> list[dict]:
    """
    Generate human-like mouse traces for unseen tunnel CAPTCHA.

    Strategy:
    - Interpolate smooth path through tunnel centerline
    - Estimate local curvature to modulate movement speed
    - Generate realistic timing with variable dt (mostly 8-15ms)
    - Add persistent jitter correlated with curvature
    - Enforce tunnel boundary constraints
    """
    np.random.seed(seed)

    control_points = tunnel_spec['control_points']
    tunnel_width = tunnel_spec['tunnel_width']

    # Extract coordinates
    xs = np.array([p['x'] for p in control_points], dtype=np.float64)
    ys = np.array([p['y'] for p in control_points], dtype=np.float64)

    # Resample centerline at high density
    t_coarse = np.arange(len(xs))
    t_dense = np.linspace(0, len(xs) - 1, 650)

    path_x = np.interp(t_dense, t_coarse, xs)
    path_y = np.interp(t_dense, t_coarse, ys)

    # Estimate curvature via finite differences
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Curvature = |dx*d2y - dy*d2x| / (dx^2 + dy^2)^(3/2)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-6
    curvature = np.abs(dx * d2y - dy * d2x) / denom

    # Smooth curvature with local averaging
    curv_smooth = np.zeros_like(curvature)
    window = 4
    for i in range(len(curvature)):
        start = max(0, i - window)
        end = min(len(curvature), i + window + 1)
        curv_smooth[i] = np.mean(curvature[start:end])

    # Normalize curvature to [0, 1]
    c_min, c_max = curv_smooth.min(), curv_smooth.max()
    if c_max > c_min:
        curv_norm = (curv_smooth - c_min) / (c_max - c_min)
    else:
        curv_norm = np.zeros_like(curv_smooth)

    # Speed varies inversely with curvature (faster on straights)
    speed_multiplier = 0.55 + 0.45 * (1.0 - curv_norm)

    # Build event stream
    events = []

    # Initial timestamp (realistic human-like timing)
    t_current = float(np.random.randint(18000, 520000))

    # Mousedown at start
    events.append({
        'x': float(path_x[0]),
        'y': float(path_y[0]),
        'timestamp': t_current,
        'event_type': 'mousedown',
        'inside_tunnel': True
    })

    # State for correlated jitter (Ornstein-Uhlenbeck process)
    jitter_state_x = 0.0
    jitter_state_y = 0.0

    # Generate mousemove events
    for i in range(1, len(path_x)):
        # Base time interval, modulated by speed
        dt_base = 8.8  # milliseconds
        dt = dt_base / speed_multiplier[i]

        # Add realistic variation (~10% std dev)
        dt += np.random.normal(0, dt * 0.10)
        dt = np.clip(dt, 4.5, 24.0)

        # Occasional pauses (user hesitation)
        if np.random.random() < 0.016:
            dt += np.random.exponential(48)

        t_current += dt

        # Get centerline position
        cx = path_x[i]
        cy = path_y[i]

        # Jitter magnitude increases at curves
        jitter_scale = 0.65 * (1.0 + 1.5 * curv_norm[i])

        # Persistent jitter (smooth random walk)
        jitter_state_x = 0.76 * jitter_state_x + np.random.normal(0, jitter_scale * 0.32)
        jitter_state_y = 0.76 * jitter_state_y + np.random.normal(0, jitter_scale * 0.32)

        # Clamp jitter amplitude
        jitter_x = np.clip(jitter_state_x, -2.1, 2.1)
        jitter_y = np.clip(jitter_state_y, -2.1, 2.1)

        # Apply jitter to position
        x = cx + jitter_x
        y = cy + jitter_y

        # Enforce tunnel boundary (stay within width/2 of centerline)
        jitter_dist = np.sqrt(jitter_x**2 + jitter_y**2)
        max_jitter = tunnel_width / 2 - 0.4

        if jitter_dist > max_jitter:
            scale_factor = max_jitter / (jitter_dist + 1e-8)
            x = cx + jitter_x * scale_factor
            y = cy + jitter_y * scale_factor

        events.append({
            'x': float(x),
            'y': float(y),
            'timestamp': t_current,
            'event_type': 'mousemove',
            'inside_tunnel': True
        })

    # Mouseup at endpoint
    events.append({
        'x': float(path_x[-1]),
        'y': float(path_y[-1]),
        'timestamp': t_current,
        'event_type': 'mouseup',
        'inside_tunnel': True
    })

    return events
