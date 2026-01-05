import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =========================
# CONFIG
# =========================
RUN_DIR = r"runs/20260103_221930"  # e.g. runs/20260103_210512
POSE_PATH   = os.path.join(RUN_DIR, "pose.csv")
EVENTS_PATH = os.path.join(RUN_DIR, "events.csv")
OUT_MP4     = os.path.join(RUN_DIR, "path_animation.mp4")

# Animation timing (independent of recorded FPS)
ANIM_FPS = 30

INVERT_Y_AXIS = True     

# =========================
# Load pose.csv
# =========================
def load_pose_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                valid = str(r.get("valid", "")).strip().lower() in ("true", "1", "yes")
                if not valid:
                    continue
                rows.append({
                    "t": float(r["t"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "yaw": float(r["yaw"]),
                })
            except Exception:
                continue
    rows.sort(key=lambda d: d["t"])
    return rows

pose = load_pose_csv(POSE_PATH)
if len(pose) < 2:
    raise RuntimeError("Not enough valid pose samples found in pose.csv")

t   = np.array([p["t"] for p in pose])
x   = np.array([p["x"] for p in pose])
y   = np.array([p["y"] for p in pose])
yaw = np.array([p["yaw"] for p in pose])

# =========================
# Load events.csv (optional)
# =========================
def load_events_csv(path):
    goals = []   # (t, gx, gy)
    runs  = []   # (t, enabled)
    if not os.path.exists(path):
        return goals, runs

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            typ = str(r.get("type", "")).strip().lower()
            try:
                tt = float(r.get("t"))
            except Exception:
                continue

            if typ == "goal":
                try:
                    gx = float(r.get("x"))
                    gy = float(r.get("y"))
                    goals.append((tt, gx, gy))
                except Exception:
                    pass

            if typ == "run":
                en = str(r.get("enabled", "")).strip().lower()
                enabled = en in ("true", "1", "yes")
                runs.append((tt, enabled))

    goals.sort(key=lambda z: z[0])
    runs.sort(key=lambda z: z[0])
    return goals, runs

goals, runs = load_events_csv(EVENTS_PATH)

# =========================
# Build a uniform-time animation timeline
# =========================
t0, t1 = float(t[0]), float(t[-1])
if t1 <= t0:
    raise RuntimeError("Pose timestamps invalid (t1 <= t0).")

# Create uniformly spaced animation timestamps
dt_anim = 1.0 / ANIM_FPS
t_anim = np.arange(t0, t1 + 1e-9, dt_anim)

# Interpolate x, y over time
x_anim = np.interp(t_anim, t, x)
y_anim = np.interp(t_anim, t, y)

# Interpolate yaw carefully (unwrap to avoid pi jumps)
yaw_unwrapped = np.unwrap(yaw)
yaw_anim_unwrapped = np.interp(t_anim, t, yaw_unwrapped)
yaw_anim = (yaw_anim_unwrapped + np.pi) % (2*np.pi) - np.pi

# =========================
# Plot setup
# =========================
pad = 0.05 * max(float(x.max() - x.min()), float(y.max() - y.min()), 1e-6)
xmin, xmax = float(x.min() - pad), float(x.max() + pad)
ymin, ymax = float(y.min() - pad), float(y.max() + pad)

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
if INVERT_Y_AXIS:
    ax.invert_yaxis()
ax.set_title("Rover Path Reconstruction")
ax.set_xlabel("x (world)")
ax.set_ylabel("y (world)")
ax.grid(True)

# Draw goals (static markers)
if goals:
    gx = [g[1] for g in goals]
    gy = [g[2] for g in goals]
    ax.scatter(gx, gy, marker="x", s=70, label="Goals")

trail_line, = ax.plot([], [], linewidth=2, label="Trail")
dot, = ax.plot([], [], marker="o", markersize=7, label="Rover")

# A “current goal” highlight that updates over time
cur_goal_marker, = ax.plot([], [], marker="x", markersize=12, linestyle="None")

ax.legend(loc="best")


def current_goal_at_time(tt):
    """Return (gx,gy) for the latest goal with t_goal <= tt, else None."""
    if not goals:
        return None
    # goals sorted by time; linear scan is fine for small sizes
    g = None
    for (tg, gx, gy) in goals:
        if tg <= tt:
            g = (gx, gy)
        else:
            break
    return g

def init():
    trail_line.set_data([], [])
    dot.set_data([], [])
    cur_goal_marker.set_data([], [])
    return trail_line, dot, cur_goal_marker

def update(i):
    # Trail up to i
    trail_line.set_data(x_anim[:i+1], y_anim[:i+1])

    # Rover point
    dot.set_data([x_anim[i]], [y_anim[i]])

    # Current goal highlight (latest goal so far)
    g = current_goal_at_time(float(t_anim[i]))
    if g is not None:
        cur_goal_marker.set_data([g[0]], [g[1]])
    else:
        cur_goal_marker.set_data([], [])

    return trail_line, dot, cur_goal_marker

ani = FuncAnimation(fig, update, frames=len(t_anim), init_func=init, interval=1000/ANIM_FPS, blit=True)

# =========================
# Export MP4 in the same RUN_DIR
# =========================
# Requires ffmpeg installed and accessible in PATH.
writer = FFMpegWriter(fps=ANIM_FPS, metadata={"title": "Rover Path"}, bitrate=3000)
print(f"[EXPORT] Writing MP4 to: {OUT_MP4}")
ani.save(OUT_MP4, writer=writer)
print("[EXPORT] Done.")