import os 
import csv 
from datetime import datetime
import cv2
import numpy as np
import json
import socket
import time
from pupil_apriltags import Detector

# ========== CONFIG ==========
PI_IP = "172.20.10.3"

POSE_PORT = 5005     # Pi listens for pose here
CMD_PORT  = 6006     # Pi listens for goal/run/arm commands here

TAG_ID = 1
CAM_INDEX = 1
H_FILE = "homography_floor.npz"

WORLD_W_METERS = 1.60
WORLD_H_METERS = 1.80

# ========== Logging / Recording ==========
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RUNS_DIR, run_stamp)
os.makedirs(run_dir, exist_ok=True)

video_path  = os.path.join(run_dir, "video.mp4")
pose_path   = os.path.join(run_dir, "pose.csv")
events_path = os.path.join(run_dir, "events.csv")

# CSV writers (opened later after camera is opened)
pose_f = None
events_f = None
pose_writer = None
events_writer = None

# Video writer (opened after first frame)
vw = None

# Optional meta
with open(os.path.join(run_dir, "meta.json"), "w") as f:
    json.dump({
        "run_dir": run_dir,
        "pi_ip": PI_IP,
        "pose_port": POSE_PORT,
        "cmd_port": CMD_PORT,
        "tag_id": TAG_ID,
        "world_w_m": WORLD_W_METERS,
        "world_h_m": WORLD_H_METERS,
        "cam_index": CAM_INDEX,
        "homography_file": H_FILE,
        "start_time_unix": time.time(),
    }, f, indent=2)

print(f"[LOG] Run directory: {run_dir}")

# ========== UDP setup ==========
sock_pose = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_cmd  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_pose(payload):
    msg = json.dumps(payload).encode("utf-8")
    sock_pose.sendto(msg, (PI_IP, POSE_PORT))

def send_cmd(payload):
    payload["t"] = time.time()
    msg = json.dumps(payload).encode("utf-8")
    sock_cmd.sendto(msg, (PI_IP, CMD_PORT))

    # Log commands for offline reconstruction
    t = payload["t"]
    typ = payload.get("type", "")
    if typ == "goal":
        log_event("goal", x=payload.get("x"), y=payload.get("y"))
    elif typ == "run":
        log_event("run", enabled=bool(payload.get("enabled", False)))
    elif typ == "arm":
        log_event("arm", action=payload.get("action"))
    else:
        log_event("cmd", raw=json.dumps(payload))

# ========== LOGGING ==========
def log_event(event_type, **kwargs):
    global events_writer
    if events_writer is None:
        return
    row = {"t": time.time(), "type": event_type}
    row.update(kwargs)
    events_writer.writerow(row)

# ========== AprilTag detector ==========
detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0, quad_sigma=0.0)

# ========== Homography utils ==========
clicked_points = []
H = None

last_goal = None
last_goal_time = 0.0

def mouse_cb_calib(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Clicked: ({x},{y})  ({len(clicked_points)}/4)")

def load_homography():
    try:
        data = np.load(H_FILE)
        return data["H"]
    except Exception:
        return None

def save_homography(H_):
    np.savez(H_FILE, H=H_)
    print(f"Saved homography to {H_FILE}")

def compute_homography(frame):
    """
    User clicks 4 image points corresponding to:
      (0,0), (W,0), (W,H), (0,H)
    """
    global clicked_points
    clicked_points = []

    clone = frame.copy()
    cv2.namedWindow("CALIBRATE")
    cv2.setMouseCallback("CALIBRATE", mouse_cb_calib)

    print("\nCalibration: click 4 points in order:")
    print("1) TOP-LEFT corner of arena (world 0,0)")
    print("2) TOP-RIGHT corner of arena (world W,0)")
    print("3) BOTTOM-RIGHT corner of arena (world W,H)")
    print("4) BOTTOM-LEFT corner of arena (world 0,H)")
    print("Press ESC to cancel.\n")

    while True:
        vis = clone.copy()
        for p in clicked_points:
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0,255,0), -1)
        cv2.imshow("CALIBRATE", vis)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow("CALIBRATE")
            return None
        if len(clicked_points) == 4:
            cv2.destroyWindow("CALIBRATE")
            break

    img_pts = np.array(clicked_points, dtype=np.float32)
    world_pts = np.array([
        [0.0, 0.0],
        [WORLD_W_METERS, 0.0],
        [WORLD_W_METERS, WORLD_H_METERS],
        [0.0, WORLD_H_METERS],
    ], dtype=np.float32)

    H_, _ = cv2.findHomography(img_pts, world_pts)
    return H_

def img_to_world(H_, pts_xy):
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(pts, H_).reshape(-1, 2)
    return w

# ========== Simple on-screen buttons (PICK / PLACE) ==========
BTN_W = 140
BTN_H = 50
BTN_PAD = 12

# These are module-level variables updated each frame
pick_btn = None   # (x1,y1,x2,y2)
place_btn = None  # (x1,y1,x2,y2)

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_button(frame, rect, text):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
    cv2.putText(frame, text, (x1+12, y1+33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


# ========== Mouse callback: click buttons OR click floor to set goal ==========
def mouse_cb_main(event, x, y, flags, param):
    global last_goal, last_goal_time, H, pick_btn, place_btn

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # 1) If click on PICK button:
    if pick_btn is not None and point_in_rect(x, y, pick_btn):
        send_cmd({"type": "arm", "action": "pick"})
        print("Sent ARM: PICK")
        return

    # 2) If click on PLACE button:
    if place_btn is not None and point_in_rect(x, y, place_btn):
        send_cmd({"type": "arm", "action": "place"})
        print("Sent ARM: PLACE")
        return

    # 3) Otherwise, click-to-set-goal on the floor:
    if H is None:
        print("No homography yet. Press 'c' to calibrate first.")
        return

    wx, wy = img_to_world(H, [[x, y]])[0]
    wx, wy = float(wx), float(wy)

    send_cmd({"type": "goal", "x": wx, "y": wy})
    last_goal = (wx, wy)
    last_goal_time = time.time()
    print(f"Sent GOAL: x={wx:.2f}, y={wy:.2f}")

# ==================== Main ====================
# ==============================================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Open CSV log files
pose_f = open(pose_path, "w", newline="")
events_f = open(events_path, "w", newline="")

pose_writer = csv.DictWriter(
    pose_f,
    fieldnames=["t", "valid", "x", "y", "yaw", "cx_px", "cy_px"]
)
pose_writer.writeheader()

events_writer = csv.DictWriter(
    events_f,
    fieldnames=["t", "type", "x", "y", "enabled", "action", "raw"]
)
events_writer.writeheader()

log_event("session_start", raw=f"run_dir={run_dir}")

H = load_homography()
print("Homography loaded:", H is not None)
print("Controls:")
print("  c = calibrate homography")
print("  s = START rover (enable motion)")
print("  x = STOP rover (freeze)")
print("  p = PICK (arm)")
print("  o = PLACE (arm)")
print("  left-click on PICK/PLACE buttons = arm action")
print("  left-click on floor = set GOAL")
print("  q = quit\n")

win_name = "VISION UI | c=calib s=start x=stop p=pick o=place q=quit"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, mouse_cb_main)

last_send = 0.0
SEND_HZ = 30
send_dt = 1.0 / SEND_HZ

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # Create video writer once we know frame size
    if vw is None:
        # Try to get FPS from camera; fall back if unreliable
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        if cam_fps is None or cam_fps <= 1 or cam_fps > 240:
            cam_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_path, fourcc, cam_fps, (w, h))
        print(f"[VIDEO] Recording to {video_path} @ {cam_fps:.1f} FPS ({w}x{h})")

    # Define button rectangles (top-right): PICK above PLACE
    x2 = w - BTN_PAD
    x1 = x2 - BTN_W

    y1_pick = BTN_PAD
    y2_pick = y1_pick + BTN_H

    y1_place = y2_pick + BTN_PAD
    y2_place = y1_place + BTN_H

    pick_btn = (x1, y1_pick, x2, y2_pick)
    place_btn = (x1, y1_place, x2, y2_place)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('c') or H is None:
        H_new = compute_homography(frame)
        if H_new is not None:
            H = H_new
            save_homography(H)
        else:
            print("Calibration cancelled.")
        continue

    if key == ord('s'):
        send_cmd({"type": "run", "enabled": True})
        print("Sent: START")

    if key == ord('x'):
        send_cmd({"type": "run", "enabled": False})
        print("Sent: STOP")

    if key == ord('p'):
        send_cmd({"type": "arm", "action": "pick"})
        print("Sent ARM: PICK")

    if key == ord('o'):
        send_cmd({"type": "arm", "action": "place"})
        print("Sent ARM: PLACE")

    # Detect AprilTags
    tags = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    payload = {"t": time.time(), "valid": False}

    tag = None
    for t in tags:
        if t.tag_id == TAG_ID:
            tag = t
            break

    if tag is not None and H is not None:
        cx, cy = tag.center[0], tag.center[1]
        corners = tag.corners

        world_center = img_to_world(H, [[cx, cy]])[0]
        xw, yw = float(world_center[0]), float(world_center[1])

        c0w = img_to_world(H, [corners[0]])[0]
        c1w = img_to_world(H, [corners[1]])[0]
        vx, vy = (c1w[0] - c0w[0]), (c1w[1] - c0w[1])
        yaw = float(np.arctan2(vy, vx))

        payload = {"t": time.time(), "x": xw, "y": yw, "yaw": yaw, "valid": True}

        cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        for i in range(4):
            p1 = tuple(corners[i].astype(int))
            p2 = tuple(corners[(i+1) % 4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        cv2.putText(frame, f"x={xw:.2f}m y={yw:.2f}m yaw={yaw:.2f}rad",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frame, "Tag not detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Show last goal text
    if last_goal is not None:
        gx, gy = last_goal
        cv2.putText(frame, f"GOAL: x={gx:.2f} y={gy:.2f} (click to change)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if time.time() - last_goal_time < 1.5:
            cv2.putText(frame, "Goal sent", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Draw buttons
    draw_button(frame, pick_btn, "PICK")
    draw_button(frame, place_btn, "PLACE")

    # Send pose at fixed rate
    now = time.time()
    if now - last_send >= send_dt:
        send_pose(payload)
        last_send = now

    # Log pose for offline analysis (log every frame; includes invalid states)
    if pose_writer is not None:
        pose_writer.writerow({
            "t": float(payload.get("t", time.time())),
            "valid": bool(payload.get("valid", False)),
            "x": payload.get("x", None),
            "y": payload.get("y", None),
            "yaw": payload.get("yaw", None),
            "cx_px": float(cx) if tag is not None else None,
            "cy_px": float(cy) if tag is not None else None,
        })

    cv2.imshow(win_name, frame)

    # Write annotated frame to video
    if vw is not None:
        vw.write(frame)

cap.release()
cv2.destroyAllWindows()

if vw is not None:
    vw.release()

if pose_f is not None:
    pose_f.close()
if events_f is not None:
    log_event("session_end", raw="normal_exit")
    events_f.close()

print(f"[DONE] Saved: {video_path}")
print(f"[DONE] Saved: {pose_path}")
print(f"[DONE] Saved: {events_path}")