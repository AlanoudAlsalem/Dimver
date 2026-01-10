import os
import csv
from datetime import datetime
import cv2
import numpy as np
import json
import socket
import time
from pupil_apriltags import Detector

# =========================
# CONFIG
# =========================
PI_IP = "172.20.10.3"

POSE_PORT = 5005
CMD_PORT  = 6006

TAG_ID = 1
CAM_INDEX = 1

H_FILE = "homography_floor.npz"
CALIB_FILE = "camera_calib.npz"

WORLD_W_METERS = 2.24
WORLD_H_METERS = 5.37

# --- Performance controls ---
FORCE_CAPTURE_RES = False          # Set True to force resolution (then re-calibrate!)
CAP_W, CAP_H = 1920, 1080

TAG_DET_SCALE = 0.5                # Run AprilTag on downscaled gray; 0.4 if needed

# =========================
# Checkerboard calibration 
# =========================
# 11x8 squares; 8 squares horizontal in camera view => inner corners (7,10)
PATTERN_SIZE = (7, 10)             # (cols, rows) inner corners
SQUARE_SIZE_M = 0.13383            # meters 
CALIB_MIN_SAMPLES = 25

# =========================
# Logging / Recording
# =========================
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RUNS_DIR, run_stamp)
os.makedirs(run_dir, exist_ok=True)

video_path    = os.path.join(run_dir, "video.mp4")
pose_path     = os.path.join(run_dir, "pose.csv")
events_path   = os.path.join(run_dir, "events.csv")
calibq_path   = os.path.join(run_dir, "calib_quality.csv")
calibimg_path = os.path.join(run_dir, "calib_quality.png")

pose_f = None
events_f = None
pose_writer = None
events_writer = None
vw = None

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
        "camera_calib_file": CALIB_FILE,
        "force_capture_res": FORCE_CAPTURE_RES,
        "cap_w": CAP_W,
        "cap_h": CAP_H,
        "tag_det_scale": TAG_DET_SCALE,
        "chessboard_pattern_size": list(PATTERN_SIZE),
        "square_size_m": SQUARE_SIZE_M,
        "start_time_unix": time.time(),
    }, f, indent=2)

print(f"[LOG] Run directory: {run_dir}")

# =========================
# UDP
# =========================
sock_pose = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_cmd  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_pose(payload):
    msg = json.dumps(payload).encode("utf-8")
    sock_pose.sendto(msg, (PI_IP, POSE_PORT))

def send_cmd(payload):
    payload["t"] = time.time()
    msg = json.dumps(payload).encode("utf-8")
    sock_cmd.sendto(msg, (PI_IP, CMD_PORT))

    typ = payload.get("type", "")
    if typ == "goal":
        log_event("goal", x=payload.get("x"), y=payload.get("y"))
    elif typ == "run":
        log_event("run", enabled=bool(payload.get("enabled", False)))
    elif typ == "arm":
        log_event("arm", action=payload.get("action"))
    else:
        log_event("cmd", raw=json.dumps(payload))

# =========================
# LOGGING
# =========================
def log_event(event_type, **kwargs):
    global events_writer
    if events_writer is None:
        return
    row = {"t": time.time(), "type": event_type}
    row.update(kwargs)
    events_writer.writerow(row)

# =========================
# AprilTag detector
# =========================
detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0, quad_sigma=0.0)

# =========================
# Homography utils
# =========================
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
    print(f"[H] Saved homography to {H_FILE}")

def compute_homography_live(cap, undistort_on, maps_ready, map1, map2):
    """
    LIVE homography calibration: shows live frames while you click points.
    You click 4 points in order:
      (0,0), (W,0), (W,H), (0,H)
    Press ESC to cancel.
    """
    global clicked_points
    clicked_points = []

    cv2.namedWindow("CALIBRATE_H")
    cv2.setMouseCallback("CALIBRATE_H", mouse_cb_calib)

    print("\n[H] Calibration: click 4 points in order:")
    print("1) TOP-LEFT corner of arena (world 0,0)")
    print("2) TOP-RIGHT corner of arena (world W,0)")
    print("3) BOTTOM-RIGHT corner of arena (world W,H)")
    print("4) BOTTOM-LEFT corner of arena (world 0,H)")
    print("Press ESC to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Apply undistort if enabled so H matches runtime geometry
        if undistort_on and maps_ready and map1 is not None and map2 is not None:
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        vis = frame.copy()
        for p in clicked_points:
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)

        cv2.imshow("CALIBRATE_H", vis)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow("CALIBRATE_H")
            return None

        if len(clicked_points) == 4:
            cv2.destroyWindow("CALIBRATE_H")
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

# =========================
# Camera calibration utils
# =========================
def save_camera_calib(path, K, dist, newK, roi, img_size, rms):
    np.savez(path, K=K, dist=dist, newK=newK, roi=np.array(roi), img_size=np.array(img_size), rms=rms)
    print(f"[CALIB] Saved camera calibration to {path}")

def load_camera_calib(path):
    try:
        d = np.load(path)
        K = d["K"]
        dist = d["dist"]
        newK = d["newK"]
        roi = tuple(d["roi"].tolist())
        img_size = tuple(d["img_size"].tolist())
        rms = float(d["rms"])
        print(f"[CALIB] Loaded camera calibration from {path} (RMS={rms:.4f})")
        return K, dist, newK, roi, img_size, rms
    except Exception:
        return None

def try_find_chessboard(gray):
    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners

    return False, None

def per_view_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    out = []
    for i, (objp_i, imgp_i, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objp_i, rv, tv, K, dist)
        proj = proj.reshape(-1, 2)
        obs = imgp_i.reshape(-1, 2)
        dif = obs - proj
        err = np.linalg.norm(dif, axis=1)
        out.append({
            "i": i,
            "mean_px": float(np.mean(err)),
            "max_px": float(np.max(err)),
            "rmse_px": float(np.sqrt(np.mean(err * err))),
        })
    return out

def calibrate_from_samples(objpoints, imgpoints, img_size):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=1.0, newImgSize=img_size)
    errors = per_view_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)
    return K, dist, newK, roi, float(rms), rvecs, tvecs, errors

def draw_calib_quality_viz(errors, rms, img_size):
    W = 900
    Hh = 600
    canvas = np.zeros((Hh, W, 3), dtype=np.uint8)

    cv2.putText(canvas, "Calibration Quality", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    cv2.putText(canvas, f"Overall RMS: {rms:.4f} px | img_size={img_size[0]}x{img_size[1]}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if not errors:
        cv2.putText(canvas, "No errors to display.", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        return canvas

    means = np.array([e["mean_px"] for e in errors], dtype=np.float32)
    mx = float(np.max(means))
    mn = float(np.min(means))

    bins = 12
    hist, _ = np.histogram(means, bins=bins, range=(0.0, max(1e-6, mx)))
    hist = hist.astype(np.float32)
    hist = hist / (np.max(hist) + 1e-6)

    x0, y0 = 60, 120
    x1, y1 = 840, 420
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80,80,80), 2)

    bw = (x1 - x0) / bins
    for i in range(bins):
        bx0 = int(x0 + i * bw)
        bx1 = int(x0 + (i + 1) * bw) - 2
        bh = int((y1 - y0) * hist[i])
        by0 = y1 - bh
        cv2.rectangle(canvas, (bx0, by0), (bx1, y1), (200,200,200), -1)

    cv2.putText(canvas, f"Mean reprojection error per sample (min={mn:.3f}px, max={mx:.3f}px)",
                (60, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    worst = sorted(errors, key=lambda e: e["mean_px"], reverse=True)[:5]
    cv2.putText(canvas, "Worst samples (by mean px error):", (60, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    yy = 530
    for e in worst:
        cv2.putText(canvas,
                    f"sample {e['i']:02d}: mean={e['mean_px']:.3f}px rmse={e['rmse_px']:.3f}px max={e['max_px']:.3f}px",
                    (60, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        yy += 28

    return canvas

# Prepare chessboard object points (Z=0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

objpoints = []
imgpoints = []
sample_times = []

# =========================
# Buttons (PICK / PLACE)
# =========================
BTN_W = 140
BTN_H = 50
BTN_PAD = 12

pick_btn = None
place_btn = None

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_button(frame, rect, text):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
    cv2.putText(frame, text, (x1+12, y1+33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

# =========================
# Mouse callback
# =========================
def mouse_cb_main(event, x, y, flags, param):
    global last_goal, last_goal_time, H, pick_btn, place_btn

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if pick_btn is not None and point_in_rect(x, y, pick_btn):
        send_cmd({"type": "arm", "action": "pick"})
        print("Sent ARM: PICK")
        return

    if place_btn is not None and point_in_rect(x, y, place_btn):
        send_cmd({"type": "arm", "action": "place"})
        print("Sent ARM: PLACE")
        return

    if H is None:
        print("No homography yet. Press 'c' to calibrate homography first.")
        return

    wx, wy = img_to_world(H, [[x, y]])[0]
    wx, wy = float(wx), float(wy)

    send_cmd({"type": "goal", "x": wx, "y": wy})
    last_goal = (wx, wy)
    last_goal_time = time.time()
    print(f"Sent GOAL: x={wx:.2f}, y={wy:.2f}")

# =========================
# Main
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

if FORCE_CAPTURE_RES:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

pose_f = open(pose_path, "w", newline="")
events_f = open(events_path, "w", newline="")

pose_writer = csv.DictWriter(pose_f, fieldnames=["t", "valid", "x", "y", "yaw", "cx_px", "cy_px"])
pose_writer.writeheader()

events_writer = csv.DictWriter(events_f, fieldnames=["t", "type", "x", "y", "enabled", "action", "raw"])
events_writer.writeheader()

log_event("session_start", raw=f"run_dir={run_dir}")

# Load camera calibration
calib = load_camera_calib(CALIB_FILE)
K = dist = newK = None
roi = None
calib_img_size = None
undistort_on = False

map1 = map2 = None
maps_ready = False

# Load homography
H = load_homography()

if calib is not None:
    K, dist, newK, roi, calib_img_size, rms_loaded = calib
    undistort_on = True

# UI state
calib_mode = False
show_calib_quality = False
quality_img = None

print("Homography loaded:", H is not None)
print("Camera calib loaded:", calib is not None)
print("\nControls:")
print("  m = toggle CALIBRATION MODE (runs chessboard detection each frame)")
print("  k = capture chessboard sample")
print("  K = solve camera calibration (needs >= 25 samples)")
print("  v = toggle calibration quality window (after solving)")
print("  c = calibrate homography (LIVE)")
print("  u = toggle undistort on/off (invalidates homography)")
print("  s = START rover | x = STOP rover | p = PICK | o = PLACE | q = quit")
print("  left-click floor = set GOAL; left-click buttons = arm\n")

win_name = "VISION UI"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, mouse_cb_main)

last_send = 0.0
SEND_HZ = 30
send_dt = 1.0 / SEND_HZ

cx = cy = None

# FPS overlay
fps_t0 = time.time()
fps_count = 0
fps_val = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # Build undistort maps once (fast path)
    if (not maps_ready) and undistort_on and (K is not None) and (dist is not None) and (newK is not None):
        if calib_img_size is not None and calib_img_size != (w, h):
            print(f"[WARN] Saved camera calib img_size={calib_img_size}, current={(w,h)}. "
                  "Recalibrate if you changed resolution.")
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
        maps_ready = True
        print("[UNDIST] Undistort maps ready.")

    # Video writer once
    if vw is None:
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        if cam_fps is None or cam_fps <= 1 or cam_fps > 240:
            cam_fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_path, fourcc, cam_fps, (w, h))
        print(f"[VIDEO] Recording to {video_path} @ {cam_fps:.1f} FPS ({w}x{h})")

    # Buttons
    x2 = w - BTN_PAD
    x1 = x2 - BTN_W
    y1_pick = BTN_PAD
    y2_pick = y1_pick + BTN_H
    y1_place = y2_pick + BTN_PAD
    y2_place = y1_place + BTN_H
    pick_btn = (x1, y1_pick, x2, y2_pick)
    place_btn = (x1, y1_place, x2, y2_place)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('m'):
        calib_mode = not calib_mode
        print(f"[CALIB_MODE] {'ON' if calib_mode else 'OFF'}")

    if key == ord('v'):
        show_calib_quality = not show_calib_quality
        if show_calib_quality and quality_img is None:
            print("[CALIB_QUALITY] No calibration quality image yet. Solve calibration with 'K' first.")
        else:
            print(f"[CALIB_QUALITY] {'SHOW' if show_calib_quality else 'HIDE'}")

    if key == ord('u'):
        if calib is None:
            print("[UNDIST] No camera calibration loaded yet.")
        else:
            undistort_on = not undistort_on
            print(f"[UNDIST] Undistort {'ON' if undistort_on else 'OFF'}")
            maps_ready = False
            map1 = map2 = None
            H = None
            print("[UNDIST] Homography invalidated. Press 'c' to recalibrate homography.")

    # Apply fast undistortion if enabled
    if undistort_on and maps_ready and map1 is not None and map2 is not None:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Homography calibration (LIVE) ONLY when user presses 'c'
    if key == ord('c'):
        H_new = compute_homography_live(cap, undistort_on, maps_ready, map1, map2)
        if H_new is not None:
            H = H_new
            save_homography(H)
        else:
            print("[H] Calibration cancelled.")
        continue

    # Calibration mode: detect chessboard every frame (for guidance)
    board_found = False
    board_corners = None
    if calib_mode:
        board_found, board_corners = try_find_chessboard(gray)
        if board_found:
            cv2.drawChessboardCorners(frame, PATTERN_SIZE, board_corners, True)

    # Capture a sample
    if key == ord('k'):
        if calib_mode:
            found, corners = board_found, board_corners
        else:
            found, corners = try_find_chessboard(gray)

        if found:
            imgpoints.append(corners)
            objpoints.append(objp.copy())
            sample_times.append(time.time())
            print(f"[CALIB] Captured sample {len(imgpoints)}")
        else:
            print("[CALIB] Chessboard not found in this frame.")

    # Solve camera calibration
    if key == ord('K'):
        if len(imgpoints) < CALIB_MIN_SAMPLES:
            print(f"[CALIB] Need at least {CALIB_MIN_SAMPLES} samples, have {len(imgpoints)}")
        else:
            img_size = (w, h)
            K, dist, newK, roi, rms, rvecs, tvecs, errors = calibrate_from_samples(objpoints, imgpoints, img_size)
            save_camera_calib(CALIB_FILE, K, dist, newK, roi, img_size, rms)
            calib = (K, dist, newK, roi, img_size, rms)

            undistort_on = True
            maps_ready = False
            map1 = map2 = None

            # Must redo homography after camera calibration
            H = None

            # Log per-sample errors
            with open(calibq_path, "w", newline="") as fcsv:
                wcsv = csv.DictWriter(fcsv, fieldnames=["sample_i", "t_unix", "mean_px", "rmse_px", "max_px"])
                wcsv.writeheader()
                for e in errors:
                    i = e["i"]
                    t_s = sample_times[i] if i < len(sample_times) else None
                    wcsv.writerow({
                        "sample_i": i,
                        "t_unix": float(t_s) if t_s is not None else None,
                        "mean_px": e["mean_px"],
                        "rmse_px": e["rmse_px"],
                        "max_px": e["max_px"],
                    })
            print(f"[CALIB] Per-sample reprojection errors saved: {calibq_path}")

            # Quality visualization
            quality_img = draw_calib_quality_viz(errors, rms, img_size)
            cv2.imwrite(calibimg_path, quality_img)
            print(f"[CALIB] Quality visualization saved: {calibimg_path}")

            print("[CALIB] Calibration complete. Homography invalidated.")
            print("[CALIB] Now press 'c' to recalibrate homography (LIVE).")
            print(f"[CALIB] Overall RMS = {rms:.4f} px")

    # Rover commands
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

    # AprilTag detection (downscaled)
    scale = float(TAG_DET_SCALE)
    if 0.05 < scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        tags = detector.detect(small, estimate_tag_pose=False, camera_params=None, tag_size=None)
    else:
        tags = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    payload = {"t": time.time(), "valid": False}

    tag = None
    for t in tags:
        if t.tag_id == TAG_ID:
            tag = t
            break

    if tag is not None and H is not None:
        if 0.05 < scale < 1.0:
            cx = float(tag.center[0] / scale)
            cy = float(tag.center[1] / scale)
            corners_tag = (tag.corners / scale).astype(np.float32)
        else:
            cx = float(tag.center[0])
            cy = float(tag.center[1])
            corners_tag = tag.corners.astype(np.float32)

        world_center = img_to_world(H, [[cx, cy]])[0]
        xw, yw = float(world_center[0]), float(world_center[1])

        c0w = img_to_world(H, [corners_tag[0]])[0]
        c1w = img_to_world(H, [corners_tag[1]])[0]
        vx, vy = (c1w[0] - c0w[0]), (c1w[1] - c0w[1])
        yaw = float(np.arctan2(vy, vx))

        payload = {"t": time.time(), "x": xw, "y": yw, "yaw": yaw, "valid": True}

        cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        for i in range(4):
            p1 = tuple(corners_tag[i].astype(int))
            p2 = tuple(corners_tag[(i+1) % 4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        cv2.putText(frame, f"x={xw:.2f}m y={yw:.2f}m yaw={yaw:.2f}rad",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frame, "Tag not detected OR no homography", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Last goal
    if last_goal is not None:
        gx, gy = last_goal
        cv2.putText(frame, f"GOAL: x={gx:.2f} y={gy:.2f} (click to change)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if time.time() - last_goal_time < 1.5:
            cv2.putText(frame, "Goal sent", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Buttons
    draw_button(frame, pick_btn, "PICK")
    draw_button(frame, place_btn, "PLACE")

    # FPS
    fps_count += 1
    now = time.time()
    if now - fps_t0 >= 1.0:
        fps_val = fps_count / (now - fps_t0)
        fps_count = 0
        fps_t0 = now

    # HUD
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Undistort: {'ON' if undistort_on else 'OFF'} | TagScale: {TAG_DET_SCALE}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"H: {'OK' if H is not None else 'MISSING'} | CalibMode: {'ON' if calib_mode else 'OFF'} | Samples: {len(imgpoints)}",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if calib_mode:
        cv2.putText(frame, f"Board: {'FOUND' if board_found else 'not found'}",
                    (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Send pose
    if now - last_send >= send_dt:
        send_pose(payload)
        last_send = now

    # Log pose
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

    # Show windows
    cv2.imshow(win_name, frame)

    if show_calib_quality and quality_img is not None:
        cv2.imshow("CALIB_QUALITY", quality_img)
    else:
        try:
            cv2.destroyWindow("CALIB_QUALITY")
        except Exception:
            pass

    # Record annotated video
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
print(f"[DONE] If you calibrated: {calibq_path} and {calibimg_path}")
