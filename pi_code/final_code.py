import time
import json
import math
import socket
import threading
from queue import Queue, Empty

import serial
import serial.tools.list_ports

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo
from time import sleep


# ========================= USER CONFIG =========================
BAUD = 115200

POSE_UDP_PORT = 5005          # from laptop vision
CMD_UDP_PORT  = 6006          # from laptop UI (goal/run/arm)

POSE_STALE_SEC = 0.35

# ---- Telemetry back to laptop ----
LAPTOP_IP = "172.20.10.5"     # <-- set this to your laptop IP
TEL_PORT  = 7007
tel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Default goal (can be overwritten live)
GOAL_X_DEFAULT = 0.0
GOAL_Y_DEFAULT = -0.6

GOAL_POS_TOL = 0.06
TURN_SIGN = -1
YAW_OFFSET = -1.75

# Motor / avoidance tuning
LOOP_HZ = 20
BASE_FWD = 120
TURN = 110
IR_NEAR = 350

# Go-to-goal controller gains
K_W = 2.2
K_V = 220
V_MAX = 140
W_MAX = 120

# ========================= ARM CONFIG =========================
Device.pin_factory = PiGPIOFactory()

MIN_PULSE = 0.001000
MAX_PULSE = 0.002000

BASE_CENTER = -40.0
BASE_TARGET = -40.0

ARM_NEUTRAL = -30.0
ARM_DOWN    = -85.0
ARM_UP      = 45.0

GRIP_OPEN   = -85.0
GRIP_CLOSE  = -35.0

SMOOTH_STEP_DEG = 2
SMOOTH_DT_S = 0.020000

# ========================= Serial bridge =========================
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "ACM" in p.device or "USB" in p.device:
            return p.device
    return ports[0].device if ports else None

class MBotBridge:
    def __init__(self, port, baud=BAUD, timeout=0.2):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(1.0)
        self.ser.reset_input_buffer()

    def send_line(self, s: str):
        self.ser.write((s.strip() + "\n").encode("utf-8"))

    def read_line(self):
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        return line if line else None

    def set_motors(self, m1, m2, m3, m4):
        self.send_line(f"M {int(m1)} {int(m2)} {int(m3)} {int(m4)}")
        _ = self.read_line()

    def estop(self):
        self.send_line("E")
        _ = self.read_line()

    def get_sensors(self):
        self.send_line("S")
        line = self.read_line()
        if not line:
            return None
        parts = line.split()
        if len(parts) != 8 or parts[0] != "S":
            return None
        return {
            "irL": int(parts[1]),
            "irC": int(parts[2]),
            "irR": int(parts[3]),
            "impact1": int(parts[4]),  # active-low
            "impact2": int(parts[5]),  # active-low
            "line1": int(parts[6]),
            "line2": int(parts[7]),
        }

def clamp(v, lo=-255, hi=255):
    return lo if v < lo else hi if v > hi else v

def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi


# ========================= Pose receiver (UDP) =========================
class PoseReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.pose = {"valid": False, "t": 0.0}

    def poll(self):
        while True:
            try:
                data, _addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                self.pose = msg
            except Exception:
                pass

    def get(self):
        self.poll()
        if not self.pose.get("valid", False):
            return None
        age = time.time() - float(self.pose.get("t", 0.0))
        if age > POSE_STALE_SEC:
            return None
        return self.pose

# ========================= Command receiver (UDP) =========================
class CommandReceiver:
    """
    Expected JSON messages from laptop:
      {"type":"goal", "x":0.5, "y":0.2}
      {"type":"run", "enabled": true}
      {"type":"stop"}  (alias to run false)
      {"type":"arm", "action":"pick"}   # NEW
      {"type":"arm", "action":"place"}  # NEW
    """
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

    def poll(self):
        msgs = []
        while True:
            try:
                data, _addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            try:
                msgs.append(json.loads(data.decode("utf-8")))
            except Exception:
                pass
        return msgs


# ========================= Arm controller (non-blocking) =========================
class ArmController:
    def __init__(self):
        self.base_servo = AngularServo(
            18, min_angle=-90, max_angle=90,
            min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE
        )
        self.vertical_servo = AngularServo(
            13, min_angle=-90, max_angle=90,
            min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE
        )
        self.gripper_servo = AngularServo(
            12, min_angle=-90, max_angle=90,
            min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE
        )

        self.queue = Queue()
        self.busy = False
        self.holding = False
        self.last_action = "idle"
        self.last_error = None

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def move_smooth(self, servo, target, step=SMOOTH_STEP_DEG, dt=SMOOTH_DT_S):
        cur = servo.angle
        if cur is None:
            cur = 0.0

        step = max(1, int(step))
        target = max(-90.0, min(90.0, float(target)))

        if target > cur:
            a = cur
            while a < target:
                a = min(target, a + step)
                servo.angle = a
                sleep(dt)
        else:
            a = cur
            while a > target:
                a = max(target, a - step)
                servo.angle = a
                sleep(dt)

    def neutral(self):
        self.move_smooth(self.base_servo, BASE_CENTER)
        self.move_smooth(self.vertical_servo, ARM_NEUTRAL)
        self.move_smooth(self.gripper_servo, GRIP_OPEN)
        sleep(0.2)

    def pick(self):
        """
        Pick up object and keep holding it (does NOT place).
        Safe to call multiple times; if already holding, it will just re-grab.
        """
        self.last_action = "pick"
        self.neutral()

        # rotate to target (if needed)
        self.move_smooth(self.base_servo, BASE_TARGET)
        sleep(0.2)

        # lower
        self.move_smooth(self.vertical_servo, ARM_DOWN)
        sleep(0.2)

        # close gripper (grab)
        self.move_smooth(self.gripper_servo, GRIP_CLOSE, step=1, dt=SMOOTH_DT_S)
        sleep(0.2)

        # lift
        self.move_smooth(self.vertical_servo, ARM_UP)
        sleep(0.2)

        self.holding = True

    def place(self):
        """
        Place object down (opens gripper). If not holding, it still performs a place motion.
        """
        self.last_action = "place"

        # go to target
        self.move_smooth(self.base_servo, BASE_TARGET)
        sleep(0.2)

        # lower
        self.move_smooth(self.vertical_servo, ARM_DOWN)
        sleep(0.2)

        # open gripper (release)
        self.move_smooth(self.gripper_servo, GRIP_OPEN, step=1, dt=SMOOTH_DT_S)
        sleep(0.2)

        # lift back up, then neutral
        self.move_smooth(self.vertical_servo, ARM_UP)
        sleep(0.2)

        self.neutral()
        self.holding = False

    def enqueue(self, action: str):
        action = (action or "").strip().lower()
        if action not in ("pick", "place"):
            return False
        self.queue.put(action)
        return True

    def _run(self):
        while True:
            try:
                action = self.queue.get(timeout=0.1)
            except Empty:
                continue

            self.busy = True
            self.last_error = None
            try:
                if action == "pick":
                    self.pick()
                elif action == "place":
                    self.place()
            except Exception as e:
                self.last_error = str(e)
            finally:
                self.busy = False
                self.queue.task_done()

# ========================= Main controller =========================
def main():
    port = find_serial_port()
    if not port:
        raise RuntimeError("No serial port found. Check USB connection.")
    print("Using serial port:", port)

    robot = MBotBridge(port)
    robot.estop()
    time.sleep(0.2)

    # Arm
    arm = ArmController()
    print("ArmController initialized.")

    pose_rx = PoseReceiver(POSE_UDP_PORT)
    cmd_rx  = CommandReceiver(CMD_UDP_PORT)

    goal_x = GOAL_X_DEFAULT
    goal_y = GOAL_Y_DEFAULT
    run_enabled = False  # start stationary by default

    # If arm is busy we force motors to 0 (and temporarily ignore run)
    run_before_arm = False
    arm_lock = False

    last_tel = 0.0
    TEL_HZ = 20
    tel_dt = 1.0 / TEL_HZ

    print(f"Pose UDP: {POSE_UDP_PORT} | Command UDP: {CMD_UDP_PORT}")
    print("Starting stationary. Use laptop UI to set goal, START, and PICK/PLACE.")

    def set_drive(forward, turn):
        left_cmd  = clamp(forward - turn)
        right_cmd = clamp(forward + turn)

        # M1=FR, M2=BR, M3=BL (flip), M4=FL (flip)
        m1 = right_cmd
        m2 = right_cmd
        m3 = -left_cmd
        m4 = -left_cmd
        robot.set_motors(m1, m2, m3, m4)

    dt = 1.0 / LOOP_HZ
    try:
        while True:
            loop_t = time.time()

            # ----- Receive commands -----
            for msg in cmd_rx.poll():
                mtype = msg.get("type", "")

                if mtype == "goal":
                    try:
                        goal_x = float(msg["x"])
                        goal_y = float(msg["y"])
                        print(f"\nNew goal set: ({goal_x:.2f}, {goal_y:.2f})")
                    except Exception:
                        pass

                elif mtype == "run":
                    # If arm is actively running an action, ignore run toggles for safety
                    if arm_lock or arm.busy:
                        continue
                    run_enabled = bool(msg.get("enabled", False))
                    print(f"\nRun enabled: {run_enabled}")
                    if not run_enabled:
                        set_drive(0, 0)

                elif mtype == "stop":
                    run_enabled = False
                    print("\nRun enabled: False")
                    set_drive(0, 0)

                elif mtype == "arm":
                    action = str(msg.get("action", "")).lower().strip()
                    if action in ("pick", "place"):
                        ok = arm.enqueue(action)
                        if ok:
                            # latch lock; we will stop rover while arm is executing
                            if not arm_lock and not arm.busy:
                                run_before_arm = run_enabled
                            arm_lock = True
                            print(f"\nARM command enqueued: {action}")

            # If arm just finished, release lock (motors remain stopped until user STARTs again)
            if arm_lock and (not arm.busy) and arm.queue.empty():
                arm_lock = False
                run_enabled = False
                set_drive(0, 0)
                print("\nARM done. Rover kept STOPPED for safety. Press START on laptop to move again.")

            # Always get sensors + pose
            s = robot.get_sensors()
            pose = pose_rx.get()

            # ----- Telemetry -----
            now = time.time()
            if now - last_tel >= tel_dt:
                tel = {
                    "t": now,
                    "pose_valid": bool(pose is not None),
                    "x": float(pose["x"]) if pose else None,
                    "y": float(pose["y"]) if pose else None,
                    "yaw_tag": float(pose["yaw"]) if pose else None,
                    "goal_x": float(goal_x),
                    "goal_y": float(goal_y),
                    "run_enabled": bool(run_enabled),

                    "arm_busy": bool(arm.busy or (not arm.queue.empty())),
                    "arm_holding": bool(arm.holding),
                    "arm_last_action": arm.last_action,
                    "arm_last_error": arm.last_error,
                }

                if s is not None:
                    tel.update({
                        "irL": int(s["irL"]),
                        "irC": int(s["irC"]),
                        "irR": int(s["irR"]),
                        "impact1": int(s["impact1"]),
                        "impact2": int(s["impact2"]),
                    })
                else:
                    tel.update({
                        "irL": None, "irC": None, "irR": None,
                        "impact1": None, "impact2": None,
                    })

                tel_sock.sendto(json.dumps(tel).encode("utf-8"), (LAPTOP_IP, TEL_PORT))
                last_tel = now

            # If sensors missing, do nothing
            if s is None:
                continue

            # If arm is busy/locked => force stop motors
            if arm_lock or arm.busy:
                set_drive(0, 0)
                elapsed = time.time() - loop_t
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            irL, irC, irR = s["irL"], s["irC"], s["irR"]
            obsL = (irL < IR_NEAR)
            obsC = (irC < IR_NEAR)
            obsR = (irR < IR_NEAR)

            impact_pressed = (s["impact1"] == 0) or (s["impact2"] == 0)

            # If not running, stay stationary
            if not run_enabled:
                set_drive(0, 0)
                elapsed = time.time() - loop_t
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # ----- Safety: impact -----
            if impact_pressed:
                set_drive(0, 0)
                time.sleep(0.20)
                set_drive(+BASE_FWD, 0)
                time.sleep(0.25)
                set_drive(0, 0)
                continue

            # ----- Reactive avoidance override -----
            if obsC or obsL or obsR or pose is None:
                if obsC or (obsL and obsR):
                    if irL < irR:
                        set_drive(0, -TURN)
                    else:
                        set_drive(0, +TURN)
                elif obsL:
                    set_drive(BASE_FWD, -TURN * 0.6)
                elif obsR:
                    set_drive(BASE_FWD, +TURN * 0.6)
                else:
                    set_drive(80, 0)

            else:
                # ----- Go-to-goal -----
                yaw_tag = float(pose["yaw"])
                yaw = wrap_pi(yaw_tag + YAW_OFFSET)

                x = float(pose["x"])
                y = float(pose["y"])

                dx = goal_x - x
                dy = goal_y - y
                dist = math.hypot(dx, dy)

                desired_yaw = math.atan2(dy, dx)
                yaw_err = wrap_pi(desired_yaw - yaw)

                w_cmd = clamp(int(TURN_SIGN * (K_W * yaw_err * 100)), -W_MAX, W_MAX)

                if dist < GOAL_POS_TOL:
                    set_drive(0, 0)
                else:
                    v_cmd = clamp(int(K_V * dist), 0, V_MAX)
                    if abs(yaw_err) > 0.35:
                        v_cmd = 0
                    set_drive(v_cmd, w_cmd)

            elapsed = time.time() - loop_t
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.estop()

if __name__ == "__main__":
    main()