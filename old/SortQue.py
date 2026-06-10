"""
main_pipeline.py  —  Sequential classify-then-sort pipeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SEQUENTIAL (not parallel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The belt motor and the sort conveyor motor are the SAME physical
  motor (motorCtrlPin). You cannot rotate the belt to bring the
  next plate to the camera while the sort conveyor is still running
  — they'd fight over the same motor.

  Therefore the flow MUST be:
    ① send "next:"       → belt rotates plate to camera → CAM_STOP
    ② weigh + YOLO       → classify → assign bin
    ③ send "trayPos:N"   → Arduino sorts (motor runs to bin, servo fires)
    ④ wait SORT_DONE     → sort complete, motor free
    ⑤ loop to ①

  The queue still exists on Python side: if classification is fast
  and multiple plates need sorting, they queue up and execute one
  at a time. But "next:" is NEVER sent while a sort is in progress.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERIAL TOKENS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Python sends:  "next:", "trayPos:N"
  Arduino sends: "CAM_STOP" (belt arrived), "SORT_DONE" (sort complete)

  Single thread reads serial — no race conditions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from ultralytics import YOLO
import cv2
import numpy as np
from ardcommsTest import arduinoCommunication
import time
import mysql.connector
import sys
import queue
import threading
import os
from enum import Enum
import requests

from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QWidget, QMessageBox, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import uic
import torch

print("Application Starting")
os.makedirs("captures", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
class Config:
    CONF_THRESHOLD      = 0.75   # min conf to COUNT a finger (3/4/5 grading)
    PRESENCE_CONF       = 0.35   # min conf to say "a banana is present at all"
    IOU_THRESHOLD       = 0.60
    MASK_ALPHA          = 0.50
    CAPTURE_FRAMES      = 5
    CAPTURE_INTERVAL_MS = 150
    MIN_VALID_FRAMES    = 3
    FLUSH_FRAMES        = 5
    FLUSH_DELAY_MS      = 60
    ZOOM_FACTOR         = 1.3   # digital center-crop zoom (1.0 = no zoom)

    WEIGHT_THRESHOLD_G  = 5.0
    WEIGHT_STABLE_N     = 4
    WEIGHT_EMPTY_N      = 5       # consecutive near-zero reads ⇒ empty plate (fast reject)
    WEIGHT_TIMEOUT_S    = 20
    MIN_VALID_WEIGHT_G  = 320.0  # lowest GRADE is 350g; lighter bananas are out-of-spec, not "empty"
    EMPTY_MAX_G         = 50.0   # < this (near-zero) ⇒ truly empty plate; ≥ this ⇒ a banana is present → let YOLO classify it
    MAX_VALID_WEIGHT_G  = 1500.0
    WEIGHT_SETTLE_S     = 3.5   # EMA α=0.25 needs ~3s to converge after banana lands

    W_25BCP_MIN  = 621;  W_25BCP_MAX  = 730
    W_30BCP_MIN  = 521;  W_30BCP_MAX  = 620
    W_33BCP_MIN  = 400;  W_33BCP_MAX  = 520
    W_30TR_MIN   = 541;  W_30TR_MAX   = 650
    W_F36TR_MIN  = 466;  W_F36TR_MAX  = 540
    W_IF38TR_MIN = 350;  W_IF38TR_MAX = 465

    FIREBASE_URL       = "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app"
    FIREBASE_TIMEOUT_S = 5
    FIREBASE_RETRY     = 3

    ARDUINO_PORT      = "COM5"
    ARDUINO_BAUD      = 115200
    SCALE_TIMEOUT_S   = 60    # max wait for next SCALE_STOP event
    CLASSIFY_COOLDOWN_S = 0.1


# ─────────────────────────────────────────────────────
# FIREBASE
# ─────────────────────────────────────────────────────
firebase_connected = False

def testFirebaseConnection():
    global firebase_connected
    try:
        r = requests.get(f"{Config.FIREBASE_URL}/.json",
                         timeout=Config.FIREBASE_TIMEOUT_S)
        firebase_connected = r.status_code == 200
    except Exception as e:
        print(f"✗ Firebase: {e}")
        firebase_connected = False
    print(f"{'✓' if firebase_connected else '✗'} Firebase")
    return firebase_connected


def getRawWeightFromFirebase():
    """Raw weight reading with no validity gating.
       Returns grams (float), or None if Firebase couldn't be read.
       Needed so waitForStableWeight can tell a stable *low* reading
       (empty/too-light plate) apart from a missing/failed read."""
    for attempt in range(Config.FIREBASE_RETRY):
        try:
            r = requests.get(f"{Config.FIREBASE_URL}/Weight.json",
                            timeout=Config.FIREBASE_TIMEOUT_S)
            if r.status_code == 200 and r.json() is not None:
                return float(r.json())
        except requests.exceptions.Timeout:
            print(f"  Firebase timeout ({attempt+1}/{Config.FIREBASE_RETRY})")
        except Exception as e:
            print(f"  Firebase error: {e}")
        if attempt < Config.FIREBASE_RETRY - 1:
            time.sleep(0.3)
    return None


def getWeightFromFirebase() -> float:
    raw = getRawWeightFromFirebase()
    if raw is None:
        return -1
    if Config.MIN_VALID_WEIGHT_G <= raw <= Config.MAX_VALID_WEIGHT_G:
        return raw
    return -1

def waitForStableWeight(stop_flag=None) -> tuple:
    """Poll Firebase for a stable plate weight.

    Returns (weight, status):
      status "ok"      → stable reading ≥ EMPTY_MAX_G; weight is the avg. This
                         INCLUDES light single/double bananas that sit below the
                         grading floor — they must still reach YOLO so it can
                         count them and mark them "Out of Specification".
      status "empty"   → weight stayed stably near zero (< EMPTY_MAX_G) ⇒ a truly
                         empty plate. Fast reject, no 20 s timeout.
      status "timeout" → no stable reading within WEIGHT_TIMEOUT_S (sensor/Firebase issue)

    Banana *presence* is decided by YOLO, not by the grading weight floor — so
    only a near-empty plate is rejected here.
    """
    readings     = []
    empty_streak = 0
    start = time.time()
    while (time.time() - start) < Config.WEIGHT_TIMEOUT_S:
        if stop_flag and stop_flag.is_set():
            return -1, "timeout"
        raw = getRawWeightFromFirebase()
        if raw is None or raw > Config.MAX_VALID_WEIGHT_G:
            empty_streak = 0        # unreadable / over-range — not an "empty" vote
            time.sleep(0.3); continue
        if raw < Config.EMPTY_MAX_G:
            empty_streak += 1
            readings.clear()
            if empty_streak >= Config.WEIGHT_EMPTY_N:
                print(f"  ✓ Empty plate: {empty_streak} reads < "
                      f"{Config.EMPTY_MAX_G:.0f}g (last {raw:.1f}g)")
                return -1, "empty"
            time.sleep(0.3); continue
        empty_streak = 0
        readings.append(raw)
        if len(readings) > Config.WEIGHT_STABLE_N:
            readings.pop(0)
        if len(readings) == Config.WEIGHT_STABLE_N:
            variation = max(readings) - min(readings)
            if variation <= Config.WEIGHT_THRESHOLD_G:
                avg = sum(readings) / len(readings)
                print(f"  ✓ Weight: {avg:.1f}g  (var {variation:.1f}g)")
                return avg, "ok"
        time.sleep(0.3)
    print("  ✗ Weight timeout")
    return -1, "timeout"


# ─────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────
DB_CONFIG = dict(host="localhost", user="root", password="Password1", database="grade")

def _testDbConnection():
    try:
        c = mysql.connector.connect(**DB_CONFIG)
        c.close()
        print("Database: OK")
        return True
    except mysql.connector.Error as e:
        print(f"Database: connection failed — {e}")
        return False

_testDbConnection()

def saveToDatabase(farm, cls, weight, finger, size, conf, x1, y1, x2, y2):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute(
            """INSERT INTO finger_classes
               (Farm, Classes, weight, classes_name, size, conf, x1, y1, x2, y2)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (farm, cls, weight, finger, size, conf, x1, y1, x2, y2))
        conn.commit(); cur.close(); conn.close()
        print(f"  ✓ DB: {cls}")
    except mysql.connector.Error as e:
        print(f"  ✗ DB error: {e}")


# ─────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────
model:   YOLO | None                 = None
arduino: arduinoCommunication | None = None

def loadModel():
    global model
    model = YOLO("weights/segment1.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=Config.CONF_THRESHOLD,
                  iou=Config.IOU_THRESHOLD, save=False, verbose=False)
    print("YOLO model loaded and warmed up")

def startArduino():
    global arduino
    arduino = arduinoCommunication(Config.ARDUINO_PORT, Config.ARDUINO_BAUD)
    print("Arduino: OK")


# ─────────────────────────────────────────────────────
# CLASSIFICATION ENUMS & LOGIC
# ─────────────────────────────────────────────────────
class FingerCount(Enum):
    THREE   = "3-finger"
    FOUR    = "4-finger"
    FIVE    = "5-finger"
    UNKNOWN = "unknown"

class HandSize(Enum):
    REGULAR = "regular"
    SMALL   = "small"
    UNKNOWN = "unknown"

class BananaClass(Enum):
    C25BCP  = "25BCP"
    C30BCP  = "30BCP"
    C33BCP  = "33BCP"
    C30TR   = "30TR"
    CF36TR  = "IF36TR"
    CIF38TR = "IF38TR"
    UNKNOWN = "Out of Specification"

CLASS_TO_BIN = {
    BananaClass.C33BCP:  1,
    BananaClass.C25BCP:  2,
    BananaClass.C30BCP:  3,
    BananaClass.CIF38TR: 4,
    BananaClass.CF36TR:  5,
    BananaClass.C30TR:   6,
}

def parseFinger(label: str) -> FingerCount:
    s = str(label).strip().lower()
    if "3" in s: return FingerCount.THREE
    if "4" in s: return FingerCount.FOUR
    if "5" in s: return FingerCount.FIVE
    return FingerCount.UNKNOWN

def inferHand(finger: FingerCount, weight: float) -> HandSize:
    c = Config
    if finger == FingerCount.THREE:
        if c.W_IF38TR_MIN <= weight <= c.W_30TR_MAX: return HandSize.REGULAR
    elif finger in (FingerCount.FOUR, FingerCount.FIVE):
        if c.W_25BCP_MIN <= weight <= c.W_25BCP_MAX: return HandSize.REGULAR
        if c.W_30BCP_MIN <= weight <= c.W_30BCP_MAX: return HandSize.REGULAR
        if c.W_33BCP_MIN <= weight <= c.W_33BCP_MAX: return HandSize.SMALL
    return HandSize.UNKNOWN

def classifyBanana(finger: FingerCount, weight: float):
    if finger == FingerCount.UNKNOWN:
        return BananaClass.UNKNOWN, HandSize.UNKNOWN
    hand = inferHand(finger, weight)
    if hand == HandSize.UNKNOWN:
        return BananaClass.UNKNOWN, HandSize.UNKNOWN
    c = Config
    if finger in (FingerCount.FOUR, FingerCount.FIVE):
        if c.W_25BCP_MIN <= weight <= c.W_25BCP_MAX: return BananaClass.C25BCP, hand
        if c.W_30BCP_MIN <= weight <= c.W_30BCP_MAX: return BananaClass.C30BCP, hand
        if c.W_33BCP_MIN <= weight <= c.W_33BCP_MAX: return BananaClass.C33BCP, hand
    if finger == FingerCount.THREE:
        if c.W_30TR_MIN   <= weight <= c.W_30TR_MAX:  return BananaClass.C30TR,   hand
        if c.W_F36TR_MIN  <= weight <= c.W_F36TR_MAX: return BananaClass.CF36TR,  hand
        if c.W_IF38TR_MIN <= weight <= c.W_IF38TR_MAX:return BananaClass.CIF38TR, hand
    return BananaClass.UNKNOWN, hand


# ─────────────────────────────────────────────────────
# CAPTURE IMAGE
# ─────────────────────────────────────────────────────
COLORS = [
    (255,230,0),(255,80,80),(80,200,255),
    (180,0,255),(80,255,100),(255,160,0),
]

def captureImage(get_frame_fn) -> dict:
    res = {"finger":[-1,"invalid"],"status":"no_banana",
           "x1":0,"y1":0,"x2":0,"y2":0,"image_path":""}
    for _ in range(Config.FLUSH_FRAMES):
        get_frame_fn(); time.sleep(Config.FLUSH_DELAY_MS/1000)

    vote_counts = {}
    best_by_label = {}   # label → (conf, box, frame, result)
    frames_read        = 0   # frames grabbed from the camera (ret == True)
    frames_with_banana = 0   # frames where YOLO found ≥1 banana mask (any count)

    for i in range(Config.CAPTURE_FRAMES):
        ret, frame = get_frame_fn()
        if not ret or frame is None:
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        frames_read += 1
        # Detect at the low PRESENCE_CONF so even a faint/occluded banana
        # registers — this is the dedicated "banana vs no-banana" check.
        result = model.predict(source=frame, conf=Config.PRESENCE_CONF,
                               iou=Config.IOU_THRESHOLD, save=False, verbose=False)[0]
        if result.masks is None or len(result.masks.xy) == 0:
            # Nothing at all, even at low conf → empty plate for this frame.
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        confs_all     = result.boxes.conf
        present_masks = [m for m in result.masks.xy if len(m) >= 3]
        if not present_masks:
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        frames_with_banana += 1   # a banana IS on the plate (≥ PRESENCE_CONF)
        # Count fingers using only confident masks (≥ CONF_THRESHOLD).
        count_masks  = [m for m, cf in zip(result.masks.xy, confs_all)
                        if len(m) >= 3 and float(cf) >= Config.CONF_THRESHOLD]
        banana_count = len(count_masks)
        if banana_count == 0:
            # masks seen (≥ PRESENCE_CONF) but none confident enough to count
            print(f"  Frame {i+1}: banana present ({len(present_masks)} mask) "
                  f"but no confident finger — unreadable")
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        # 3/4/5 are gradeable; 1, 2, 6+ are confidently counted but out of the
        # gradeable range — label them generically so they still vote and flow
        # to classify, which maps any non-3/4/5 finger to "Out of Specification".
        label = {3:"3-finger",4:"4-finger",5:"5-finger"}.get(
            banana_count, f"{banana_count}-finger")
        vote_counts[label] = vote_counts.get(label,0) + 1
        confs = confs_all
        if len(confs) > 0:
            idx = int(confs.argmax()); conf_val = float(confs[idx])
            prev = best_by_label.get(label)
            if prev is None or conf_val > prev[0]:
                best_by_label[label] = (conf_val,
                                        list(map(int, result.boxes.xyxy[idx])),
                                        frame.copy(), result)
        print(f"  Frame {i+1}: {label}  conf={float(confs.max()):.2f}")
        time.sleep(Config.CAPTURE_INTERVAL_MS/1000)

    if frames_read == 0:
        print("  ✗ Camera read failed — no frames")
        res["status"] = "camera_fail"; return res
    if not vote_counts:
        # YOLO ran but never produced a usable 3/4/5 finger count.
        # Use mask presence to tell "empty plate" from "unreadable banana".
        if frames_with_banana > 0:
            print("  ✗ Banana present but finger count unreadable")
            res["status"] = "detect_fail"
        else:
            print("  ✗ No banana detected on plate")
            res["status"] = "no_banana"
        return res
    winner    = max(vote_counts, key=vote_counts.get)
    win_votes = vote_counts[winner]
    total     = sum(vote_counts.values())
    print(f"  Votes: {vote_counts} → {winner} ({win_votes}/{total})")
    if win_votes < Config.MIN_VALID_FRAMES:
        # We had valid counts, just not enough agreement — banana is present.
        print(f"  ✗ Weak consensus — banana seen, count unstable")
        res["status"] = "detect_fail"; return res

    best_conf, best_box, best_frame, best_result = best_by_label.get(
        winner, (0.0, None, None, None))

    res["finger"] = [best_conf, winner]
    res["status"] = "ok"
    if best_box:
        res["x1"],res["y1"] = best_box[0],best_box[1]
        res["x2"],res["y2"] = best_box[2],best_box[3]

    if best_frame is not None and best_box:
        overlay = best_frame.copy()
        ann = best_result   # reuse cached result — no second inference
        if ann.masks is not None:
            shown = 0   # only draw/number confident fingers (matches the count)
            for mpts, c in zip(ann.masks.xy, ann.boxes.conf):
                if len(mpts) < 3 or float(c) < Config.CONF_THRESHOLD: continue
                color = COLORS[shown % len(COLORS)]
                shown += 1
                pts   = mpts.astype(np.int32).reshape((-1,1,2))
                cv2.fillPoly(overlay,[pts],color)
                cv2.polylines(best_frame,[pts],True,color,2)
                cx,cy = int(mpts[:,0].mean()),int(mpts[:,1].mean())
                cv2.circle(best_frame,(cx,cy),14,color,-1)
                cv2.circle(best_frame,(cx,cy),14,(255,255,255),2)
                num = str(shown)
                (tw,th),_ = cv2.getTextSize(num,cv2.FONT_HERSHEY_SIMPLEX,0.40,2)
                cv2.putText(best_frame,num,(cx-tw//2,cy+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,0,0),2)
        best_frame = cv2.addWeighted(overlay,Config.MASK_ALPHA,
                                     best_frame,1-Config.MASK_ALPHA,0)
        cv2.rectangle(best_frame,(0,0),(360,44),(10,10,10),-1)
        cv2.putText(best_frame,
            f"Fingers:{winner}  conf:{best_conf:.2f}  [{win_votes}/{total}fr]",
            (8,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,230,255),2)
        path = f"captures/img_{int(time.time()*1000)}.jpg"
        cv2.imwrite(path, best_frame)
        res["image_path"] = path
    return res


# ─────────────────────────────────────────────────────
# STARTUP THREAD — loads model/firebase/arduino off the main thread
# ─────────────────────────────────────────────────────
class StartupThread(QThread):
    status_signal = pyqtSignal(str)
    ready_signal  = pyqtSignal(object)  # dict: model/firebase/arduino → bool

    def run(self):
        results = {'model': False, 'firebase': False, 'arduino': False}

        self.status_signal.emit("Loading YOLO model…")
        try:
            loadModel()
            results['model'] = True
            self.status_signal.emit("YOLO model ready")
        except Exception as e:
            print(f"Model load failed: {e}")

        self.status_signal.emit("Connecting to Firebase…")
        try:
            testFirebaseConnection()
            results['firebase'] = firebase_connected

        except Exception as e:
            print(f"Firebase check failed: {e}")

        self.status_signal.emit("Starting Arduino…")
        try:
            startArduino()
            results['arduino'] = arduino is not None
        except Exception as e:
            print(f"Arduino init failed: {e}")

        self.ready_signal.emit(results)


# ─────────────────────────────────────────────────────
# WEIGHT POLLER THREAD
# ─────────────────────────────────────────────────────
class WeightPollerThread(QThread):
    weight_signal = pyqtSignal(float)   # -1.0 = no reading

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            w = -1.0
            try:
                r = requests.get(f"{Config.FIREBASE_URL}/Weight.json", timeout=2)
                if r.status_code == 200 and r.json() is not None:
                    w = float(r.json())
            except Exception:
                pass
            self.weight_signal.emit(w)
            self.msleep(300)

    def stop(self):
        self.running = False
        self.wait(2000)


# ─────────────────────────────────────────────────────
# VIDEO THREAD
# ─────────────────────────────────────────────────────
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    ready_signal = pyqtSignal()   # emitted after camera warm-up succeeds

    def __init__(self):
        super().__init__()
        self.running = True
        self._frame  = None
        self._lock   = threading.Lock()

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None: return True, self._frame.copy()
        return False, None

    @staticmethod
    def _zoom(frame, factor):
        if factor <= 1.0: return frame
        h, w = frame.shape[:2]
        ch, cw = int(h / factor), int(w / factor)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cam.set(cv2.CAP_PROP_BUFFERSIZE,   2)
        if not cam.isOpened():
            self.error_signal.emit("Cannot open camera.")
            return
        for _ in range(8):
            ret, frame = cam.read()
            if not ret:
                cam.release()
                self.error_signal.emit("Camera warm-up failed.")
                return
            if frame is not None:
                frame = self._zoom(frame, Config.ZOOM_FACTOR)
                with self._lock: self._frame = frame.copy()
                self.frame_signal.emit(frame)
        print("✓ Camera ready")
        self.ready_signal.emit()

        while self.running:
            t0 = time.time()
            ret, frame = cam.read()
            if ret and frame is not None:
                frame = self._zoom(frame, Config.ZOOM_FACTOR)
                with self._lock: self._frame = frame.copy()
                self.frame_signal.emit(frame)
            else:
                self.error_signal.emit("Camera read failed")
                self.running = False; break
            self.msleep(max(0,int((1/30-(time.time()-t0))*1000)))
        cam.release()
        print("VideoThread stopped")

    def stop(self): self.running = False; self.wait()


# ─────────────────────────────────────────────────────
# LIMIT SWITCH ONE-SHOT READER
# Used when pipeline is NOT running (no SerialReaderThread).
# Sends checkLimitSw:, reads up to 8 "Limit Switch" lines, emits result.
# ─────────────────────────────────────────────────────
class LimitSwCheckThread(QThread):
    result_signal = pyqtSignal(list)

    def __init__(self, serial_comm):
        super().__init__()
        self.serial_comm = serial_comm

    def run(self):
        lines = []
        start = time.time()
        while len(lines) < 8 and (time.time() - start) < 5.0:
            try:
                if self.serial_comm.in_waiting > 0:
                    raw  = self.serial_comm.readline()
                    line = raw.decode('utf-8', errors='replace').rstrip()
                    if line.startswith("Limit Switch"):
                        lines.append(line)
                else:
                    time.sleep(0.02)
            except Exception as e:
                print(f"  [LimitSwCheckThread] {e}")
                break
        if lines:
            self.result_signal.emit(lines)
        else:
            print("  [LimitSwCheckThread] no response from Arduino")


# ─────────────────────────────────────────────────────
# SERIAL READER THREAD
#
# Reads every line from Arduino and routes it:
#   "SCALE_STOP"      → sets scale_event (wakes PipelineThread)
#   "PLATE_IN_BIN:N"  → emits plate_sorted_signal for UI update
#   everything else   → printed for debug
#
# Keeps serial reading off the pipeline thread so classify work
# (Firebase polling, YOLO) never blocks incoming Arduino messages.
# ─────────────────────────────────────────────────────
class SerialReaderThread(QThread):
    plate_sorted_signal = pyqtSignal(int)    # bin number
    limit_sw_signal     = pyqtSignal(list)   # list of 8 raw "Limit Switch …" lines

    def __init__(self, serial_comm):
        super().__init__()
        self.serial_comm  = serial_comm
        self.running      = True
        self.scale_event  = threading.Event()  # set when SCALE_STOP arrives
        self._sw_lines    = []                 # accumulator for checkLimitSw: response

    def run(self):
        while self.running:
            try:
                if self.serial_comm.in_waiting > 0:
                    raw  = self.serial_comm.readline()
                    line = raw.decode('utf-8', errors='replace').rstrip()
                    if not line:
                        continue
                    if line.startswith("BIN_CLICK:"):
                        # BIN_CLICK:N,count:C,need:R  — pretty-print click progress
                        try:
                            parts   = line[len("BIN_CLICK:"):].split(",")
                            bin_n   = parts[0]
                            count   = parts[1].split(":")[1]
                            need    = parts[2].split(":")[1]
                            bar     = "#" * int(count) + "-" * (int(need) - int(count))
                            print(f"  [bin {bin_n} click] {count}/{need}  [{bar}]")
                        except Exception:
                            print(f"  [serial rx] '{line}'")
                        continue

                    if line.startswith("Limit Switch"):
                        self._sw_lines.append(line)
                        if len(self._sw_lines) >= 8:
                            self.limit_sw_signal.emit(self._sw_lines[:8])
                            self._sw_lines = []
                        continue

                    print(f"  [serial rx] '{line}'")

                    if line == "SCALE_STOP":
                        self.scale_event.set()

                    elif line.startswith("PLATE_IN_BIN:"):
                        try:
                            bin_num = int(line.split(":")[1])
                            self.plate_sorted_signal.emit(bin_num)
                        except ValueError:
                            pass

                    # ASSIGNED:N and WARN/debug lines just print (already logged)

            except Exception as e:
                print(f"  [serial reader err] {e}")
            self.msleep(20)
        print("SerialReaderThread stopped")

    def stop(self):
        self.running = False
        self.wait()


# ─────────────────────────────────────────────────────
# PIPELINE THREAD
#
# Circular conveyor flow — one iteration per plate:
#
#  ① wait SCALE_STOP   → Arduino stopped motor, plate at scale
#  ② settle + weigh    → Firebase (plate is stationary = fast convergence)
#  ③ YOLO              → camera shot while plate is still stopped
#  ④ classify          → finger count + weight → grade → bin N
#  ⑤ save DB
#  ⑥ send assign:N     → Arduino restarts motor immediately
#  ⑦ emit classified   → UI table row added
#  ⑧ loop to ①         (sort confirmation arrives async via sorted_signal)
#
# Motor is NOT stopped for sorting — the conveyor keeps running and
# the plate falls into its bin automatically when it passes the open
# servo gate.  "sorted_signal" fires when Arduino sends PLATE_IN_BIN:N.
# ─────────────────────────────────────────────────────
class PipelineThread(QThread):
    classified_signal = pyqtSignal(dict)
    sorted_signal     = pyqtSignal(dict)   # fired by _on_plate_sorted
    error_signal      = pyqtSignal(str)

    def __init__(self, ard, video_thread, farm: str, serial_reader):
        super().__init__()
        self.arduino       = ard
        self.video_thread  = video_thread
        self.farm          = farm
        self.serial_reader = serial_reader
        self.running       = True
        self._stop_flag    = threading.Event()
        self._paused       = False
        self._plate_num    = 0
        # FIFO list per bin — multiple plates can share the same bin.
        # The oldest (first) entry is always the plate currently in transit
        # for that bin (plates maintain their order on the conveyor).
        self._active_jobs: dict[int, list] = {}   # bin_num → [job, ...]
        self._jobs_lock    = threading.Lock()

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    def on_plate_sorted(self, bin_num: int):
        """Called from SerialReaderThread signal when PLATE_IN_BIN:N arrives."""
        with self._jobs_lock:
            pending = self._active_jobs.get(bin_num, [])
            if not pending:
                print(f"  [warn] PLATE_IN_BIN:{bin_num} but no job queued for that bin")
                return
            job = pending.pop(0)   # FIFO — oldest assignment first
            if not pending:
                del self._active_jobs[bin_num]
        print(f"  ✓ PLATE_IN_BIN:{bin_num}  plate#{job['plate']} sorted")
        self.sorted_signal.emit(job)

    def run(self):
        while self.running:
            if self._paused:
                self.msleep(200); continue
            try:
                self._process_one_plate()
            except Exception as e:
                import traceback; traceback.print_exc()
                self.error_signal.emit(f"Pipeline error: {e}")
                self.msleep(2000)
        print("PipelineThread stopped")

    def _process_one_plate(self):
        self._plate_num += 1
        p = self._plate_num
        print(f"\n{'═'*50}")
        print(f"  PLATE #{p}")
        print(f"{'═'*50}")

        # ── ① WAIT FOR SCALE_STOP ────────────────────────────────
        # Arduino fires this automatically when a plate arrives at the
        # weighing station and stops the motor.
        # scale_event was already cleared just before the previous sendAssign,
        # so any SCALE_STOP that arrived after the motor restarted is preserved.
        print(f"  [1] Waiting for plate at scale…")
        arrived = self.serial_reader.scale_event.wait(
            timeout=Config.SCALE_TIMEOUT_S)
        if not self.running:
            return
        if not arrived:
            self.error_signal.emit(f"Scale timeout plate#{p}")
            # Clear before restarting so the next plate's SCALE_STOP isn't missed
            self.serial_reader.scale_event.clear()
            return
        print(f"  [1] ✓ Plate at scale (motor stopped by Arduino)")

        # ── ② WEIGH (Firebase) ───────────────────────────────────
        # Plate is stationary — weight converges quickly.
        self._stop_flag.wait(Config.WEIGHT_SETTLE_S)
        if not self.running:
            return
        print(f"  [2] Weighing…")
        weight, wstatus = waitForStableWeight(self._stop_flag)
        if not self.running:
            return
        if wstatus != "ok":
            # "empty"  → plate weight stayed below the lightest grade: no /
            #            too-light banana — fast reject without the 20 s timeout.
            # "timeout"→ Firebase/scale never produced a stable reading.
            if wstatus == "empty":
                cls_label = "No Banana Detected"
                print(f"  [2] ✗ No banana on plate (sub-floor weight) — skipping plate#{p}")
            else:
                cls_label = "Weight Read Failed"
                print(f"  [2] ✗ Weight read failed — skipping plate#{p}")
            self.serial_reader.scale_event.clear()
            self.arduino.sendAssign(0)
            job = {"plate": p, "bin": 0, "cls": cls_label,
                   "weight": -1, "finger": "-", "size": "-",
                   "img": "", "farm": self.farm}
            self.classified_signal.emit(job)
            self.error_signal.emit(f"{cls_label} plate#{p}")
            return
        print(f"  [2] ✓ {weight:.1f}g")

        # ── ③ YOLO (camera) ──────────────────────────────────────
        # Camera is at the scale station, plate is stationary = clean shot.
        print(f"  [3] YOLO…")
        det    = captureImage(self.video_thread.get_latest_frame)
        finger = det["finger"][1]
        conf   = det["finger"][0]
        if finger == "invalid":
            # YOLO is the banana-presence check: distinguish a genuinely empty
            # plate ("No Banana Detected") from a banana it couldn't read
            # ("Detection Failed") or a dead camera ("Camera Read Failed").
            cls_label = {
                "no_banana":   "No Banana Detected",
                "detect_fail": "Detection Failed",
                "camera_fail": "Camera Read Failed",
            }.get(det.get("status", "no_banana"), "Detection Failed")
            print(f"  [3] ✗ {cls_label} — skipping plate#{p}")
            self.serial_reader.scale_event.clear()
            self.arduino.sendAssign(0)
            # "No banana" means no meaningful weight to report → show a dash.
            # "Detection Failed"/"Camera Read Failed" keep the weight (banana present).
            rpt_weight = -1 if cls_label == "No Banana Detected" else weight
            job = {"plate": p, "bin": 0, "cls": cls_label,
                   "weight": rpt_weight, "finger": "-", "size": "-",
                   "img": det.get("image_path", ""), "farm": self.farm}
            self.classified_signal.emit(job)
            self.error_signal.emit(f"{cls_label} plate#{p}")
            return
        print(f"  [3] ✓ {finger}  conf:{conf:.2f}")

        # ── ④ CLASSIFY ──────────────────────────────────────────
        finger_enum      = parseFinger(finger)
        banana_cls, hand = classifyBanana(finger_enum, weight)
        bin_num          = CLASS_TO_BIN.get(banana_cls)
        print(f"  [4] ✓ {banana_cls.value}  {hand.value}  → bin:{bin_num}")

        if banana_cls == BananaClass.UNKNOWN or bin_num is None:
            print(f"  [4] Invalid Classes — skipping plate#{p}")
            self.serial_reader.scale_event.clear()
            self.arduino.sendAssign(0)
            job = {"plate": p, "bin": 0, "cls": banana_cls.value,
                   "weight": weight, "finger": finger, "size": hand.value,
                   "img": det.get("image_path", ""), "farm": self.farm}
            self.classified_signal.emit(job)
            self.error_signal.emit(f"Invalid Classes plate#{p}")
            return

        # ── ⑤ SAVE TO DATABASE ──────────────────────────────────
        saveToDatabase(self.farm, banana_cls.value, weight, finger,
                       hand.value, conf,
                       det["x1"], det["y1"], det["x2"], det["y2"])

        # ── ⑥ BUILD JOB + SEND ASSIGN ───────────────────────────
        job = {"plate": p, "bin": bin_num, "cls": banana_cls.value,
               "weight": weight, "finger": finger, "size": hand.value,
               "img": det.get("image_path", ""), "farm": self.farm}

        with self._jobs_lock:
            if bin_num not in self._active_jobs:
                self._active_jobs[bin_num] = []
            self._active_jobs[bin_num].append(job)
            depth = len(self._active_jobs[bin_num])

        if depth > 3:
            print(f"  [warn] bin {bin_num} has {depth} unconfirmed plates"
                  f" — PLATE_IN_BIN may be lost")

        # Clear the event BEFORE restarting the motor so any SCALE_STOP
        # that arrives after this point belongs to the next plate and is kept.
        self.serial_reader.scale_event.clear()
        self._stop_flag.wait(2.0)
        if not self.running:
            return
        self.arduino.sendAssign(bin_num)   # Arduino restarts motor immediately
        print(f"  [5] ✓ assign:{bin_num} sent — motor restarted by Arduino")

        # ── ⑦ EMIT CLASSIFIED → UI row ──────────────────────────
        self.classified_signal.emit(job)

        time.sleep(Config.CLASSIFY_COOLDOWN_S)
        # ── ⑧ Loop back to ① — sort happens automatically ────────

    def stop(self):
        self.running = False
        self._stop_flag.set()
        if self.serial_reader:
            self.serial_reader.scale_event.set()  # unblock scale_event.wait()
        self.wait(6000)


# ─────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────
def showMsg(title, text):
    m = QMessageBox()
    m.setWindowTitle(title); m.setIcon(QMessageBox.Information)
    m.setText(text); m.exec()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("old/ui/resultUi.ui", self)
        self.setWindowTitle("Banana Sorter — Starting…")
        self.showMaximized()

        # Compact result table rows (header/cell fonts come from the .ui stylesheet)
        self.ui.tblResult.verticalHeader().setDefaultSectionSize(28)
        self.ui.tblResult.verticalHeader().setVisible(False)
        # Distribute every column evenly across the table width so each header
        # sits in its own equal share — no single column balloons to absorb the gap.
        from PyQt5.QtWidgets import QHeaderView
        _hdr = self.ui.tblResult.horizontalHeader()
        _hdr.setSectionResizeMode(QHeaderView.Stretch)

        self.pipeline_thread = None
        self.serial_reader   = None
        self._startup_ok     = False
        self._camera_ok      = False
        self._startup_results = {}

        self.ui.btnStart.clicked.connect(self.onStart)
        self.ui.btnStop.clicked.connect(self.onStop)
        self.ui.btnNext.clicked.connect(self.onNext)
        self.ui.btnTestServo.clicked.connect(self.onTestServo)
        self.ui.btnCheckSw.clicked.connect(self.onCheckLimitSw)
        self.ui.btnClearTable.clicked.connect(self.onClearTable)


        self.ui.btnStart.setEnabled(False)

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self._showFrame)
        self.video_thread.error_signal.connect(lambda m: showMsg("Video", m))
        self.video_thread.ready_signal.connect(self._onCameraReady)
        self.video_thread.start()

        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtCore import Qt
        self._weight_label = QLabel("Weight: -- g")
        self._weight_label.setAlignment(Qt.AlignCenter)
        self._weight_label.setStyleSheet(
            "background: rgba(0,0,0,160); color: #00FF88;"
            "font-size: 16px; font-weight: bold; padding: 4px; border-radius: 4px;")
        self.ui.frame.layout().addWidget(self._weight_label, 1, 0)

        self._weight_poller = WeightPollerThread()
        self._weight_poller.weight_signal.connect(self._onWeightUpdate)
        self._weight_poller.start()

        self._startup = StartupThread()
        self._startup.status_signal.connect(
            lambda msg: self.setWindowTitle(f"Banana Sorter — {msg}"))
        self._startup.ready_signal.connect(self._onStartupReady)
        self._startup.start()

    def _onCameraReady(self):
        self._camera_ok = True
        self._checkReady()

    def _onStartupReady(self, results: dict):
        self._startup_results = results
        errors = []
        if not results['model']:
            errors.append("YOLO model failed to load (check weights/segment1.pt)")
        if not results['arduino']:
            errors.append(f"Arduino not found on {Config.ARDUINO_PORT}")
        if errors:
            showMsg("Startup Failed",
                    "System cannot start — fix these issues and restart:\n\n• "
                    + "\n• ".join(errors))
            self.setWindowTitle("Banana Sorter — Startup failed")
            return   # btnStart stays disabled
        if not results['firebase']:
            showMsg("Warning", "Firebase not connected — weight sensing unavailable.")
        self._startup_ok = True
        self._checkReady()

    def _checkReady(self):
        if self._startup_ok and self._camera_ok:
            self.ui.btnStart.setEnabled(True)
            self.ui.btnCheckSw.setEnabled(True)
            self.setWindowTitle("Banana Sorter — Ready")
            print("App ready")

    def onClearTable(self):
        self.ui.tblResult.setRowCount(0)

    def onNext(self):
        if arduino:
            try:
                arduino.writeSerial("next:")
                print("Manual next: motor started")
            except Exception as e:
                showMsg("Error", f"Failed to send next: {e}")

    def onTestServo(self):
        if not arduino:
            showMsg("Test Servo", "Arduino not connected.")
            return
        n, ok = QInputDialog.getInt(self, "Test Servo", "Servo number (1–6):", 1, 1, 6)
        if not ok:
            return
        try:
            arduino.writeSerial(f"testServo:{n}")
            print(f"Test servo: testServo:{n} sent")
        except Exception as e:
            showMsg("Error", f"Failed to send testServo:{n}: {e}")

    def onCheckLimitSw(self):
        if not arduino:
            showMsg("Check Switches", "Arduino not connected.")
            return
        try:
            arduino.writeSerial("checkLimitSw:")
            print("checkLimitSw: sent")
        except Exception as e:
            showMsg("Error", f"Failed to send checkLimitSw: {e}")
            return

        if self.serial_reader and self.serial_reader.isRunning():
            # Pipeline running — SerialReaderThread catches the 8 response lines
            # and emits limit_sw_signal (connected in _startPipeline)
            pass
        else:
            # No pipeline — spin up one-shot reader that owns the port briefly
            t = LimitSwCheckThread(arduino.serialComm)
            t.result_signal.connect(self._onLimitSwData)
            t.start()

    def _onLimitSwData(self, lines: list):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel
        from PyQt5.QtCore import Qt

        SW_NAMES = [
            ("SW1", "Scale / camera"),
            ("SW2", "Spare"),
            ("SW3", "Bin 1"),
            ("SW4", "Bin 2"),
            ("SW5", "Bin 3"),
            ("SW6", "Bin 4"),
            ("SW7", "Bin 5"),
            ("SW8", "Bin 6"),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Limit Switch States")
        dlg.setMinimumWidth(320)
        lay = QVBoxLayout(dlg)
        grid = QGridLayout()

        header_style = "font-weight:bold; padding:4px;"
        grid.addWidget(QLabel("<b>Switch</b>", styleSheet=header_style), 0, 0)
        grid.addWidget(QLabel("<b>Location</b>", styleSheet=header_style), 0, 1)
        grid.addWidget(QLabel("<b>State</b>", styleSheet=header_style), 0, 2)

        for row, (line, (sw_id, location)) in enumerate(zip(lines, SW_NAMES), start=1):
            # line format: "Limit Switch N (label):  0" or ":  1"
            raw_val = line.strip().split(":")[-1].strip()
            pressed = raw_val == "0"   # INPUT_PULLUP: 0=pressed, 1=open
            state_text  = "PRESSED" if pressed else "OPEN"
            state_color = "#FF4444" if pressed else "#44CC44"

            lbl_sw  = QLabel(sw_id)
            lbl_loc = QLabel(location)
            lbl_st  = QLabel(state_text)
            lbl_st.setAlignment(Qt.AlignCenter)
            lbl_st.setStyleSheet(
                f"color:white; background:{state_color};"
                "font-weight:bold; padding:2px 8px; border-radius:3px;")

            grid.addWidget(lbl_sw,  row, 0)
            grid.addWidget(lbl_loc, row, 1)
            grid.addWidget(lbl_st,  row, 2)

        lay.addLayout(grid)
        dlg.exec_()

    def onStart(self):
        if not firebase_connected:
            if QMessageBox.question(self, "Firebase?",
                "Firebase not connected. Continue?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return

        farm = self.ui.cBoxFarm.currentText()

        self.serial_reader = SerialReaderThread(arduino.serialComm)
        self.serial_reader.start()

        self.serial_reader.scale_event.clear()
        try:
            arduino.writeSerial("next:")
            print("Motor started — waiting for first plate at scale…")
        except Exception as e:
            print(f"  [motor start failed] {e}")

        self._startPipeline(farm)

        self.ui.btnStart.setEnabled(False)
        self.ui.cBoxFarm.setEnabled(False)
        self.ui.btnStop.setEnabled(True)
        self.ui.btnNext.setEnabled(True)
        self.setWindowTitle("Banana Sorter — Running")

    def _startPipeline(self, farm: str):
        self.pipeline_thread = PipelineThread(
            arduino, self.video_thread, farm, self.serial_reader)
        self.pipeline_thread.classified_signal.connect(self._onClassified)
        self.pipeline_thread.sorted_signal.connect(self._onSorted)
        self.pipeline_thread.error_signal.connect(self._onError)
        self.serial_reader.plate_sorted_signal.connect(
            self.pipeline_thread.on_plate_sorted)
        self.serial_reader.limit_sw_signal.connect(self._onLimitSwData)
        self.pipeline_thread.start()
        self.setWindowTitle("Banana Sorter — Running")
        print("Pipeline started")

    def onStop(self):
        for t in (self.pipeline_thread, self.serial_reader):
            if t: t.stop()
        self.pipeline_thread = self.serial_reader = None
        if arduino:
            try:
                arduino.writeSerial("motorStop:")
            except Exception as e:
                print(f"  [motorStop send failed] {e}")
        self.ui.btnStart.setEnabled(True)
        self.ui.cBoxFarm.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.setWindowTitle("Banana Sorter — Stopped")
        print("Pipeline stopped")

    def closeEvent(self, event):
        self.onStop()
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        if self._weight_poller:
            self._weight_poller.stop()
            self._weight_poller = None
        event.accept()

    def _onWeightUpdate(self, w: float):
        if w < 0:
            self._weight_label.setText("Weight: -- g")
        else:
            self._weight_label.setText(f"Weight: {w:.1f} g")

    def _showFrame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.ui.lblImg.setPixmap(QPixmap.fromImage(qi))
        except Exception as e:
            print(f"Frame display error: {e}")

    def _onClassified(self, job):
        if job.get("img") and os.path.exists(job["img"]):
            qi = QImage(job["img"])
            if not qi.isNull():
                self.ui.lblImg.setPixmap(QPixmap.fromImage(qi))
        elif not job["bin"]:
            # Reject with no annotated capture — drop any stale still so the
            # image panel doesn't sit there contradicting the reject row.
            self.ui.lblImg.clear()
            self.ui.lblImg.setText("No Image")
        row = self.ui.tblResult.rowCount()
        self.ui.tblResult.insertRow(row)
        weight_str = f"{job['weight']:.1f}" if job["weight"] >= 0 else "-"
        vals = [job["cls"], weight_str,
                job["finger"], job["size"], str(job["bin"]) if job["bin"] else "-", job["farm"]]
        cols = min(len(vals), self.ui.tblResult.columnCount())
        for col in range(cols):
            self.ui.tblResult.setItem(row, col, QTableWidgetItem(vals[col]))
        if not job["bin"]:
            red   = QBrush(QColor(200, 60, 60))
            white = QBrush(QColor(255, 255, 255))
            for col in range(cols):
                item = self.ui.tblResult.item(row, col)
                if item:
                    item.setBackground(red)
                    item.setForeground(white)
        self.ui.tblResult.scrollToBottom()
        if job["bin"]:
            self.setWindowTitle(
                f"Sorter — plate#{job['plate']} → {job['cls']}"
                f" bin:{job['bin']}  sorting…")
        else:
            self.setWindowTitle(
                f"Sorter — plate#{job['plate']} → {job['cls']} — skipped")

    def _onSorted(self, job):
        print(f"UI: ✓ Sorted plate#{job['plate']} {job['cls']}"
              f" → bin:{job['bin']}")
        self.setWindowTitle(
            f"Sorter — plate#{job['plate']} sorted:"
            f" {job['cls']} → bin:{job['bin']}")

    def _onError(self, msg):
        print(f"[ERROR] {msg}")
        self.setWindowTitle(f"Sorter — ⚠ {msg[:70]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
