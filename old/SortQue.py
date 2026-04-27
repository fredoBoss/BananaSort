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

from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
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
    CONF_THRESHOLD      = 0.75
    IOU_THRESHOLD       = 0.60
    MASK_ALPHA          = 0.50
    CAPTURE_FRAMES      = 5
    CAPTURE_INTERVAL_MS = 150
    MIN_VALID_FRAMES    = 3
    FLUSH_FRAMES        = 5
    FLUSH_DELAY_MS      = 60

    WEIGHT_THRESHOLD_G  = 5.0
    WEIGHT_STABLE_N     = 4
    WEIGHT_TIMEOUT_S    = 20
    MIN_VALID_WEIGHT_G  = 80.0
    MAX_VALID_WEIGHT_G  = 1500.0
    WEIGHT_SETTLE_S     = 1.0   # plate is already stopped at scale — settles faster

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

def getWeightFromFirebase() -> float:
    for attempt in range(Config.FIREBASE_RETRY):
        try:
            r = requests.get(f"{Config.FIREBASE_URL}/Weight.json",
                            timeout=Config.FIREBASE_TIMEOUT_S)
            if r.status_code == 200 and r.json() is not None:
                w = float(r.json())
                if Config.MIN_VALID_WEIGHT_G <= w <= Config.MAX_VALID_WEIGHT_G:
                    return w
                print(f"  Weight out of range: {w:.1f}g")
                return -1
        except requests.exceptions.Timeout:
            print(f"  Firebase timeout ({attempt+1}/{Config.FIREBASE_RETRY})")
        except Exception as e:
            print(f"  Firebase error: {e}")
        if attempt < Config.FIREBASE_RETRY - 1:
            time.sleep(0.3)
    return -1

def waitForStableWeight() -> tuple:
    readings, zero_streak = [], 0
    start = time.time()
    while (time.time() - start) < Config.WEIGHT_TIMEOUT_S:
        w = getWeightFromFirebase()
        if w < 0:
            time.sleep(0.3); continue
        if w < Config.MIN_VALID_WEIGHT_G:
            zero_streak += 1
            if zero_streak >= 4:
                readings.clear(); zero_streak = 0
            time.sleep(0.3); continue
        zero_streak = 0
        readings.append(w)
        if len(readings) > Config.WEIGHT_STABLE_N:
            readings.pop(0)
        if len(readings) == Config.WEIGHT_STABLE_N:
            variation = max(readings) - min(readings)
            if variation <= Config.WEIGHT_THRESHOLD_G:
                avg = sum(readings) / len(readings)
                print(f"  ✓ Weight: {avg:.1f}g  (var {variation:.1f}g)")
                return avg, True
        time.sleep(0.3)
    print("  ✗ Weight timeout")
    return -1, False


# ─────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────
try:
    db = mysql.connector.connect(
        host="localhost", user="root",
        password="Password1", database="grade"
    )
    print("Database: OK")
except mysql.connector.Error as e:
    print(f"Database: connection failed — {e}")
    db = None

def saveToDatabase(farm, cls, weight, finger, size, conf, x1, y1, x2, y2):
    if db is None:
        print("  ✗ DB: not connected, skipping save")
        return
    try:
        cur = db.cursor()
        cur.execute(
            """INSERT INTO finger_classes
               (Farm, Classes, weight, classes_name, size, conf, x1, y1, x2, y2)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (farm, cls, weight, finger, size, conf, x1, y1, x2, y2))
        db.commit(); cur.close()
        print(f"  ✓ DB: {cls}")
    except mysql.connector.Error as e:
        print(f"  ✗ DB error: {e}")


# ─────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────
model:   YOLO | None                 = None
arduino: arduinoCommunication | None = None
cam:     cv2.VideoCapture | None     = None

def loadModel():
    global model
    model = YOLO("weights/segment1.pt")
    print("YOLO model loaded")

def startArduino():
    global arduino
    arduino = arduinoCommunication(Config.ARDUINO_PORT, Config.ARDUINO_BAUD)
    print("Arduino: OK")

def initCamera() -> bool:
    global cam
    if cam and cam.isOpened(): cam.release()
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   2)
    if not cam.isOpened(): return False
    for _ in range(8):
        ret, _ = cam.read()
        if not ret: cam.release(); return False
    print("✓ Camera ready")
    return True


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
    UNKNOWN = "Invalid Classes"

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
    res = {"finger":[-1,"invalid"],"x1":0,"y1":0,"x2":0,"y2":0,"image_path":""}
    for _ in range(Config.FLUSH_FRAMES):
        get_frame_fn(); time.sleep(Config.FLUSH_DELAY_MS/1000)

    vote_counts = {}
    best_conf, best_box, best_frame = 0.0, None, None

    for i in range(Config.CAPTURE_FRAMES):
        ret, frame = get_frame_fn()
        if not ret or frame is None:
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        result = model.predict(source=frame, conf=Config.CONF_THRESHOLD,
                               iou=Config.IOU_THRESHOLD, save=False, verbose=False)[0]
        if result.masks is None or len(result.masks.xy) == 0:
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        valid_masks  = [m for m in result.masks.xy if len(m) >= 3]
        banana_count = len(valid_masks)
        label        = {3:"3-finger",4:"4-finger",5:"5-finger"}.get(banana_count)
        if label is None:
            print(f"  Frame {i+1}: count={banana_count} outside range")
            time.sleep(Config.CAPTURE_INTERVAL_MS/1000); continue
        vote_counts[label] = vote_counts.get(label,0) + 1
        confs = result.boxes.conf
        if len(confs) > 0:
            idx = int(confs.argmax()); conf_val = float(confs[idx])
            if conf_val > best_conf:
                best_conf  = conf_val
                best_box   = list(map(int, result.boxes.xyxy[idx]))
                best_frame = frame.copy()
        print(f"  Frame {i+1}: {label}  conf={best_conf:.2f}")
        time.sleep(Config.CAPTURE_INTERVAL_MS/1000)

    if not vote_counts:
        print("  ✗ No detections"); return res
    winner    = max(vote_counts, key=vote_counts.get)
    win_votes = vote_counts[winner]
    total     = sum(vote_counts.values())
    print(f"  Votes: {vote_counts} → {winner} ({win_votes}/{total})")
    if win_votes < Config.MIN_VALID_FRAMES:
        print(f"  ✗ Weak consensus"); return res

    res["finger"] = [best_conf, winner]
    if best_box:
        res["x1"],res["y1"] = best_box[0],best_box[1]
        res["x2"],res["y2"] = best_box[2],best_box[3]

    if best_frame is not None and best_box:
        overlay = best_frame.copy()
        ann = model.predict(source=best_frame, conf=Config.CONF_THRESHOLD,
                            iou=Config.IOU_THRESHOLD, save=False, verbose=False)[0]
        if ann.masks is not None:
            for idx,(mpts,box,c) in enumerate(
                    zip(ann.masks.xy,ann.boxes.xyxy,ann.boxes.conf)):
                if len(mpts) < 3: continue
                color = COLORS[idx % len(COLORS)]
                pts   = mpts.astype(np.int32).reshape((-1,1,2))
                cv2.fillPoly(overlay,[pts],color)
                cv2.polylines(best_frame,[pts],True,color,2)
                cx,cy = int(mpts[:,0].mean()),int(mpts[:,1].mean())
                cv2.circle(best_frame,(cx,cy),14,color,-1)
                cv2.circle(best_frame,(cx,cy),14,(255,255,255),2)
                num = str(idx+1)
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
# VIDEO THREAD
# ─────────────────────────────────────────────────────
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, cam):
        super().__init__()
        self.cam = cam; self.running = True
        self._frame = None; self._lock = threading.Lock()

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None: return True, self._frame.copy()
        return False, None

    def run(self):
        while self.running:
            t0 = time.time()
            ret, frame = self.cam.read()
            if ret and frame is not None:
                with self._lock: self._frame = frame.copy()
                self.frame_signal.emit(frame)
            else:
                self.error_signal.emit("Camera read failed")
                self.running = False; break
            self.msleep(max(0,int((1/30-(time.time()-t0))*1000)))
        print("VideoThread stopped")

    def stop(self): self.running = False; self.wait()


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
    plate_sorted_signal = pyqtSignal(int)   # bin number

    def __init__(self, serial_comm):
        super().__init__()
        self.serial_comm  = serial_comm
        self.running      = True
        self.scale_event  = threading.Event()  # set when SCALE_STOP arrives

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
        if not arrived:
            self.error_signal.emit(f"Scale timeout plate#{p}")
            # Clear before restarting so the next plate's SCALE_STOP isn't missed
            self.serial_reader.scale_event.clear()
            return
        print(f"  [1] ✓ Plate at scale (motor stopped by Arduino)")

        # ── ② WEIGH (Firebase) ───────────────────────────────────
        # Plate is stationary — weight converges quickly.
        time.sleep(Config.WEIGHT_SETTLE_S)
        print(f"  [2] Weighing…")
        weight, ok = waitForStableWeight()
        if not ok or weight <= 0:
            # Weight failed — skip plate, restart motor
            print(f"  [2] ✗ Weight fail — skipping plate#{p}")
            self.serial_reader.scale_event.clear()
            self.arduino.sendAssign(0)
            self.error_signal.emit(f"Weight fail plate#{p}")
            return
        print(f"  [2] ✓ {weight:.1f}g")

        # ── ③ YOLO (camera) ──────────────────────────────────────
        # Camera is at the scale station, plate is stationary = clean shot.
        print(f"  [3] YOLO…")
        det    = captureImage(self.video_thread.get_latest_frame)
        finger = det["finger"][1]
        conf   = det["finger"][0]
        if finger == "invalid":
            print(f"  [3] ✗ YOLO fail — skipping plate#{p}")
            self.serial_reader.scale_event.clear()
            self.arduino.sendAssign(0)
            self.error_signal.emit(f"YOLO fail plate#{p}")
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
        self.arduino.sendAssign(bin_num)   # Arduino restarts motor immediately
        print(f"  [5] ✓ assign:{bin_num} sent — motor restarted by Arduino")

        # ── ⑦ EMIT CLASSIFIED → UI row ──────────────────────────
        self.classified_signal.emit(job)

        time.sleep(Config.CLASSIFY_COOLDOWN_S)
        # ── ⑧ Loop back to ① — sort happens automatically ────────

    def stop(self): self.running = False; self.wait()


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
        self.setWindowTitle("Banana Sorter — Pipeline")
        self.showMaximized()

        self.video_thread    = None
        self.pipeline_thread = None
        self.serial_reader   = None

        self.ui.btnStart.clicked.connect(self.onStart)
        self.ui.btnStop.clicked.connect(self.onStop)
        self.ui.btnTare.clicked.connect(self.onTare)

        testFirebaseConnection()
        loadModel()
        startArduino()
        time.sleep(2)
        if not firebase_connected:
            showMsg("Warning", "Firebase not connected.")
        print("App ready")

    def onTare(self):
        showMsg("Tare", "Send 'forcetare' via NodeMCU serial to re-tare.")

    def onStart(self):
        global cam
        if not firebase_connected:
            if QMessageBox.question(self, "Firebase?",
                "Firebase not connected. Continue?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        if not initCamera():
            showMsg("Camera Error", "Cannot open camera.")
            return

        farm = self.ui.cBoxFarm.currentText()

        # ── Serial reader (must start first — pipeline waits on its event) ──
        self.serial_reader = SerialReaderThread(arduino.serialComm)
        self.serial_reader.start()

        # ── Kick the motor so a plate advances to the scale ─────────────
        # The scale switch will stop the motor automatically; the pipeline
        # then wakes up on the resulting SCALE_STOP event.
        self.serial_reader.scale_event.clear()   # ensure clean state before first motor start
        try:
            arduino.writeSerial("next:")
            print("Motor started — waiting for first plate at scale…")
        except Exception as e:
            print(f"  [motor start failed] {e}")

        self.video_thread = VideoThread(cam)
        self.video_thread.frame_signal.connect(self._showFrame)
        self.video_thread.error_signal.connect(lambda m: showMsg("Video", m))

        self.pipeline_thread = PipelineThread(
            arduino, self.video_thread, farm, self.serial_reader)
        self.pipeline_thread.classified_signal.connect(self._onClassified)
        self.pipeline_thread.sorted_signal.connect(self._onSorted)
        self.pipeline_thread.error_signal.connect(self._onError)

        # Wire PLATE_IN_BIN events to pipeline so sorted_signal fires
        self.serial_reader.plate_sorted_signal.connect(
            self.pipeline_thread.on_plate_sorted)

        self.video_thread.start()
        self.pipeline_thread.start()

        self.ui.btnStart.setEnabled(False)
        self.ui.cBoxFarm.setEnabled(False)
        self.ui.btnStop.setEnabled(True)
        self.setWindowTitle("Banana Sorter — Running")
        print("Pipeline started")

    def onStop(self):
        global cam
        # Stop threads first so no more serial writes race with motorStop:
        for t in (self.pipeline_thread, self.video_thread, self.serial_reader):
            if t: t.stop()
        self.pipeline_thread = self.video_thread = self.serial_reader = None
        # Stop the motor and clear Arduino job state
        if arduino:
            try:
                arduino.writeSerial("motorStop:")
            except Exception as e:
                print(f"  [motorStop send failed] {e}")
        if cam and cam.isOpened(): cam.release(); cam = None
        self.ui.btnStart.setEnabled(True)
        self.ui.cBoxFarm.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.setWindowTitle("Banana Sorter — Stopped")
        print("Pipeline stopped")

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
        row = self.ui.tblResult.rowCount()
        self.ui.tblResult.insertRow(row)
        vals = [job["cls"], f"{job['weight']:.1f}",
                job["finger"], job["size"], str(job["bin"]) if job["bin"] else "-", job["farm"]]
        cols = min(len(vals), self.ui.tblResult.columnCount())
        for col in range(cols):
            self.ui.tblResult.setItem(row, col, QTableWidgetItem(vals[col]))
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
