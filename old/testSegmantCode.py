from ultralytics import YOLO
import cv2
import numpy as np
from ardcommsTest import arduinoCommunication
import time
import mysql.connector
import sys
from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5 import uic
import torch
import requests
import os
from enum import Enum
import threading

print("Application Starting")
os.makedirs("runs/segment", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")

# ─────────────────────────────────────────────
# CONFIGURATION — tweak these without touching logic
# ─────────────────────────────────────────────
class Config:
    # YOLO
    CONF_THRESHOLD      = 0.75      # lowered slightly for better recall
    IOU_THRESHOLD       = 0.60
    # MASK_ALPHA  = 0.50
    CAPTURE_FRAMES      = 5         # how many frames to sample for finger detection
    CAPTURE_INTERVAL_MS = 150       # ms between sampled frames
    MIN_VALID_FRAMES    = 3         # minimum frames that must agree on finger count    

    # Weight
    WEIGHT_THRESHOLD_G  = 5.0      # max variation for stable reading
    WEIGHT_STABLE_N     = 4        # readings that must agree
    WEIGHT_TIMEOUT_S    = 20       # extended timeout
    MIN_VALID_WEIGHT_G  = 80.0     # ignore readings below this (tray empty / drift)
    MAX_VALID_WEIGHT_G  = 1500.0   # ignore clearly erroneous readings
    WEIGHT_SETTLE_S     = 2.5      # seconds to wait after motor stops

    # Classification weight ranges (grams) — INCLUSIVE on both ends, no gaps
    # 4-finger / 5-finger
    W_25BCP_MIN  = 621
    W_25BCP_MAX  = 730
    W_30BCP_MIN  = 521
    W_30BCP_MAX  = 620
    W_33BCP_MIN  = 400
    W_33BCP_MAX  = 520
    # 3-finger
    W_30TR_MIN   = 541
    W_30TR_MAX   = 650      # extended upper bound (was 600, gaps above)
    W_F36TR_MIN  = 466
    W_F36TR_MAX  = 540
    W_IF38TR_MIN = 350
    W_IF38TR_MAX = 465

    # Firebase
    FIREBASE_URL        = "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app"
    FIREBASE_TIMEOUT_S  = 5
    FIREBASE_RETRY      = 3         # retries before giving up on one reading

    # Arduino
    ARDUINO_PORT        = "COM5"
    ARDUINO_BAUD        = 115200
    MOTOR_TIMEOUT_S     = 25
    SERVO_TIMEOUT_S     = 35

    # Cycle
    COOLDOWN_S          = 2.0


# ─────────────────────────────────────────────
# FIREBASE
# ─────────────────────────────────────────────
firebase_connected = False

def testFirebaseConnection():
    global firebase_connected
    try:
        r = requests.get(f"{Config.FIREBASE_URL}/.json", timeout=Config.FIREBASE_TIMEOUT_S)
        firebase_connected = r.status_code == 200
        print(f"{'✓' if firebase_connected else '✗'} Firebase: {'OK' if firebase_connected else f'HTTP {r.status_code}'}")
    except Exception as e:
        print(f"✗ Firebase: {e}")
        firebase_connected = False
    return firebase_connected

def getWeightFromFirebase() -> float:
    """Returns weight in grams or -1 on error."""
    global firebase_connected
    for attempt in range(Config.FIREBASE_RETRY):
        try:
            r = requests.get(
                f"{Config.FIREBASE_URL}/Weight.json",
                timeout=Config.FIREBASE_TIMEOUT_S
            )
            if r.status_code == 200 and r.json() is not None:
                w = float(r.json())
                if Config.MIN_VALID_WEIGHT_G <= w <= Config.MAX_VALID_WEIGHT_G:
                    return w
                elif w < Config.MIN_VALID_WEIGHT_G:
                    print(f"  Firebase weight too low ({w:.1f}g) — ignored")
                    return -1   # let caller handle low weight
                else:
                    print(f"  Firebase weight out of range ({w:.1f}g)")
                    return -1
            else:
                print(f"  Firebase empty/error (HTTP {r.status_code})")
        except requests.exceptions.Timeout:
            print(f"  Firebase timeout (attempt {attempt+1}/{Config.FIREBASE_RETRY})")
        except Exception as e:
            print(f"  Firebase error: {e}")
        if attempt < Config.FIREBASE_RETRY - 1:
            time.sleep(0.3)
    testFirebaseConnection()
    return -1

def wait_for_stable_weight(timeout_sec=None):
    """
    Polls Firebase until weight stabilises.
    Returns (weight_g, True) or (-1, False).
    """
    timeout_sec  = timeout_sec or Config.WEIGHT_TIMEOUT_S
    threshold    = Config.WEIGHT_THRESHOLD_G
    needed       = Config.WEIGHT_STABLE_N
    min_w        = Config.MIN_VALID_WEIGHT_G

    readings  = []
    zero_streak = 0
    start     = time.time()

    while (time.time() - start) < timeout_sec:
        w = getWeightFromFirebase()

        if w < 0:
            time.sleep(0.3)
            continue

        if w < min_w:
            zero_streak += 1
            print(f"  Low weight ({w:.1f}g) — streak {zero_streak}")
            if zero_streak >= 4:
                readings.clear()
                zero_streak = 0
            time.sleep(0.3)
            continue

        zero_streak = 0
        readings.append(w)
        if len(readings) > needed:
            readings.pop(0)

        if len(readings) == needed:
            variation = max(readings) - min(readings)
            if variation <= threshold:
                avg = sum(readings) / len(readings)
                print(f"✓ Weight stable: {avg:.1f}g  (variation {variation:.1f}g)")
                return avg, True
            else:
                print(f"  Variation {variation:.1f}g > {threshold}g — waiting…")

        time.sleep(0.3)

    print(f"✗ Weight timeout after {timeout_sec}s")
    return -1, False


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
db = mysql.connector.connect(
    host="localhost", user="root",
    password="Password1", database="grade"
)
print("Database: OK")

def saveToDatabase(farm, cls, weight, finger, size, conf, x1, y1, x2, y2):
    try:
        cur = db.cursor()
        cur.execute(
            """INSERT INTO finger_classes
               (Farm, Classes, weight, classes_name, size, conf, x1, y1, x2, y2)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (farm, cls, weight, finger, size, conf, x1, y1, x2, y2)
        )
        db.commit()
        cur.close()
        print(f"✓ DB saved: {cls}")
    except mysql.connector.Error as e:
        print(f"✗ DB error: {e}")


# ─────────────────────────────────────────────
# MODEL & CAMERA
# ─────────────────────────────────────────────
model: YOLO | None = None
arduino = None
cam: cv2.VideoCapture | None = None

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
    print("Initialising camera…")
    if cam and cam.isOpened():
        cam.release()
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   2)      # reduce latency
    if not cam.isOpened():
        print("✗ Cannot open camera")
        return False
    # warm-up
    for _ in range(8):
        ret, _ = cam.read()
        if not ret:
            print("✗ Camera warm-up failed")
            cam.release()
            return False
    print("✓ Camera ready")
    return True


# ─────────────────────────────────────────────
# MULTI-FRAME FINGER DETECTION
# ─────────────────────────────────────────────
def captureImage(get_frame_fn):
    """
    Captures Config.CAPTURE_FRAMES frames from get_frame_fn(),
    runs YOLO on each, and returns the majority-vote finger count.

    Returns finalRes dict (same schema as before).
    """
    finalRes = {
        "finger": [-1, "invalid"],
        "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "image_path": ""
    }

    vote_counts = {}   # {finger_label: count}
    best_conf   = 0.0
    best_box    = None
    best_frame  = None

    for i in range(Config.CAPTURE_FRAMES):
        ret, frame = get_frame_fn()
        if not ret or frame is None:
            print(f"  Frame {i+1}: no frame")
            time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
            continue

        results = model.predict(
            source=frame,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            save=False, verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            print(f"  Frame {i+1}: no detection")
            time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
            continue

        banana_count = len(results[0].boxes)
        label_map = {3: "3-finger", 4: "4-finger", 5: "5-finger"}
        label = label_map.get(banana_count)

        if label is None:
            print(f"  Frame {i+1}: unexpected count {banana_count}")
            time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
            continue

        vote_counts[label] = vote_counts.get(label, 0) + 1

        # Track best-confidence box across frames
        boxes    = results[0].boxes
        idx      = int(boxes.conf.argmax())
        conf_val = float(boxes.conf[idx])
        if conf_val > best_conf:
            best_conf  = conf_val
            best_box   = list(map(int, boxes.xyxy[idx]))
            best_frame = frame.copy()

        print(f"  Frame {i+1}: {label} (conf {conf_val:.2f})")
        time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)

    if not vote_counts:
        print("✗ No valid detections across all frames")
        return finalRes

    # Majority vote
    winner     = max(vote_counts, key=vote_counts.get)
    win_count  = vote_counts[winner]
    total      = sum(vote_counts.values())

    print(f"  Votes: {vote_counts}  → winner: {winner} ({win_count}/{total})")

    if win_count < Config.MIN_VALID_FRAMES:
        print(f"✗ Consensus too weak ({win_count} < {Config.MIN_VALID_FRAMES} required)")
        return finalRes

    finalRes["finger"]              = [best_conf, winner]
    finalRes["x1"], finalRes["y1"] = best_box[0], best_box[1]
    finalRes["x2"], finalRes["y2"] = best_box[2], best_box[3]

    # Annotate and save best frame
    if best_frame is not None:
        cv2.rectangle(best_frame,
            (best_box[0], best_box[1]), (best_box[2], best_box[3]),
            (0, 200, 0), 3)
        cv2.putText(best_frame,
            f"{winner} {best_conf:.2f}  [{win_count}/{total} frames]",
            (best_box[0], best_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        ts = int(time.time() * 1000)
        path = f"captures/img_{ts}.jpg"
        os.makedirs("captures", exist_ok=True)
        cv2.imwrite(path, best_frame)
        finalRes["image_path"] = path

    return finalRes


# ─────────────────────────────────────────────
# CLASSIFICATION LOGIC
# ─────────────────────────────────────────────
class FingerCount(Enum):
    THREE  = "3-finger"
    FOUR   = "4-finger"
    FIVE   = "5-finger"
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
    UNKNOWN = "unknown"

def parse_finger(label: str) -> FingerCount:
    label = str(label).strip().lower()
    if "3" in label: return FingerCount.THREE
    if "4" in label: return FingerCount.FOUR
    if "5" in label: return FingerCount.FIVE
    return FingerCount.UNKNOWN

def infer_hand(finger: FingerCount, weight: float) -> HandSize:
    """Infer hand size purely from weight ranges."""
    c = Config
    if finger == FingerCount.THREE:
        if c.W_IF38TR_MIN <= weight <= c.W_30TR_MAX:
            return HandSize.REGULAR
    elif finger in (FingerCount.FOUR, FingerCount.FIVE):
        if c.W_25BCP_MIN <= weight <= c.W_25BCP_MAX:   return HandSize.REGULAR
        if c.W_30BCP_MIN <= weight <= c.W_30BCP_MAX:   return HandSize.REGULAR
        if c.W_33BCP_MIN <= weight <= c.W_33BCP_MAX:   return HandSize.SMALL
    return HandSize.UNKNOWN

def classify(finger: FingerCount, weight: float):
    """
    Returns (BananaClass, HandSize).
    All weight ranges are closed intervals with NO gaps between them.
    """
    c = Config
    if finger == FingerCount.UNKNOWN:
        return BananaClass.UNKNOWN, HandSize.UNKNOWN

    hand = infer_hand(finger, weight)
    if hand == HandSize.UNKNOWN:
        print(f"  ⚠ No hand match — finger={finger.value}, weight={weight:.1f}g")
        return BananaClass.UNKNOWN, HandSize.UNKNOWN

    # 4 / 5 finger
    if finger in (FingerCount.FOUR, FingerCount.FIVE):
        if hand == HandSize.REGULAR:
            if c.W_25BCP_MIN <= weight <= c.W_25BCP_MAX:
                return BananaClass.C25BCP, hand
            if c.W_30BCP_MIN <= weight <= c.W_30BCP_MAX:
                return BananaClass.C30BCP, hand
        if hand == HandSize.SMALL:
            if c.W_33BCP_MIN <= weight <= c.W_33BCP_MAX:
                return BananaClass.C33BCP, hand

    # 3 finger
    if finger == FingerCount.THREE:
        if c.W_30TR_MIN  <= weight <= c.W_30TR_MAX:   return BananaClass.C30TR,   hand
        if c.W_F36TR_MIN <= weight <= c.W_F36TR_MAX:  return BananaClass.CF36TR,  hand
        if c.W_IF38TR_MIN<= weight <= c.W_IF38TR_MAX: return BananaClass.CIF38TR, hand

    print(f"  ⚠ No class matched — finger={finger.value}, hand={hand.value}, weight={weight:.1f}g")
    return BananaClass.UNKNOWN, hand

# Servo mapping — which trayPos command per class
SERVO_MAP = {
    BananaClass.C33BCP:  (lambda a: a.servoRotate1(), 1),
    BananaClass.C25BCP:  (lambda a: a.servoRotate2(), 2),
    BananaClass.C30BCP:  (lambda a: a.servoRotate3(), 3),
    BananaClass.CIF38TR: (lambda a: a.servoRotate4(), 4),
    BananaClass.CF36TR:  (lambda a: a.servoRotate5(), 5),
    BananaClass.C30TR:   (lambda a: a.servoRotate6(), 6),
}

def sortBananaClass(finger_label: str, weight: float, ard=None):
    """
    Returns (class_str, hand_str) and triggers the correct servo.
    """
    try:
        finger     = parse_finger(finger_label)
        weight_f   = float(weight)
        print(f"  Classifying: finger={finger.value}, weight={weight_f:.1f}g")

        banana_cls, hand = classify(finger, weight_f)
        print(f"  → {banana_cls.value} | hand={hand.value}")

        if ard and banana_cls in SERVO_MAP:
            fn, num = SERVO_MAP[banana_cls]
            fn(ard)
            print(f"  ✓ trayPos:{num} sent")
        elif banana_cls == BananaClass.UNKNOWN:
            print("  ✗ Unknown class — no servo action")

        return banana_cls.value, hand.value

    except Exception as e:
        print(f"  sortBananaClass error: {e}")
        return BananaClass.UNKNOWN.value, HandSize.UNKNOWN.value


# ─────────────────────────────────────────────
# VIDEO THREAD  (thread-safe frame buffer)
# ─────────────────────────────────────────────
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, cam):
        super().__init__()
        self.cam        = cam
        self.running    = True
        self._frame     = None
        self._lock      = threading.Lock()

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def run(self):
        while self.running:
            t0 = time.time()
            ret, frame = self.cam.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame.copy()
                self.frame_signal.emit(frame)
            else:
                self.error_signal.emit("Camera read failed")
                self.running = False
                break
            sleep = max(0, (1/30) - (time.time() - t0))
            self.msleep(int(sleep * 1000))
        print("VideoThread stopped")

    def stop(self):
        self.running = False
        self.wait()


# ─────────────────────────────────────────────
# ARDUINO THREAD  (classification loop)
# ─────────────────────────────────────────────
class ArduinoThread(QThread):
    result_signal = pyqtSignal(dict)
    error_signal  = pyqtSignal(str)

    def __init__(self, ard, mdl, farm, video_thread):
        super().__init__()
        self.arduino      = ard
        self.model        = mdl
        self.farm         = farm
        self.video_thread = video_thread
        self.running      = True
        self._trigger     = False
        self._busy        = False

    def trigger(self):
        if not self._busy and not self._trigger:
            self._trigger = True

    def run(self):
        while self.running:
            if not self._trigger:
                self.msleep(100)
                continue

            self._trigger = False
            self._busy    = True

            try:
                self._run_cycle()
            except Exception as e:
                import traceback
                print(f"✗ Cycle exception: {e}")
                traceback.print_exc()
                self.error_signal.emit(f"Cycle error: {e}")
                self.msleep(3000)

            self._busy    = False
            self._trigger = True   # queue next cycle automatically

        print("ArduinoThread stopped")

    def _run_cycle(self):
        print("\n" + "="*55)
        print("CLASSIFICATION CYCLE START")
        print("="*55)

        # ── STEP 1: advance conveyor to camera position ──────────────
        print("→ [1] Rotate to camera position…")
        self.arduino.reqRotateNext()
        if not self.arduino.waitForMotorStop(timeout=Config.MOTOR_TIMEOUT_S):
            self._fail("Motor timeout (camera position)")
            return
        print("✓ At camera position")

        # ── STEP 2: let scale settle ──────────────────────────────────
        print(f"→ [2] Scale settle ({Config.WEIGHT_SETTLE_S}s)…")
        time.sleep(Config.WEIGHT_SETTLE_S)

        # ── STEP 3: stable weight ─────────────────────────────────────
        print("→ [3] Waiting for stable weight…")
        weight, ok = wait_for_stable_weight()
        if not ok or weight <= 0:
            self._fail(f"Weight invalid: {weight:.1f}g")
            return
        print(f"✓ Weight: {weight:.1f}g")

        # ── STEP 4: multi-frame finger detection ──────────────────────
        print(f"→ [4] Finger detection ({Config.CAPTURE_FRAMES} frames)…")
        res    = captureImage(self.video_thread.get_latest_frame)
        finger = res["finger"][1]
        conf   = res["finger"][0]

        if finger == "invalid":
            self._fail("YOLO: no valid detection")
            return
        print(f"✓ Finger: {finger}  conf: {conf:.2f}")

        # ── STEP 5: classification + servo ────────────────────────────
        print("→ [5] Classifying…")
        cls, size = sortBananaClass(finger, weight, self.arduino)
        print(f"  Class={cls}  Size={size}")

        if cls == "unknown":
            self._fail("Classification: unknown class (out of weight range?)")
            return

        # ── STEP 6: wait for servo / tray to finish ───────────────────
        print("→ [6] Waiting for servo…")
        if not self.arduino.waitForServoStop(timeout=Config.SERVO_TIMEOUT_S):
            print("  ⚠ Servo timeout — saving anyway")

        # ── STEP 7: database ──────────────────────────────────────────
        print("→ [7] Saving to database…")
        saveToDatabase(
            self.farm, cls, weight, finger, size,
            conf, res["x1"], res["y1"], res["x2"], res["y2"]
        )

        # ── STEP 8: UI update ─────────────────────────────────────────
        self.result_signal.emit({
            "img":    res.get("image_path", ""),
            "class":  cls,
            "finger": finger,
            "size":   size,
            "weight": f"{weight:.1f}",
            "farm":   self.farm
        })

        print(f"✓ Cycle complete — cooldown {Config.COOLDOWN_S}s")
        time.sleep(Config.COOLDOWN_S)

    def _fail(self, reason: str):
        print(f"✗ {reason} — retrying next cycle in {Config.COOLDOWN_S}s")
        self.error_signal.emit(reason)
        time.sleep(Config.COOLDOWN_S)

    def stop(self):
        self.running = False
        self.wait()


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def msg(title, text):
    m = QMessageBox()
    m.setWindowTitle(title)
    m.setIcon(QMessageBox.Information)
    m.setText(text)
    m.exec()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("old/ui/resultUi.ui", self)
        self.setWindowTitle("Banana Sorter — Enhanced")
        self.showMaximized()
        self.video_thread   = None
        self.arduino_thread = None

        self.ui.btnStart.clicked.connect(self.onStart)
        self.ui.btnStop.clicked.connect(self.onStop)
        self.ui.btnTare.clicked.connect(self.onTare)

        testFirebaseConnection()
        loadModel()
        startArduino()
        time.sleep(2)

        if not firebase_connected:
            msg("Warning", "Firebase not connected — weight reading will fail.\nCheck WiFi on NodeMCU.")

        print("App ready")

    def onTare(self):
        msg("Tare", "Weight is read from Firebase (NodeMCU/HX711).\nSend 'tare:' via NodeMCU serial to tare the scale.")

    def onStart(self):
        global cam
        if not firebase_connected:
            if QMessageBox.question(self, "Firebase?",
                "Firebase not connected. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        if not initCamera():
            msg("Camera Error", "Cannot open camera.")
            return

        self.video_thread = VideoThread(cam)
        self.video_thread.frame_signal.connect(self._showFrame)
        self.video_thread.error_signal.connect(lambda m: msg("Video Error", m))

        self.arduino_thread = ArduinoThread(
            arduino, model,
            self.ui.cBoxFarm.currentText(),
            self.video_thread
        )
        self.arduino_thread.result_signal.connect(self._showResult)
        self.arduino_thread.error_signal.connect(self._onError)

        self.video_thread.start()
        self.arduino_thread.start()
        self.arduino_thread.trigger()   # kick off first cycle

        self.ui.btnStart.setEnabled(False)
        self.ui.cBoxFarm.setEnabled(False)
        self.ui.btnStop.setEnabled(True)
        self.setWindowTitle("Banana Sorter — Running")

    def onStop(self):
        global cam
        for t in (self.video_thread, self.arduino_thread):
            if t:
                t.stop()
        self.video_thread = self.arduino_thread = None
        if cam and cam.isOpened():
            cam.release()
            cam = None
        self.ui.btnStart.setEnabled(True)
        self.ui.cBoxFarm.setEnabled(True)
        self.ui.btnStop.setEnabled(False)
        self.setWindowTitle("Banana Sorter — Stopped")
        print("Stopped")

    def _showFrame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.ui.lblImg.setPixmap(QPixmap.fromImage(qi))
        except Exception as e:
            print(f"Frame display error: {e}")

    def _showResult(self, res):
        if res.get("img") and os.path.exists(res["img"]):
            qi = QImage(res["img"])
            if not qi.isNull():
                self.ui.lblImg.setPixmap(QPixmap.fromImage(qi))

        row = self.ui.tblResult.rowCount()
        self.ui.tblResult.insertRow(row)
        for col, key in enumerate(["class", "weight", "finger", "size", "farm"]):
            self.ui.tblResult.setItem(row, col, QTableWidgetItem(str(res[key])))
        self.ui.tblResult.scrollToBottom()
        self.setWindowTitle(f"Banana Sorter — Last: {res['class']}  {res['weight']}g")

    def _onError(self, msg_txt):
        print(f"Error: {msg_txt}")
        self.setWindowTitle(f"Banana Sorter — ⚠ {msg_txt[:60]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())