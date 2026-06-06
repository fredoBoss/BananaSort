"""
test_classifier_ui.py — Live camera + weight test UI for banana classification.
No Arduino required. Camera index 0, weight from Firebase.
"""

import sys, time, threading
import numpy as np
import cv2
import requests

from enum import Enum
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import torch


# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    CONF_THRESHOLD   = 0.75
    IOU_THRESHOLD    = 0.60
    MASK_ALPHA       = 0.50
    CAPTURE_FRAMES   = 5
    CAPTURE_INTERVAL_MS = 150
    MIN_VALID_FRAMES = 3
    FLUSH_FRAMES     = 5
    FLUSH_DELAY_MS   = 60
    ZOOM_FACTOR      = 1.3

    WEIGHT_THRESHOLD_G  = 5.0
    WEIGHT_STABLE_N     = 4
    MIN_VALID_WEIGHT_G  = 80.0
    MAX_VALID_WEIGHT_G  = 1500.0

    W_25BCP_MIN  = 621;  W_25BCP_MAX  = 730
    W_30BCP_MIN  = 521;  W_30BCP_MAX  = 620
    W_33BCP_MIN  = 400;  W_33BCP_MAX  = 520
    W_30TR_MIN   = 541;  W_30TR_MAX   = 650
    W_F36TR_MIN  = 466;  W_F36TR_MAX  = 540
    W_IF38TR_MIN = 350;  W_IF38TR_MAX = 465

    FIREBASE_URL     = "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app"
    FIREBASE_TIMEOUT = 2
    MODEL_PATH       = "weights/segment1.pt"


# ── Enums & classification ─────────────────────────────────────────────────────
class FingerCount(Enum):
    THREE   = "3-finger"
    FOUR    = "4-finger"
    FIVE    = "5-finger"
    UNKNOWN = "unknown"

class BananaClass(Enum):
    C25BCP  = "25BCP"
    C30BCP  = "30BCP"
    C33BCP  = "33BCP"
    C30TR   = "30TR"
    CF36TR  = "IF36TR"
    CIF38TR = "IF38TR"
    UNKNOWN = "Unknown"

CLASS_TO_BIN = {
    BananaClass.C33BCP:  1, BananaClass.C25BCP:  2,
    BananaClass.C30BCP:  3, BananaClass.CIF38TR: 4,
    BananaClass.CF36TR:  5, BananaClass.C30TR:   6,
}
BIN_COLOR = {
    1:"#e74c3c", 2:"#e67e22", 3:"#f1c40f",
    4:"#2ecc71", 5:"#3498db", 6:"#9b59b6",
}

def classifyBanana(finger: FingerCount, weight: float) -> BananaClass:
    c = Config
    if finger in (FingerCount.FOUR, FingerCount.FIVE):
        if c.W_25BCP_MIN <= weight <= c.W_25BCP_MAX: return BananaClass.C25BCP
        if c.W_30BCP_MIN <= weight <= c.W_30BCP_MAX: return BananaClass.C30BCP
        if c.W_33BCP_MIN <= weight <= c.W_33BCP_MAX: return BananaClass.C33BCP
    elif finger == FingerCount.THREE:
        if c.W_30TR_MIN   <= weight <= c.W_30TR_MAX:   return BananaClass.C30TR
        if c.W_F36TR_MIN  <= weight <= c.W_F36TR_MAX:  return BananaClass.CF36TR
        if c.W_IF38TR_MIN <= weight <= c.W_IF38TR_MAX: return BananaClass.CIF38TR
    return BananaClass.UNKNOWN


# ── Video thread ───────────────────────────────────────────────────────────────
COLORS = [(255,230,0),(255,80,80),(80,200,255),(180,0,255),(80,255,100),(255,160,0)]

class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self._frame  = None
        self._lock   = threading.Lock()

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    @staticmethod
    def _zoom(frame, factor):
        if factor <= 1.0: return frame
        h, w = frame.shape[:2]
        ch, cw = int(h/factor), int(w/factor)
        y0, x0 = (h-ch)//2, (w-cw)//2
        return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w,h), interpolation=cv2.INTER_LINEAR)

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cam.set(cv2.CAP_PROP_BUFFERSIZE,   2)
        if not cam.isOpened():
            self.error_signal.emit("Cannot open camera (index 0)")
            return
        while self.running:
            ret, frame = cam.read()
            if ret and frame is not None:
                frame = self._zoom(frame, Config.ZOOM_FACTOR)
                with self._lock: self._frame = frame.copy()
                self.frame_signal.emit(frame)
            self.msleep(33)
        cam.release()

    def stop(self): self.running = False; self.wait()


# ── Weight poller thread ───────────────────────────────────────────────────────
class WeightThread(QThread):
    weight_signal = pyqtSignal(float)  # -1 = no reading

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            w = -1.0
            try:
                r = requests.get(f"{Config.FIREBASE_URL}/Weight.json",
                                 timeout=Config.FIREBASE_TIMEOUT)
                if r.status_code == 200 and r.json() is not None:
                    w = float(r.json())
            except Exception:
                pass
            self.weight_signal.emit(w)
            self.msleep(300)

    def stop(self): self.running = False; self.wait(2000)


# ── Classify worker (runs off Qt main thread) ─────────────────────────────────
class ClassifyWorker(QThread):
    result_signal = pyqtSignal(dict)   # keys: finger, weight, banana_class, annotated_frame

    def __init__(self, model, get_frame_fn, weight):
        super().__init__()
        self._model        = model
        self._get_frame_fn = get_frame_fn
        self._weight       = weight

    def run(self):
        # flush stale frames
        for _ in range(Config.FLUSH_FRAMES):
            self._get_frame_fn()
            time.sleep(Config.FLUSH_DELAY_MS / 1000)

        vote_counts   = {}
        best_by_label = {}

        for i in range(Config.CAPTURE_FRAMES):
            ret, frame = self._get_frame_fn()
            if not ret or frame is None:
                time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
                continue
            result = self._model.predict(
                source=frame, conf=Config.CONF_THRESHOLD,
                iou=Config.IOU_THRESHOLD, save=False, verbose=False)[0]
            if result.masks is None or len(result.masks.xy) == 0:
                time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
                continue
            valid_masks  = [m for m in result.masks.xy if len(m) >= 3]
            banana_count = len(valid_masks)
            label = {3:"3-finger", 4:"4-finger", 5:"5-finger"}.get(banana_count)
            if label is None:
                time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)
                continue
            vote_counts[label] = vote_counts.get(label, 0) + 1
            confs = result.boxes.conf
            if len(confs):
                idx      = int(confs.argmax())
                conf_val = float(confs[idx])
                prev     = best_by_label.get(label)
                if prev is None or conf_val > prev[0]:
                    best_by_label[label] = (conf_val, frame.copy(), result)
            time.sleep(Config.CAPTURE_INTERVAL_MS / 1000)

        if not vote_counts:
            self.result_signal.emit({"error": "No detections"})
            return

        winner    = max(vote_counts, key=vote_counts.get)
        win_votes = vote_counts[winner]
        total     = sum(vote_counts.values())

        if win_votes < Config.MIN_VALID_FRAMES:
            self.result_signal.emit({"error": f"Weak consensus: {vote_counts}"})
            return

        best_conf, best_frame, best_result = best_by_label[winner]

        # annotate frame
        overlay = best_frame.copy()
        if best_result.masks is not None:
            for idx, (mpts, conf) in enumerate(
                    zip(best_result.masks.xy, best_result.boxes.conf)):
                color = COLORS[idx % len(COLORS)]
                pts   = mpts.astype(np.int32).reshape((-1,1,2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(best_frame, [pts], True, color, 2)
                cx, cy = int(mpts[:,0].mean()), int(mpts[:,1].mean())
                cv2.circle(best_frame, (cx,cy), 14, color, -1)
                cv2.circle(best_frame, (cx,cy), 14, (255,255,255), 2)
                num = str(idx+1)
                (tw,th),_ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 2)
                cv2.putText(best_frame, num, (cx-tw//2, cy+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 2)
        ann = cv2.addWeighted(overlay, Config.MASK_ALPHA,
                              best_frame, 1-Config.MASK_ALPHA, 0)
        cv2.rectangle(ann, (0,0), (380,44), (10,10,10), -1)
        cv2.putText(ann,
            f"Fingers:{winner}  conf:{best_conf:.2f}  [{win_votes}/{total}fr]",
            (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,230,255), 2)

        finger_enum = {
            "3-finger": FingerCount.THREE,
            "4-finger": FingerCount.FOUR,
            "5-finger": FingerCount.FIVE,
        }.get(winner, FingerCount.UNKNOWN)

        banana_class = classifyBanana(finger_enum, self._weight)

        self.result_signal.emit({
            "finger":       winner,
            "conf":         best_conf,
            "votes":        f"{win_votes}/{total}",
            "weight":       self._weight,
            "banana_class": banana_class,
            "annotated":    ann,
        })


# ── Main UI ───────────────────────────────────────────────────────────────────
def _frame_to_pixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg  = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                   rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)


class TestUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Banana Classifier — Live Test")
        self.setMinimumSize(860, 560)

        self._model          = None
        self._video_thread   = None
        self._weight_thread  = None
        self._classify_worker = None
        self._last_weight    = -1.0
        self._weight_history = []

        self._build()
        self._start_threads()
        self._load_model()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left: camera ──
        left = QVBoxLayout()
        left.setSpacing(6)

        cam_label = QLabel("Camera Feed")
        cam_label.setFont(QFont("Arial", 10, QFont.Bold))
        left.addWidget(cam_label)

        self._cam_view = QLabel()
        self._cam_view.setFixedSize(420, 420)
        self._cam_view.setStyleSheet("background:#111; border:1px solid #333;")
        self._cam_view.setAlignment(Qt.AlignCenter)
        self._cam_view.setText("Loading camera…")
        left.addWidget(self._cam_view)

        self._cam_status = QLabel("Camera: connecting…")
        self._cam_status.setFont(QFont("Arial", 9))
        self._cam_status.setStyleSheet("color:#888;")
        left.addWidget(self._cam_status)

        left.addStretch()
        root.addLayout(left)

        # ── Right: controls + results ──
        right = QVBoxLayout()
        right.setSpacing(10)

        # Weight live display
        wt_box = QGroupBox("Live Weight (Firebase)")
        wt_lay = QVBoxLayout(wt_box)
        self._weight_lbl = QLabel("— g")
        self._weight_lbl.setFont(QFont("Arial", 32, QFont.Bold))
        self._weight_lbl.setAlignment(Qt.AlignCenter)
        self._weight_lbl.setStyleSheet("color:#3498db;")
        self._weight_stable_lbl = QLabel("")
        self._weight_stable_lbl.setAlignment(Qt.AlignCenter)
        self._weight_stable_lbl.setFont(QFont("Arial", 9))
        self._weight_stable_lbl.setStyleSheet("color:#888;")
        wt_lay.addWidget(self._weight_lbl)
        wt_lay.addWidget(self._weight_stable_lbl)
        right.addWidget(wt_box)

        # Model status
        self._model_lbl = QLabel("Model: loading…")
        self._model_lbl.setFont(QFont("Arial", 9))
        self._model_lbl.setStyleSheet("color:#f39c12;")
        right.addWidget(self._model_lbl)

        # Next button
        self._btn = QPushButton("Next  →  Capture & Classify")
        self._btn.setFont(QFont("Arial", 13, QFont.Bold))
        self._btn.setMinimumHeight(50)
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self._on_next)
        right.addWidget(self._btn)

        # Result panel
        self._result_frame = QFrame()
        self._result_frame.setFrameShape(QFrame.StyledPanel)
        self._result_frame.setStyleSheet("background:#1e1e2e; border-radius:8px; padding:4px;")
        self._result_frame.setMinimumHeight(160)
        res_lay = QVBoxLayout(self._result_frame)

        self._lbl_class = QLabel("—")
        self._lbl_class.setAlignment(Qt.AlignCenter)
        self._lbl_class.setFont(QFont("Arial", 36, QFont.Bold))
        self._lbl_class.setStyleSheet("color:white;")

        self._lbl_bin = QLabel("")
        self._lbl_bin.setAlignment(Qt.AlignCenter)
        self._lbl_bin.setFont(QFont("Arial", 18))
        self._lbl_bin.setStyleSheet("color:#aaa;")

        self._lbl_detail = QLabel("")
        self._lbl_detail.setAlignment(Qt.AlignCenter)
        self._lbl_detail.setFont(QFont("Arial", 10))
        self._lbl_detail.setStyleSheet("color:#666;")

        res_lay.addWidget(self._lbl_class)
        res_lay.addWidget(self._lbl_bin)
        res_lay.addWidget(self._lbl_detail)
        right.addWidget(self._result_frame)

        # Reference table
        ref_box = QGroupBox("Ranges")
        ref_lay = QVBoxLayout(ref_box)
        ref_lay.setSpacing(1)
        for grade, bin_, fingers, wrange in [
            ("33BCP","Bin 1","4–5 finger","400–520 g"),
            ("25BCP","Bin 2","4–5 finger","621–730 g"),
            ("30BCP","Bin 3","4–5 finger","521–620 g"),
            ("IF38TR","Bin 4","3 finger","350–465 g"),
            ("IF36TR","Bin 5","3 finger","466–540 g"),
            ("30TR","Bin 6","3 finger","541–650 g"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(0)
            for txt, w in [(grade,65),(bin_,46),(fingers,86),(wrange,100)]:
                lbl = QLabel(txt)
                lbl.setFixedWidth(w)
                lbl.setFont(QFont("Arial", 9))
                row.addWidget(lbl)
            row.addStretch()
            ref_lay.addLayout(row)
        right.addWidget(ref_box)

        right.addStretch()
        root.addLayout(right)

    # ── Thread startup ────────────────────────────────────────────────────────
    def _start_threads(self):
        self._video_thread = VideoThread()
        self._video_thread.frame_signal.connect(self._on_frame)
        self._video_thread.error_signal.connect(lambda e: self._cam_status.setText(f"Camera error: {e}"))
        self._video_thread.start()

        self._weight_thread = WeightThread()
        self._weight_thread.weight_signal.connect(self._on_weight)
        self._weight_thread.start()

    def _load_model(self):
        class ModelLoader(QThread):
            done = pyqtSignal(object)
            def run(self):
                try:
                    m = YOLO(Config.MODEL_PATH)
                    dummy = np.zeros((640,640,3), dtype=np.uint8)
                    m.predict(source=dummy, conf=Config.CONF_THRESHOLD,
                              iou=Config.IOU_THRESHOLD, save=False, verbose=False)
                    self.done.emit(m)
                except Exception as e:
                    self.done.emit(str(e))

        self._loader = ModelLoader()
        self._loader.done.connect(self._on_model_loaded)
        self._loader.start()

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray):
        self._cam_view.setPixmap(_frame_to_pixmap(frame, 420, 420))
        self._cam_status.setText("Camera: live")

    def _on_weight(self, w: float):
        if w < 0:
            self._weight_lbl.setText("— g")
            self._weight_lbl.setStyleSheet("color:#e74c3c;")
            self._weight_stable_lbl.setText("No Firebase reading")
            self._last_weight = -1.0
            self._weight_history.clear()
            return

        self._last_weight = w
        self._weight_history.append(w)
        if len(self._weight_history) > Config.WEIGHT_STABLE_N:
            self._weight_history.pop(0)

        stable = (
            len(self._weight_history) == Config.WEIGHT_STABLE_N and
            (max(self._weight_history) - min(self._weight_history)) <= Config.WEIGHT_THRESHOLD_G
        )
        self._weight_lbl.setText(f"{w:.1f} g")
        self._weight_lbl.setStyleSheet("color:#2ecc71;" if stable else "color:#f39c12;")
        self._weight_stable_lbl.setText("Stable ✓" if stable else "Stabilising…")

    def _on_model_loaded(self, result):
        if isinstance(result, str):
            self._model_lbl.setText(f"Model error: {result}")
            self._model_lbl.setStyleSheet("color:#e74c3c;")
        else:
            self._model = result
            self._model_lbl.setText("Model: ready ✓")
            self._model_lbl.setStyleSheet("color:#2ecc71;")
            self._btn.setEnabled(True)

    def _on_next(self):
        if self._classify_worker and self._classify_worker.isRunning():
            return
        if self._model is None:
            return
        if self._last_weight < 0:
            self._lbl_class.setText("No weight")
            self._lbl_class.setStyleSheet("color:#e74c3c;")
            self._lbl_bin.setText("Check Firebase connection")
            self._lbl_detail.setText("")
            return

        self._btn.setEnabled(False)
        self._btn.setText("Capturing…")
        self._lbl_class.setText("…")
        self._lbl_bin.setText("")
        self._lbl_detail.setText("")

        self._classify_worker = ClassifyWorker(
            self._model,
            self._video_thread.get_latest_frame,
            self._last_weight,
        )
        self._classify_worker.result_signal.connect(self._on_classify_done)
        self._classify_worker.start()

    def _on_classify_done(self, data: dict):
        self._btn.setEnabled(True)
        self._btn.setText("Next  →  Capture & Classify")

        if "error" in data:
            self._lbl_class.setText("FAIL")
            self._lbl_class.setStyleSheet("color:#e74c3c;")
            self._lbl_bin.setText(data["error"])
            self._lbl_bin.setStyleSheet("color:#e74c3c;")
            self._lbl_detail.setText("")
            return

        # show annotated frame
        self._cam_view.setPixmap(_frame_to_pixmap(data["annotated"], 420, 420))

        bc      = data["banana_class"]
        bin_num = CLASS_TO_BIN.get(bc)
        color   = BIN_COLOR.get(bin_num, "#ffffff") if bin_num else "#e74c3c"

        self._lbl_class.setText(bc.value)
        self._lbl_class.setStyleSheet(f"color:{color};")

        if bin_num:
            self._lbl_bin.setText(f"→  Bin {bin_num}")
            self._lbl_bin.setStyleSheet(f"color:{color};")
        else:
            self._lbl_bin.setText("No bin — plate skipped")
            self._lbl_bin.setStyleSheet("color:#e74c3c;")

        self._lbl_detail.setText(
            f"{data['finger']}  |  {data['weight']:.1f} g  |  "
            f"conf {data['conf']:.2f}  |  votes {data['votes']}"
        )
        self._lbl_detail.setStyleSheet("color:#888;")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self._video_thread:  self._video_thread.stop()
        if self._weight_thread: self._weight_thread.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TestUI()
    win.show()
    sys.exit(app.exec_())
