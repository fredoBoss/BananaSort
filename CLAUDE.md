# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Banana Sorting Machine — Capstone Project

Automated system that classifies and physically sorts banana hands into 6 grade bins using computer vision, weight sensing, and embedded hardware.

---

## Running the App

**Normal run (Arduino connected on COM5):**
```bash
python old/SortQue.py
```

**Smoke-test without hardware (mocks COM5 + MySQL):**
```bash
python test_run.py
```

**Bin travel / alignment tuner (hardware required):**
```bash
python bin_travel_test.py
```

**Arduino firmware:** Open `BananaSorting_dict/BananaSorting_dict.ino` in Arduino IDE, upload to Arduino Mega 2560 on COM5 at 115200 baud.

**ESP8266 firmware:** Open `esp8266_enhanced/esp8266_enhanced.ino`, set WiFi credentials, upload to NodeMCU. Serial commands: `cal1`–`cal5`, `forcetare`, `settare:<g>`, `info`, `diag`.

---

## Repository Layout

```
capstone_codes/
├── BananaSorting_dict/     # Arduino Mega firmware (main controller)
│   ├── BananaSorting_dict.ino  # Entry point — pin defs, globals, serial command dispatcher
│   ├── motorControl.ino        # motorRotateFunc() — relay-controlled conveyor motor
│   ├── servo.ino               # fireServo(), checkServoTimers() — drop gates (non-blocking)
│   ├── ultrasonic.ino          # initLimitSwitch(), testLimitSwitch(), checkWires()
│   └── weightSensor.ino        # Stub only — weight is handled by ESP8266/Firebase, not Mega
├── esp8266_enhanced/       # NodeMCU ESP8266 weight node → Firebase RTDB
│   └── esp8266_enhanced.ino
├── old/                    # Python host application
│   ├── SortQue.py          # Main app — PyQt5 UI, pipeline, YOLO, classification
│   ├── ardcommsTest.py     # Serial comms class (arduinoCommunication)
│   ├── calibration.py      # Standalone weight calibration UI
│   └── ui/                 # PyQt5 .ui files (resultUi.ui is the active one)
├── weights/                # YOLOv8 .pt model files — segment1.pt is the active model
├── bin_travel_test.py      # Hardware tuning tool — manual bin assign + live click counter
└── test_run.py             # Smoke-test launcher — mocks COM5 + MySQL for UI testing
```

---

## Hardware Overview

| Component | Role |
|-----------|------|
| Arduino Mega (USB → COM5) | Main controller — motor relay, 6 servos, 8 limit switches |
| ESP8266 NodeMCU | Weight node — HX711 → WiFi → Firebase RTDB every 300 ms |
| Conveyor motor (pin 33, active LOW) | Single motor shared by belt rotation and sort travel |
| 6 × Servo (Y1→5, Y2→4, Y3→6, Y4→8, Y5→9, Y6→7) | Drop gates — one per bin |
| limitSw1 (pin 23) | Scale / camera position trigger |
| limitSw3–8 (pins 35, 37, 39, 41, 43, 45) | Bin position triggers — one beside each servo |
| USB Camera (index 0) | YOLOv8 segmentation feed (640×640) |

**Motor note:** Pin 33 is active LOW — `digitalWrite(LOW)` = motor ON, `digitalWrite(HIGH)` = motor OFF.

---

## Serial Protocol

**Python → Arduino (commands ending with `\n`):**

| Command | Action |
|---------|--------|
| `next:` | Force-start motor (used once at pipeline startup only) |
| `assign:N` | N=1–6: activate job for bin N, restart motor. N=0: skip plate, restart motor. |
| `motorStop:` | Emergency stop — halts motor and clears all job slots |
| `testServo:N` | Directly fire servo N for bench testing |
| `setAlignDelay:<bin>,<ms>` | Adjust per-bin coast delay at runtime (no reflash needed) |
| `printClicks:` | Print active job click counts |
| `printAlignDelay:` | Print all 6 align delays |
| `checkLimitSw:` | Print all 8 limit switch states |
| `checkWires:` | Full hardware connectivity diagnostic |

**Arduino → Python (tokens):**

| Token | Meaning | Consumed by |
|-------|---------|-------------|
| `SCALE_STOP` | Plate arrived at scale, motor stopped | `SerialReaderThread` → sets `scale_event` → wakes `PipelineThread` |
| `PLATE_IN_BIN:N` | Plate confirmed in bin N (servo fired) | `SerialReaderThread` → `plate_sorted_signal` → `on_plate_sorted()` → UI |
| `ASSIGNED:N` / `ASSIGNED:skip` | After `assign:N` accepted | Console only |
| `BIN_CLICK:N,count:C,need:R` | Each bin switch edge for an active job | Pretty-printed progress bar to console |
| `MOTOR_RUNNING` / `MOTOR_STOPPED` | Motor state changes | Console only |

**Dead code warning:** The old `CAM_STOP`/`SORT_DONE` tokens and `trayPos:N` command are no longer used. `waitForCameraStop()`/`waitForSortDone()` in `ardcommsTest.py` are retained for backward compatibility but are not called by the current pipeline.

---

## Pipeline Flow (SortQue.py — PipelineThread)

The pipeline is **circular and concurrent** — multiple plates are on the conveyor simultaneously. The Arduino manages all plates autonomously in hardware; Python handles one plate at a time at the scale station.

```
ARDUINO (hardware loop)                    PYTHON (PipelineThread)
─────────────────────────────────────      ──────────────────────────────────────
[conveyor running]
  plate reaches limitSw1 (scale, pin 23)
  → wait SCALE_SETTLE_MS (700ms)
  → stop motor
  → allocate PlateJob slot (max 5)
  → Serial.println("SCALE_STOP")       →  SerialReaderThread.scale_event.set()
                                           PipelineThread wakes from event.wait()

                                       ①  sleep(WEIGHT_SETTLE_S = 1.0 s)
                                       ②  waitForStableWeight()
                                            poll Firebase /Weight.json every 300ms
                                            need WEIGHT_STABLE_N=4 readings within 5g
                                       ③  captureImage()
                                            flush FLUSH_FRAMES=5 stale frames
                                            capture CAPTURE_FRAMES=5 with YOLO
                                            vote on finger count (3/4/5)
                                            need ≥MIN_VALID_FRAMES=3 agreeing frames
                                       ④  classifyBanana(finger, weight) → bin N
                                       ⑤  saveToDatabase(...)  [MySQL INSERT]
                                       ⑥  scale_event.clear()
                                           arduino.sendAssign(bin_num)

  handleSerial("assign:N"):
    mark slot, restart motor             ←  motor restarted by Arduino
    Serial.println("ASSIGNED:N")

[conveyor running — plate moves toward bin N]
  each binPins[N-1] LOW edge:
    binClickCount++ → print BIN_CLICK
    if count == REQUIRED_CLICKS[N-1]:
      stop motor → alignDelay coast
      fireServo(N)  [open→close→neutral, non-blocking]
      schedule motor restart (+rotateDelay ms)
      Serial.println("PLATE_IN_BIN:N")  →  plate_sorted_signal
                                           on_plate_sorted() → sorted_signal → UI

[motor restarts → next plate cycle]    ⑦  classified_signal → UI table row
                                       ⑧  loop to ①
```

**On failure** (weight timeout, YOLO fail, unknown class): sends `assign:0` to skip the plate and restart the motor, then emits `error_signal`.

---

## Threading Model

```
Qt Main Thread (event loop)
├── VideoThread          — reads camera at 30 fps
│   └── frame_signal → MainWindow._showFrame()
│   └── get_latest_frame() — thread-safe snapshot under Lock (called by PipelineThread)
│
├── SerialReaderThread   — reads all Arduino serial output
│   ├── SCALE_STOP      → scale_event.set()       (wakes PipelineThread)
│   └── PLATE_IN_BIN:N  → plate_sorted_signal(N)  → PipelineThread.on_plate_sorted(N)
│
└── PipelineThread       — classify/sort loop (one iteration per plate)
    ├── classified_signal(job)  → MainWindow._onClassified()
    ├── sorted_signal(job)      → MainWindow._onSorted()
    └── error_signal(str)       → MainWindow._onError()
```

`_active_jobs` (dict[bin → list[job]]) is shared between `PipelineThread` (appends) and `on_plate_sorted` (pops, via Qt queued signal on main thread). Protected by `_jobs_lock`.

---

## Classification Logic

| Grade Class | Bin | Finger Count | Weight Range |
|-------------|-----|--------------|--------------|
| 33BCP       | 1   | 4–5 finger   | 400–520g     |
| 25BCP       | 2   | 4–5 finger   | 621–730g     |
| 30BCP       | 3   | 4–5 finger   | 521–620g     |
| IF38TR      | 4   | 3 finger     | 350–465g     |
| IF36TR      | 5   | 3 finger     | 466–540g     |
| 30TR        | 6   | 3 finger     | 541–650g     |

Any combination outside these ranges → `BananaClass.UNKNOWN` → `assign:0` (skip plate).

---

## Bin Positioning Logic (BananaSorting_dict.ino — `onBinSwitchFired`)

Arduino counts **LOW edges** on the assigned bin's own switch pin; other bins' switches are ignored for that job.

| Bin | REQUIRED_CLICKS | Switch pin | alignDelay |
|-----|:---------------:|:----------:|:----------:|
| 1   | 3               | 35         | 200 ms     |
| 2   | 4               | 37         | 0          |
| 3   | 5               | 39         | 0          |
| 4   | 5               | 41         | 0          |
| 5   | 6               | 43         | 0          |
| 6   | 5               | 45         | 0          |

`alignDelay` lets the tray coast after the final click before the servo opens. Tune at runtime with `setAlignDelay:<bin>,<ms>` (or use `bin_travel_test.py`). Changing `REQUIRED_CLICKS` requires a firmware recompile.

**Servo sequence after bin stop:** open (180°) → close (0°) after `rotateDelay` ms → neutral (95°) after `2×rotateDelay` ms. Fully non-blocking via `checkServoTimers()` called each `loop()` tick.

---

## Key Configuration (SortQue.py — Config class)

```python
ARDUINO_PORT         = "COM5"
ARDUINO_BAUD         = 115200
FIREBASE_URL         = "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app"
CONF_THRESHOLD       = 0.75      # YOLO min confidence
IOU_THRESHOLD        = 0.60      # YOLO NMS IoU threshold
CAPTURE_FRAMES       = 5         # frames captured per plate
MIN_VALID_FRAMES     = 3         # min frames that must agree on finger count
WEIGHT_THRESHOLD_G   = 5.0       # max spread (g) in rolling window for stable weight
WEIGHT_STABLE_N      = 4         # consecutive readings needed to declare stable
WEIGHT_TIMEOUT_S     = 20        # timeout for weight stabilisation
WEIGHT_SETTLE_S      = 1.0       # pause after SCALE_STOP before polling weight
SCALE_TIMEOUT_S      = 60        # max wait for SCALE_STOP before error
FIREBASE_RETRY       = 3         # retry attempts per Firebase read
CLASSIFY_COOLDOWN_S  = 0.1       # brief pause after emit before next plate loop
```

---

## Database

MySQL on localhost, database `grade`, table `finger_classes`:

```
Farm | Classes | weight | classes_name | size | conf | x1 | y1 | x2 | y2
```

Credentials: `root` / `Password1` (localhost only).

`mysql.connector.connect(...)` runs at module import time in `SortQue.py` — if MySQL is unavailable the app crashes before the window appears. Use `test_run.py` to bypass this.

---

## EEPROM Layout (ESP8266)

| Bytes  | Content                        |
|--------|--------------------------------|
| 0–3    | ZERO_OFFSET (float)            |
| 4–7    | SCALE_FACTOR (float)           |
| 8–27   | balancePt[0..4] (5 × float)    |
| 28–31  | BASE_TARE_WEIGHT (float)       |

`weightSensor.ino` on the Arduino Mega is a stub — the Mega has no HX711. All weight data flows through ESP8266 → Firebase → Python.

---

## Notes

- Arduino supports at most **5 plates in transit simultaneously** (`jobs[5]`). If all slots are full when a plate arrives, it logs a warning and restarts the motor — the plate passes through unsorted.
- `servo.ino` contains large commented-out blocks (old iterations); active code starts at line 265.
- `ultrasonic.ino` has the old version commented out at the top; active code starts at line 231.
- `ardcommsTest.py` legacy methods (`waitForCameraStop`, `waitForSortDone`, `trayPos:N` helpers) are dead code in the current pipeline — kept for backward compatibility only.
- `old/config.json` is a legacy file from an earlier architecture; all configuration now lives in the `Config` class in `SortQue.py`.
- `weights/` holds multiple training iterations (`count2.pt`, `count4.pt`, `Counts15.pt`, etc.); only `segment1.pt` is used.
- ESP8266 WiFi SSID/password and Firebase API key are hardcoded in `esp8266_enhanced.ino`.
