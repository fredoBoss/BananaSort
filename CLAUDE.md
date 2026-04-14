# Banana Sorting Machine — Capstone Project

Automated system that classifies and physically sorts banana hands into 6 grade bins using computer vision, weight sensing, and embedded hardware.

---

## Repository Layout

```
capstone_codes/
├── BananaSorting_dict/     # Arduino Mega firmware (main controller)
│   ├── BananaSorting_dict.ino  # Entry point — pin defs, globals, serial command dispatcher
│   ├── motorControl.ino        # motorRotateFunc() — relay-controlled conveyor motor
│   ├── servo.ino               # rotateAndSort(), fireServo() — bin positioning + drop
│   ├── ultrasonic.ino          # rotateNextSwitchTrig() — tray-to-camera, wire diagnostics
│   └── weightSensor.ino        # HX711 calibration, tare, EEPROM, weightValAvg()
├── esp8266_enhanced/       # NodeMCU ESP8266 weight node → Firebase RTDB
│   └── esp8266_enhanced.ino
├── old/                    # Python host application
│   ├── SortQue.py          # Main app — PyQt5 UI, pipeline, YOLO, classification
│   ├── ardcommsTest.py     # Serial comms class (arduinoCommunication)
│   ├── calibration.py      # Standalone weight calibration UI
│   ├── config.json         # Legacy config (now superseded by Config class in SortQue.py)
│   └── ui/                 # PyQt5 .ui files (resultUi.ui is the active one)
├── weights/                # YOLOv8 .pt model files
│   └── segment1.pt         # Active model used by SortQue.py
└── test_run.py             # Smoke-test launcher — mocks COM5 + MySQL for UI testing
```

---

## Hardware Overview

| Component | Role |
|-----------|------|
| Arduino Mega (USB → COM5) | Main controller — motor relay, 6 servos, 8 limit switches, HX711 |
| ESP8266 NodeMCU | Separate weight node — HX711 → WiFi → Firebase RTDB |
| HX711 load cell (Arduino) | Weighs banana on tray (5 plates, calibrated per plate) |
| HX711 load cell (ESP8266) | Publishes live weight to Firebase every 300ms |
| Conveyor motor (pin 33, active LOW) | Belt motor — shared between belt rotation and sort travel |
| 6 × Servo (pins 4–9) | Drop gates — one per bin |
| limitSw1 (pin 23) | Camera position trigger |
| limitSw3–8 (pins 35–45) | Bin position triggers — one beside each servo |
| 2 × Ultrasonic (pins 11–13, 46) | Tray detection (legacy, still wired) |
| USB Camera (index 0) | YOLOv8 segmentation feed |

---

## Serial Protocol

**Python → Arduino (commands ending with `\n`):**

| Command | Action |
|---------|--------|
| `next:` | Rotate belt until tray reaches camera (limitSw1) |
| `trayPos:N` | Sort banana to bin N (1–6) |
| `readWt:` | Request weight reading |
| `tare:<1-5>` | Tare + set current plate |
| `calibrate:<1-5>` | Run full calibration for plate N |
| `checkLimitSw:` | Print all 8 limit switch states |
| `checkWires:` | Full hardware connectivity diagnostic |
| `printCal:` | Print EEPROM calibration values |
| `clearEEPROM:` | Wipe EEPROM calibration |
| `setRotateState:<0/1>` | 0 = disable belt rotation, 1 = enable |

**Arduino → Python (tokens):**

| Token | Meaning |
|-------|---------|
| `CAM_STOP` | Tray arrived at camera, belt motor stopped |
| `SORT_DONE` | Sort complete — servo fired, motor stopped |
| `readWt:<value>` | Weight reading response |

**These two tokens must never be swapped.** `waitForCameraStop()` listens only for `CAM_STOP`; `waitForSortDone()` listens only for `SORT_DONE`. Mixing them causes premature pipeline cycles.

---

## Pipeline Flow (SortQue.py — PipelineThread)

The belt motor and sort conveyor are the **same physical motor**. The pipeline is strictly sequential — never send `next:` while a sort is in progress.

```
① send "next:"       → belt rotates tray to camera position
② wait CAM_STOP      → tray confirmed at camera, motor free
③ wait stable weight → Firebase RTDB poll (ESP8266 node)
④ YOLO capture       → segment1.pt counts banana fingers (3/4/5)
⑤ classify           → finger count + weight range → grade class → bin number
⑥ save to MySQL      → INSERT into finger_classes table
⑦ send "trayPos:N"   → Arduino sorts: motor runs, counts limit-switch releases
⑧ wait SORT_DONE     → servo fired, motor stopped
⑨ update UI table    → loop back to ①
```

---

## Classification Logic

Banana grade is determined by **finger count** (YOLO) + **weight range** (Firebase):

| Grade Class | Bin | Finger Count | Weight Range |
|-------------|-----|--------------|--------------|
| 33BCP       | 1   | 4–5 finger   | 400–520g     |
| 25BCP       | 2   | 4–5 finger   | 621–730g     |
| 30BCP       | 3   | 4–5 finger   | 521–620g     |
| IF38TR      | 4   | 3 finger     | 350–465g     |
| IF36TR      | 5   | 3 finger     | 466–540g     |
| 30TR        | 6   | 3 finger     | 541–650g     |

---

## Bin Positioning Logic (servo.ino — rotateAndSort)

Bins are positioned along the conveyor. Arduino counts **limit-switch release edges** (LOW → HIGH) on the target bin's switch. Each bin requires a different count because they are physically farther down the belt:

| Bin | Required releases | Switch pin |
|-----|:-----------------:|:----------:|
| 1   | 2                 | 35         |
| 2   | 3                 | 37         |
| 3   | 3                 | 39         |
| 4   | 4                 | 41         |
| 5   | 4                 | 43         |
| 6   | 5                 | 45         |

`alignDelay[]` (ms) lets the tray coast slightly after the count to center under the servo before the motor stops. Tune per bin if the tray overshoots or stops short.

---

## Key Configuration (SortQue.py — Config class)

```python
ARDUINO_PORT    = "COM5"
ARDUINO_BAUD    = 115200
FIREBASE_URL    = "https://gradifier-aee7a-default-rtdb.asia-southeast1.firebasedatabase.app"
CONF_THRESHOLD  = 0.75      # YOLO min confidence
CAPTURE_FRAMES  = 5         # frames captured per plate
MIN_VALID_FRAMES = 3        # min frames that must agree on finger count
WEIGHT_THRESHOLD_G = 5.0   # max allowed variation for stable weight
WEIGHT_STABLE_N    = 4      # consecutive readings needed for stable weight
MOTOR_TIMEOUT_S    = 25     # wait for CAM_STOP
SORT_TIMEOUT_S     = 50     # wait for SORT_DONE (bin 6 = 5 clicks, takes longest)
```

---

## Database

MySQL on localhost, database `grade`, table `finger_classes`:

```
Farm | Classes | weight | classes_name | size | conf | x1 | y1 | x2 | y2
```

Credentials: `root` / `Password1` (localhost only).

---

## Running the App

**Normal run (Arduino connected on COM5):**
```bash
cd capstone_codes
python old/SortQue.py
```

**Smoke-test without hardware (mocks COM5 + MySQL):**
```bash
cd capstone_codes
python test_run.py
```

**Arduino firmware:** Open `BananaSorting_dict/BananaSorting_dict.ino` in Arduino IDE, upload to Arduino Mega 2560 on COM5 at 115200 baud.

**ESP8266 firmware:** Open `esp8266_enhanced/esp8266_enhanced.ino`, set WiFi credentials, upload to NodeMCU. Serial commands: `cal1-cal5`, `forcetare`, `settare:<g>`, `info`, `diag`.

---

## EEPROM Layout (Arduino Mega)

| Bytes  | Content                        |
|--------|--------------------------------|
| 0–3    | ZERO_OFFSET (float)            |
| 4–7    | SCALE_FACTOR (float)           |
| 8–27   | balancePt[0..4] (5 × float)    |

**EEPROM Layout (ESP8266)** — same first 28 bytes, plus:

| Bytes  | Content                        |
|--------|--------------------------------|
| 28–31  | BASE_TARE_WEIGHT (float)       |

---

## Notes

- `servo.ino` contains large commented-out blocks — these are old iterations kept for rollback reference. The active code starts at line 265.
- `ultrasonic.ino` similarly has the old version commented out at the top; active code starts at line 231.
- `weights/` holds multiple training iterations (`count2.pt`, `count4.pt`, `Counts15.pt`, etc.); only `segment1.pt` is used by the current app.
- `old/config.json` is a legacy file from an earlier architecture. All configuration now lives in the `Config` class inside `SortQue.py`.
