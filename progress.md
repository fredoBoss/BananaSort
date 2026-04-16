# Banana Sorting Machine — Progress & Technical Reference

**Project:** Capstone — Automated banana grading and physical sorting system  
**Last reviewed:** 2026-04-16

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Hardware Map](#2-hardware-map)
3. [Software Layer Overview](#3-software-layer-overview)
4. [End-to-End Pipeline Flow](#4-end-to-end-pipeline-flow)
5. [File-by-File Function Reference](#5-file-by-file-function-reference)
   - [SortQue.py (Python Host)](#sortquepy--python-host-application)
   - [ardcommsTest.py (Serial Driver)](#ardcommstestpy--arduino-serial-driver)
   - [BananaSorting_dict.ino (Arduino Main)](#bananasortingdictino--arduino-mega-main-firmware)
   - [motorControl.ino](#motorcontrolino)
   - [servo.ino](#servoino)
   - [ultrasonic.ino](#ultrasonicino)
   - [weightSensor.ino](#weightsensorino)
   - [esp8266_enhanced.ino (ESP8266 Node)](#esp8266enhancedino--esp8266-weight-node)
6. [Serial Protocol Reference](#6-serial-protocol-reference)
7. [Classification Logic](#7-classification-logic)
8. [Bin Positioning Logic](#8-bin-positioning-logic)
9. [Threading Model](#9-threading-model)
10. [Known Design Notes & Constraints](#10-known-design-notes--constraints)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HOST PC  (Windows)                             │
│                                                                         │
│   SortQue.py (PyQt5)                                                    │
│   ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │
│   │  MainWindow    │  │  VideoThread    │  │  SerialReaderThread  │   │
│   │  (PyQt5 UI)    │  │  (30 fps cam)   │  │  (serial rx)         │   │
│   └───────┬────────┘  └────────┬────────┘  └──────────┬───────────┘   │
│           │                    │                        │               │
│           └───────────┬────────┘                        │               │
│                       ▼                                  │               │
│              ┌─────────────────┐                         │               │
│              │  PipelineThread │◄────────────────────────┘               │
│              │  (classify/sort)│                                         │
│              └────────┬────────┘                                         │
│                       │  USB Serial (COM5, 115200)                       │
└───────────────────────┼─────────────────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │   Arduino Mega 2560         │
          │   BananaSorting_dict.ino    │
          │                             │
          │  Pin 33 → Motor relay       │
          │  Pins 4–9 → 6× Servo        │
          │  Pin 23 → Scale limit sw    │
          │  Pins 35–45 → Bin limit sws │
          │  Pins 11–13,46 → Ultrasonic │
          └─────────────────────────────┘

          ┌──────────────────────────────┐
          │   ESP8266 NodeMCU            │
          │   esp8266_enhanced.ino       │
          │                              │
          │   HX711 (D4/D5) → load cell  │
          │   WiFi → Firebase RTDB       │
          └──────────────────────────────┘
                        │
                        ▼ HTTP REST (every 300 ms)
          ┌──────────────────────────────┐
          │   Firebase RTDB              │
          │   /Weight  (adjusted grams)  │
          │   /WeightRaw                 │
          │   /Status  (ready/taring)    │
          └──────────────────────────────┘
                        ▲
                        │ HTTP GET poll
                   SortQue.py (Python)
```

**Data flows:**
- **Weight:** ESP8266 → Firebase RTDB → Python (HTTP poll)
- **Commands:** Python → Arduino (USB serial, text commands)
- **Events:** Arduino → Python (USB serial, token strings)
- **Video:** USB camera → VideoThread → PipelineThread (shared frame buffer)
- **Results:** PipelineThread → MySQL (INSERT) + PyQt5 UI (signals)

---

## 2. Hardware Map

| Component | Interface | Pin(s) | Role |
|-----------|-----------|--------|------|
| Conveyor motor relay | Digital OUT | 33 (active LOW) | Single motor drives belt + sort travel |
| Servo Y1 | PWM | 5 | Drop gate — Bin 1 |
| Servo Y2 | PWM | 4 | Drop gate — Bin 2 |
| Servo Y3 | PWM | 6 | Drop gate — Bin 3 |
| Servo Y4 | PWM | 8 | Drop gate — Bin 4 |
| Servo Y5 | PWM | 9 | Drop gate — Bin 5 |
| Servo Y6 | PWM | 7 | Drop gate — Bin 6 |
| limitSw1 | Digital IN (PULLUP) | 23 | Weighing/scale station arrival |
| limitSw2 | Digital IN (PULLUP) | 25 | Spare |
| limitSw3–8 | Digital IN (PULLUP) | 35,37,39,41,43,45 | Bin 1–6 position detection |
| Ultrasonic 1 | Trig/Echo | 12/11 | Tray detection (legacy, still wired) |
| Ultrasonic 2 | Trig/Echo | 13/46 | Tray detection (legacy, still wired) |
| USB Camera | USB | index 0 | YOLOv8 segmentation feed (640×640) |
| HX711 (ESP8266) | GPIO | D4/D5 | Load cell ADC for ESP8266 weight node |

**Motor note:** `motorCtrlPin` (pin 33) is **active LOW** — `digitalWrite(LOW)` turns the motor ON, `digitalWrite(HIGH)` turns it OFF. This is a relay-controlled circuit.

---

## 3. Software Layer Overview

```
┌─────────────────────────────────────────────────────────┐
│  PyQt5 UI  (MainWindow)                                 │
│  — Shows live camera feed, result table, start/stop     │
├─────────────────────────────────────────────────────────┤
│  PipelineThread  (core orchestrator)                    │
│  — Waits SCALE_STOP → weighs → YOLO → classify →       │
│     saves DB → sends assign:N → loops                   │
├────────────────┬────────────────────────────────────────┤
│  VideoThread   │  SerialReaderThread                    │
│  — 30 fps read │  — Reads all Arduino serial output     │
│  — frame_signal│  — Routes SCALE_STOP to threading.Event│
│  — get_latest_ │  — Routes PLATE_IN_BIN:N to signal     │
│    frame()     │                                        │
├────────────────┴────────────────────────────────────────┤
│  arduinoCommunication  (ardcommsTest.py)                │
│  — writeSerial / readSerial / sendAssign                │
│  — Legacy wait methods (waitForCameraStop/SortDone)     │
├─────────────────────────────────────────────────────────┤
│  YOLO / Firebase / MySQL  (module-level helpers)        │
│  — captureImage(), waitForStableWeight()                │
│  — saveToDatabase(), classifyBanana()                   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. End-to-End Pipeline Flow

The pipeline is **circular** — multiple plates are in transit simultaneously on the conveyor. The Arduino manages all plates in hardware; Python processes one plate at a time at the scale station.

```
ARDUINO (hardware loop)                    PYTHON (PipelineThread)
─────────────────────────────────────      ──────────────────────────────────────
[conveyor running]
  plate arrives at limitSw1 (pin 23)
  → onScaleArrival():
      wait SCALE_SETTLE_MS (700ms)
      stop motor
      allocate job slot (waitingAssign=true)
      Serial.println("SCALE_STOP")      →  SerialReaderThread sets scale_event
                                            PipelineThread wakes from event.wait()

                                        ①  time.sleep(WEIGHT_SETTLE_S = 1.0s)

                                        ②  waitForStableWeight()
                                             polls Firebase /Weight.json every 300ms
                                             needs 4 consecutive readings within 5g
                                             → returns (avg_grams, True)

                                        ③  captureImage()
                                             flush 5 stale camera frames
                                             capture 5 frames with YOLO predict()
                                             vote on finger count (3/4/5)
                                             need ≥3/5 agreeing frames
                                             → returns {finger, conf, box, image_path}

                                        ④  classifyBanana(finger_enum, weight)
                                             maps (finger_count, weight_range) → BananaClass
                                             maps BananaClass → bin_num (1–6)

                                        ⑤  saveToDatabase(...)
                                             INSERT into grade.finger_classes

                                        ⑥  arduino.sendAssign(bin_num)
                                             writes "assign:N\n" to serial

  handleSerial("assign:N"):
      mark slot active, assignedBin=N
      motorRotateFunc(0) → restart motor  ←
      Serial.println("ASSIGNED:N")

[conveyor running — plate moves toward bin N]
  every time binPins[N-1] triggers LOW:
    onBinSwitchFired(N):
      jobs[slot].binClickCount++
      print BIN_CLICK debug
      if count == REQUIRED_CLICKS[N-1]:
        stop motor
        schedule servoFireAt (alignDelay) or
          call fireServo(N) immediately
        schedule motorRestartAt
        Serial.println("PLATE_IN_BIN:N")  →  SerialReaderThread.plate_sorted_signal(N)
                                              pipeline_thread.on_plate_sorted(N)
                                              → sorted_signal.emit(job)
                                              → UI title updated

[motor restarts after rotateDelay ms]       ⑦  classified_signal already emitted at ⑥
                                                 UI table row added

                                        ⑧  loop back to ①  (wait for next SCALE_STOP)
```

**Concurrent plates:** Up to 5 `PlateJob` structs live in the Arduino simultaneously. Each newly classified plate gets its own slot and bin assignment. The motor runs continuously except at the scale stop and at each bin drop.

---

## 5. File-by-File Function Reference

---

### SortQue.py — Python Host Application

#### `Config` (class)
All runtime constants in one place — no config file needed.

| Constant | Value | Purpose |
|----------|-------|---------|
| `CONF_THRESHOLD` | 0.75 | Minimum YOLO confidence to accept a detection |
| `IOU_THRESHOLD` | 0.60 | YOLO NMS IoU threshold |
| `MASK_ALPHA` | 0.50 | Opacity of mask overlay on saved annotation image |
| `CAPTURE_FRAMES` | 5 | Number of frames captured per plate for voting |
| `CAPTURE_INTERVAL_MS` | 150 | Delay between capture frames (ms) |
| `MIN_VALID_FRAMES` | 3 | Minimum agreeing frames needed to accept finger count |
| `FLUSH_FRAMES` | 5 | Stale frames to discard before capture starts |
| `FLUSH_DELAY_MS` | 60 | Delay per flush frame |
| `WEIGHT_THRESHOLD_G` | 5.0 | Max spread (g) in rolling window for stable weight |
| `WEIGHT_STABLE_N` | 4 | Consecutive readings needed to declare stable |
| `WEIGHT_TIMEOUT_S` | 20 | Total timeout for weight stabilisation |
| `MIN_VALID_WEIGHT_G` | 80.0 | Reject weight readings below this |
| `MAX_VALID_WEIGHT_G` | 1500.0 | Reject weight readings above this |
| `WEIGHT_SETTLE_S` | 1.0 | Wait after SCALE_STOP before polling weight |
| `FIREBASE_URL` | gradifier-aee7a... | Firebase RTDB base URL |
| `FIREBASE_TIMEOUT_S` | 5 | HTTP request timeout |
| `FIREBASE_RETRY` | 3 | Retry attempts on Firebase failure |
| `ARDUINO_PORT` | COM5 | Serial port of Arduino Mega |
| `ARDUINO_BAUD` | 115200 | Serial baud rate |
| `SCALE_TIMEOUT_S` | 60 | Max time to wait for SCALE_STOP before error |
| `CLASSIFY_COOLDOWN_S` | 0.1 | Brief pause after emit before looping to ① |
| Weight ranges | various | Per-class g boundaries (see Classification section) |

---

#### `testFirebaseConnection() → bool`
Sends a GET to `FIREBASE_URL/.json`. Sets global `firebase_connected`. Called once at startup. Returns True if HTTP 200.

#### `getWeightFromFirebase() → float`
Single Firebase read of `/Weight.json`. Retries up to `FIREBASE_RETRY` times with 300 ms sleep between attempts. Returns the float value if it falls within `[MIN_VALID_WEIGHT_G, MAX_VALID_WEIGHT_G]`, otherwise `-1`.

#### `waitForStableWeight() → tuple[float, bool]`
Polls `getWeightFromFirebase()` in a loop until `WEIGHT_STABLE_N` consecutive readings are within `WEIGHT_THRESHOLD_G` of each other. Handles "zero streaks" (readings below minimum) by resetting the window. Times out after `WEIGHT_TIMEOUT_S`. Returns `(average_grams, True)` on success or `(-1, False)` on timeout/failure.

#### `saveToDatabase(farm, cls, weight, finger, size, conf, x1, y1, x2, y2)`
Executes `INSERT INTO grade.finger_classes (...)` via mysql.connector. Commits immediately. Prints success or error. Called once per successfully classified plate.

#### `loadModel()`
Loads `weights/segment1.pt` via `YOLO(...)` into the global `model`. Uses CUDA if available, CPU otherwise.

#### `startArduino()`
Instantiates `arduinoCommunication(COM5, 115200)` into global `arduino`. Raises on failure.

#### `initCamera() → bool`
Opens `cv2.VideoCapture(0)` at 640×640. Flushes 8 frames to warm the buffer. Returns True if camera opened and frames read successfully, False otherwise. Releases any previously open camera first.

---

#### `parseFinger(label: str) → FingerCount`
String → enum mapping. Looks for "3", "4", or "5" in the label string and returns the corresponding `FingerCount` enum value. Returns `UNKNOWN` if none match.

#### `inferHand(finger: FingerCount, weight: float) → HandSize`
Determines `HandSize` (REGULAR or SMALL) from finger count and weight:
- 3-finger in 350–650g range → REGULAR
- 4/5-finger in 621–730g → REGULAR, in 521–620g → REGULAR, in 400–520g → SMALL
- Anything outside those ranges → UNKNOWN

#### `classifyBanana(finger: FingerCount, weight: float) → tuple[BananaClass, HandSize]`
Combines finger count and weight range into a final grade class:

| Condition | Class |
|-----------|-------|
| 4/5-finger, 621–730g | 25BCP |
| 4/5-finger, 521–620g | 30BCP |
| 4/5-finger, 400–520g | 33BCP |
| 3-finger, 541–650g | 30TR |
| 3-finger, 466–540g | IF36TR |
| 3-finger, 350–465g | IF38TR |
| Any UNKNOWN | UNKNOWN |

Returns `(BananaClass.UNKNOWN, HandSize.UNKNOWN)` if finger is unknown or weight is out of all ranges.

#### `CLASS_TO_BIN` (dict)
Maps each `BananaClass` enum to a physical bin number 1–6:
`33BCP→1, 25BCP→2, 30BCP→3, IF38TR→4, IF36TR→5, 30TR→6`

---

#### `captureImage(get_frame_fn) → dict`
Core YOLO capture and voting function.

1. **Flush phase:** calls `get_frame_fn()` `FLUSH_FRAMES` times to discard stale buffered frames.
2. **Capture phase:** calls `get_frame_fn()` `CAPTURE_FRAMES` times. For each frame, runs `model.predict(...)` with the configured thresholds.
3. **Counting:** counts valid segmentation masks (≥3 polygon points). Only accepts counts of 3, 4, or 5. Maps count to label string.
4. **Voting:** accumulates `{label: count}`. Selects the majority winner.
5. **Consensus check:** rejects if winning label has fewer than `MIN_VALID_FRAMES` votes.
6. **Annotation:** runs a final YOLO pass on the best frame. Draws filled polygon masks (with `MASK_ALPHA` blend), outlines, numbered circles per banana finger. Draws a HUD bar with finger count, confidence, and vote score. Saves to `captures/img_<timestamp>.jpg`.

Returns dict: `{finger: [conf, label], x1, y1, x2, y2, image_path}`. On failure returns `finger: [-1, "invalid"]`.

---

#### `VideoThread` (QThread)
Continuously reads frames from the camera at ~30 fps in a background thread.

- **`run()`:** Loop calling `cam.read()`. Stores each frame under `_lock`. Emits `frame_signal` for live UI display. Emits `error_signal` and stops on camera failure.
- **`get_latest_frame() → (bool, ndarray)`:** Thread-safe snapshot of the most recent frame. Called by `captureImage()` inside `PipelineThread`.
- **`stop()`:** Sets `running = False` and waits for thread to join.

---

#### `SerialReaderThread` (QThread)
Reads all lines from the Arduino serial port in a dedicated background thread, preventing any serial data from being lost while the pipeline is busy with Firebase or YOLO.

- **`run()`:** Polls `serial_comm.in_waiting` every 20 ms. Decodes each line and routes it:
  - `BIN_CLICK:N,count:C,need:R` → pretty-prints a progress bar to console (not forwarded to pipeline).
  - `SCALE_STOP` → sets `scale_event` (`threading.Event`), which wakes the `PipelineThread`.
  - `PLATE_IN_BIN:N` → emits `plate_sorted_signal(bin_num)` for the pipeline's `on_plate_sorted` handler.
  - All other lines → printed to console as debug.
- **`stop()`:** Sets `running = False` and waits for thread to join.
- **`scale_event`** (`threading.Event`): The inter-thread signal between `SerialReaderThread` and `PipelineThread`. `PipelineThread` blocks on `scale_event.wait(timeout=...)` at the start of each plate cycle.

---

#### `PipelineThread` (QThread)
The heart of the Python application. Runs one iteration per plate, strictly sequential.

**Signals emitted:**
- `classified_signal(dict)` — fired after classification + DB save + assign sent; triggers UI table row
- `sorted_signal(dict)` — fired when Arduino confirms plate dropped into bin (`PLATE_IN_BIN:N`)
- `error_signal(str)` — fired on timeout or unclassifiable plate

**`__init__`**: stores arduino, video_thread, farm, serial_reader references. Initialises `_active_jobs: dict[int, list]` — a per-bin FIFO of in-transit job dicts. Multiple plates headed to the same bin queue in order.

**`pause()` / `resume()`**: Pauses the run loop (checks `_paused` flag every 200 ms).

**`on_plate_sorted(bin_num: int)`**: Called (via Qt signal) when `SerialReaderThread` receives `PLATE_IN_BIN:N`. Pops the oldest job from `_active_jobs[bin_num]`, emits `sorted_signal` with that job's data.

**`run()`**: While loop calling `_process_one_plate()` continuously. Catches exceptions, emits `error_signal`, sleeps 2 s before retrying.

**`_process_one_plate()`**: One full plate cycle:
1. Clear `scale_event`, block on `scale_event.wait(SCALE_TIMEOUT_S)` — wakes when Arduino sends `SCALE_STOP`
2. `time.sleep(WEIGHT_SETTLE_S)`, then `waitForStableWeight()`
3. `captureImage(video_thread.get_latest_frame)`
4. `parseFinger(finger_label)` → `classifyBanana(finger_enum, weight)` → lookup bin number
5. `saveToDatabase(...)` — MySQL insert
6. Push job dict into `_active_jobs[bin_num]`, call `arduino.sendAssign(bin_num)`
7. Emit `classified_signal(job)` — UI table row appears
8. `time.sleep(CLASSIFY_COOLDOWN_S)` then loop

On any failure (weight, YOLO, unknown class): calls `arduino.sendAssign(0)` to restart the motor without sorting, emits `error_signal`.

**`stop()`**: Sets `running = False` and waits for thread to join.

---

#### `MainWindow` (QWidget)
PyQt5 main window, loaded from `old/ui/resultUi.ui`.

**`__init__`**: Loads UI, connects button signals, calls `testFirebaseConnection()`, `loadModel()`, `startArduino()` at startup. Waits 2 s for Arduino reset.

**`onTare()`**: Shows a message box instructing the user to send `forcetare` via the NodeMCU serial monitor (tare is handled on the ESP8266, not from this UI button).

**`onStart()`**:
1. Checks Firebase connectivity (warns but can continue).
2. Calls `initCamera()`.
3. Reads farm name from `cBoxFarm`.
4. Creates and starts `SerialReaderThread`.
5. Sends `"next:"` to kick the motor so the first plate advances to the scale.
6. Creates `VideoThread` and `PipelineThread`.
7. Wires `plate_sorted_signal → pipeline_thread.on_plate_sorted`.
8. Starts both threads.
9. Disables Start button, enables Stop button.

**`onStop()`**:
1. Stops all three threads (`pipeline_thread`, `video_thread`, `serial_reader`) in order.
2. Sends `"motorStop:"` to Arduino to halt the conveyor and clear all job state.
3. Releases the camera.
4. Re-enables Start button.

**`_showFrame(frame)`**: Converts BGR OpenCV frame to QImage/QPixmap and sets it on `lblImg`. Called from `VideoThread.frame_signal`.

**`_onClassified(job)`**: Receives job dict from `PipelineThread.classified_signal`. Updates `lblImg` with the annotated capture image. Inserts a new row in `tblResult` with plate number, class, weight, finger, size, farm, and bin. Updates window title to "sorting…" state.

**`_onSorted(job)`**: Receives job dict from `PipelineThread.sorted_signal`. Updates window title to "sorted" state (confirmation that the banana physically dropped into its bin).

**`_onError(msg)`**: Prints error and updates window title with warning prefix.

---

### ardcommsTest.py — Arduino Serial Driver

#### `arduinoCommunication.__init__(port, baud)`
Opens `serial.Serial(port, baud, timeout=2)` and waits 2 s for Arduino bootloader. Raises on failure.

#### `checkConn() → bool`
Returns `serialComm.is_open`.

#### `close()`
Closes the serial port if open.

#### `reconnect()` / `restart()`
Closes and reopens the serial port with the same port and baud. Waits 2 s.

#### `clearInputBuffer()`
Calls `reset_input_buffer()` to discard any unread bytes.

#### `writeSerial(msg)`
Appends `\n`, encodes to UTF-8, writes to serial, flushes. Used for all outgoing commands.

#### `readSerial() → str`
Reads one line (`readline()`), decodes UTF-8 with `errors='replace'`, strips trailing whitespace. The `errors='replace'` is a deliberate fix — Arduino sends UTF-8 special characters (→, ✓, ✗) which would crash an `ascii` decode.

#### `waitForCameraStop(timeout=25) → bool`
*(Legacy — not used in current pipeline which uses `threading.Event`)*  
Blocking loop reading serial until `"CAM_STOP"` arrives. Drains any stale `"SORT_DONE"` tokens. Returns True on success, False on timeout.

#### `waitForSortDone(timeout=40) → bool`
*(Legacy — not used in current pipeline)*  
Blocking loop reading serial until `"SORT_DONE"` arrives. Drains stale `"CAM_STOP"` tokens.

#### `waitForMotorStop()` / `waitForServoStop()`
Backwards-compatibility aliases that delegate to `waitForCameraStop` and `waitForSortDone` respectively. Print a deprecation notice.

#### `reqWeight() → str`
Sends `"readWt:"`, polls for a response starting with `"readWt:"` within 2 s. Returns the value substring or `-1`. (Used for the Arduino HX711 path, which is superseded by the ESP8266/Firebase path in the current system.)

#### `reqRotateNext()`
Sends `"next:"` — tells Arduino to start the motor until the scale switch fires.

#### `tare()`
Sends `"tare1:"` to trigger plate 1 tare on Arduino side.

#### `reqStartMotor()` / `reqStopMotor()`
Sends `"mtrCtrl:1"` / `"mtrCtrl:0"` to control the motor directly. Legacy — not used in the current event-driven flow.

#### `sendAssign(bin_num: int)`
Sends `"assign:N\n"`. This is the primary command in the current pipeline:
- `bin_num` 1–6 → Arduino activates the job and restarts the motor
- `bin_num` 0 → Arduino skips the plate and restarts the motor (no sort)

#### `servoRotate1()` – `servoRotate6()` / `_trayPos(n)`
Legacy helpers that send `"trayPos:N"`. Kept for backward compatibility — the current firmware uses `assign:N` instead.

---

### BananaSorting_dict.ino — Arduino Mega Main Firmware

#### `setup()`
Initialises all peripherals:
- `Serial.begin(115200)`
- `setupServo()` — attaches all 6 servos and centres them
- `pinMode(motorCtrlPin, OUTPUT)` + `digitalWrite(HIGH)` — motor OFF at start
- `initLimitSwitch()` — sets all 8 limit switch pins to INPUT_PULLUP
- Clears all 5 `PlateJob` slots

#### `loop()`
Calls `mainLoop()` every tick.

#### `mainLoop()`
Fully non-blocking event loop. Six sections run every iteration:

1. **Serial commands** — drains `Serial.available()`, calls `handleSerial()` for each line
2. **Scale switch** — detects HIGH→LOW edge on `limitSw1` (pin 23) with 300 ms debounce; calls `onScaleArrival()`
3. **Bin switches** — detects HIGH→LOW edges on all 6 bin pins with 300 ms debounce; calls `onBinSwitchFired(binNum)`
4. **Assign timeout watchdog** — if any job has been in `waitingAssign` state for >30 s, skips it and restarts the motor to prevent line stall
5. **Post-sort motor restart** — checks `motorRestartAt` timer; when elapsed, calls `motorRotateFunc(0)` to restart the motor after a servo drop
6. **Servo timers** — calls `checkServoTimers()` to process delayed open, close, and neutral-return for all 6 servos

#### `onScaleArrival()`
Triggered by the scale limit switch (limitSw1) edge. Sequence:
1. `delay(SCALE_SETTLE_MS = 700ms)` — lets plate coast into weighing position
2. `motorRotateFunc(1)` — stops motor
3. Records `scaleStopTime = millis()` (for assign-timeout watchdog)
4. Finds a free `PlateJob` slot (index with `active=false, waitingAssign=false`)
5. Marks slot as `waitingAssign=true`
6. `Serial.println("SCALE_STOP")` — signals Python

If no free slot exists, prints a warning and restarts the motor instead (safety: conveyor never stalls).

#### `onBinSwitchFired(binNum)`
Triggered by a bin limit switch edge. For each active job assigned to `binNum`:
1. Increments `jobs[slot].binClickCount`
2. Prints `BIN_CLICK:N,count:C,need:R` debug line
3. If `binClickCount >= REQUIRED_CLICKS[binNum-1]`:
   - Stops motor (`motorRotateFunc(1)`)
   - If `alignDelay[idx] > 0`: schedules `servoFireAt[idx]` (delayed open)
   - Else: calls `fireServo(binNum)` immediately
   - Schedules `motorRestartAt = millis() + rotateDelay`
   - Prints `PLATE_IN_BIN:N`
   - Clears the job slot

Only processes the first matching job per event (one sort per switch fire).

#### `handleSerial(cmd)`
Dispatches serial commands from Python:

| Command | Action |
|---------|--------|
| `assign:N` | Assigns target bin N (1–6) or skip (0) to the waiting job slot. Restarts motor. Prints `ASSIGNED:N` or `ASSIGNED:skip`. |
| `next:` | Force-starts the motor (bypasses `curMotorState` guard). Prints `MOTOR_RUNNING`. |
| `testServo:N` | Directly fires servo N (1–6) for bench testing without the pipeline. |
| `setAlignDelay:<bin>,<ms>` | Adjusts per-bin coast delay after the bin switch fires. |
| `printAlignDelay:` | Prints all 6 align delays to serial. |
| `motorStop:` | Emergency stop: stops motor, clears all job slots. Prints `MOTOR_STOPPED`. |
| `printClicks:` | Prints active job slots with bin assignments and click counts. |
| `checkLimitSw:` | Calls `testLimitSwitch()` — prints all 8 switch states. |
| `checkWires:` | Calls `checkWires()` — full connectivity report. |

---

### motorControl.ino

#### `motorRotateFunc(en)`
Controls the motor relay via `motorCtrlPin` (pin 33, active LOW).
- `en = 0` → `digitalWrite(LOW)` → motor ON
- `en = 1` → `digitalWrite(HIGH)` → motor OFF

Only writes if the requested state differs from `curMotorState` (avoids redundant relay chatter). Always prints `"motor state:N"` to serial for debugging.

---

### servo.ino

#### `setupServo()`
Attaches each of the 6 servos to their pins (Y1→5, Y2→4, Y3→6, Y4→8, Y5→9, Y6→7). Calls `stopallServo()` to centre all gates at startup.

#### `stopallServo()`
Writes 90° to all 6 servos — the mechanical closed/neutral position used at startup.

#### `fireServo(bin)`
Opens the drop gate for the specified bin immediately by writing `servoValRot2` (180°). Non-blocking:
- Sets `servoCloseAt[idx] = millis() + rotateDelay` — close after 1500 ms
- Sets `servoReturnAt[idx] = millis() + 2 × rotateDelay` — neutral return after 3000 ms

`checkServoTimers()` processes these timers each loop tick so no `delay()` is needed.

#### `checkServoTimers()`
Called every `loop()` tick. Iterates all 6 servo slots and handles three timer stages:
- **Stage 0 (`servoFireAt`):** If elapsed, calls `fireServo(i+1)` — delayed open for bins with `alignDelay > 0`
- **Stage 1 (`servoCloseAt`):** If elapsed, writes `servoValRot1` (0°) — closes gate
- **Stage 2 (`servoReturnAt`):** If elapsed, writes 95° — returns to mechanical neutral

This 3-stage sequence (open → close → neutral) prevents mechanical binding.

---

### ultrasonic.ino

#### `initLimitSwitch()`
Sets all 8 limit switch pins (23, 25, 35, 37, 39, 41, 43, 45) to `INPUT_PULLUP`. Called once in `setup()`.

#### `testLimitSwitch()`
Reads and prints the digital state of all 8 limit switch pins. Triggered by `checkLimitSw:` serial command.

#### `readUltrasonicDistance(trigPin, echoPin) → long`
Basic HC-SR04 trigger pulse sequence returning raw `pulseIn` time. (Used in `checkWires`, not in main loop.)

#### `pingUltrasonic(trigPin, echoPin) → long`
Same as above but with a 30 ms hard timeout on `pulseIn` — prevents blocking if no echo returns.

#### `checkWires()`
Full connectivity diagnostic triggered by `checkWires:` serial command. Reports:
- All 8 limit switch states (HIGH = open, LOW = pressed)
- Both ultrasonic sensors with distance reading or FAIL
- Motor control pin state (HIGH = off, LOW = on)
- All 6 servo attachment status

---

### weightSensor.ino

This file is intentionally minimal — it contains only a comment explaining the weight architecture decision:

> Weight sensing is handled by the ESP8266 node, which publishes to Firebase RTDB. The Python host reads weight from Firebase directly.

The Arduino Mega has no HX711 in the current design. All weight data flows through the ESP8266 → Firebase → Python path.

---

### esp8266_enhanced.ino — ESP8266 Weight Node

#### `setup()`
Full initialisation sequence:
1. `Serial.begin(115200)`
2. `EEPROM.begin(512)`
3. `connectWiFi()`
4. Firebase: `signUp()` (anonymous auth), `Firebase.begin()`, `reconnectWiFi(true)`
5. `initLoadCell()` — HX711 init, calibration load, tare

#### `loop()`
Three tasks each iteration:
1. WiFi watchdog — calls `connectWiFi()` if connection dropped
2. Firebase sync — every `SYNC_INTERVAL = 300 ms`, calls `syncDataToFirebase()`
3. Serial command handler — if bytes available, calls `handleSerialCommand()`

#### `connectWiFi()`
Calls `WiFi.begin(SSID, PASSWORD)` and polls up to 40×500ms for connection. Prints IP on success, warning on failure (loop retries).

#### `saveBaseTare(value)` / `loadBaseTare()`
Persists the tray tare baseline (empty tray weight) to EEPROM at byte offset 28. `loadBaseTare()` validates the stored float (must be non-NaN, positive, <5000g) before applying; falls back to compile-time default (912.88g) if invalid.

#### `clearEEPROM()`
Zeroes all 512 EEPROM bytes and resets in-memory calibration to defaults. Triggered by serial `clear` command.

#### `saveCalibration(zo, sf)` / `loadCalibration()`
Persists zero offset (bytes 0–3) and scale factor (bytes 4–7) to EEPROM. Validates scale factor range (0.00005–0.02) before saving to catch implausible values. `loadCalibration()` sets `calibrated = true` if values pass validation.

#### `saveBalancePts()` / `loadBalancePts()`
Persists 5 per-plate balance point floats to EEPROM at bytes 8–27. One balance point per plate (tray weight when empty, used as reference).

#### `initLoadCell()`
1. `scale.begin(D4, D5)` — HX711 on NodeMCU GPIO
2. Retries up to 5 times until `scale.is_ready()`
3. Loads calibration, balance points, and base tare from EEPROM
4. `scale.tare(20)` — initial hardware tare
5. Records first empty-tray reading as `balancePt[currentPlate-1]`

#### `convertToWeight(raw) → float`
Applies calibration: `(raw - ZERO_OFFSET) × SCALE_FACTOR`. Returns weight in grams.

#### `weightValAvg() → float`
Collects 7 raw HX711 readings, sorts them (bubble sort), returns the median converted to grams. Powers down the HX711 for 40 ms between batches to reduce noise. Median-of-7 provides better outlier rejection than the previous median-of-5.

#### `syncDataToFirebase()`
Called every 300 ms from `loop()`.
- If `isTaring`, uploads `Status = "taring"` and returns (Python should not read weight while taring)
- Calls `weightValAvg()` → subtracts `BASE_TARE_WEIGHT` → floors negative to 0
- `Firebase.RTDB.setFloat(&fbdo, "Weight", adjusted)` — the value Python reads
- Also uploads `WeightRaw` (for debug) and `Status = "ready"`

#### `forceTare()`
Interactive tare: waits for user to confirm via serial ("y" + Enter), then:
1. `isTaring = true` (blocks Firebase weight uploads)
2. `scale.tare(25)` — hardware zero
3. Reads and saves new baseline via `saveBaseTare(weightValAvg())`
4. `isTaring = false`

#### `calibrateWeight(plate)`
3-step interactive calibration:
1. Empty plate → `scale.tare(20)` → record zero offset
2. Place known 978g weight → read → compute scale factor = 978 / (reading - zero)
3. Remove weight → record empty-plate balance point → `saveBaseTare(bp)`

#### `printCalibration()`
Prints all stored calibration values to serial: ZO, SF, calibrated flag, BASE_TARE, and all 5 balance points.

#### `handleSerialCommand()`
Dispatches serial commands from USB:

| Command | Action |
|---------|--------|
| `cal1`–`cal5` | Calibrate specific plate |
| `info` | Print all calibration values |
| `weight` | Print current adjusted weight |
| `sync` | Force one Firebase sync now |
| `forcetare` | Interactive re-tare + save |
| `settare:<g>` | Manually set tare base in grams |
| `clear` | Wipe EEPROM |
| `diag` | Run HX711 wiring diagnostic |
| `plate1`–`plate5` | Switch active plate |
| `help` | Print command list |

#### `checkHX711Wiring()`
Hardware diagnostic triggered by `diag` command:
1. Re-initialises HX711 (fresh begin)
2. Reports `is_ready()` status
3. Takes 3 raw readings and reports variance (>50000 → warns of noisy wiring)
4. Power-cycles HX711 and confirms it comes back ready

---

## 6. Serial Protocol Reference

### Python → Arduino

| Command | When sent | Effect |
|---------|-----------|--------|
| `next:\n` | `onStart()` — first plate kickoff | Force-starts motor; scale switch stops it |
| `assign:N\n` | After each plate is classified | N=1–6: activate job, restart motor. N=0: skip plate, restart motor |
| `motorStop:\n` | `onStop()` | Stops motor, clears all job slots |
| `testServo:N\n` | Manual testing | Fires servo N directly |
| `setAlignDelay:<bin>,<ms>\n` | Manual tuning | Adjusts per-bin coast delay |
| `printAlignDelay:\n` | Diagnostic | Print current delays |
| `printClicks:\n` | Diagnostic | Print active job click counts |
| `checkLimitSw:\n` | Diagnostic | Print all switch states |
| `checkWires:\n` | Diagnostic | Full wiring check |

### Arduino → Python

| Token | Sent when | Consumed by |
|-------|-----------|-------------|
| `SCALE_STOP` | Plate arrives and motor stops | `SerialReaderThread` → sets `scale_event` → wakes `PipelineThread` |
| `PLATE_IN_BIN:N` | Plate confirmed in bin N (servo fired) | `SerialReaderThread` → `plate_sorted_signal` → `on_plate_sorted()` → `sorted_signal` → UI |
| `ASSIGNED:N` | After `assign:N` processed | Printed to console (not acted on by pipeline) |
| `ASSIGNED:skip` | After `assign:0` processed | Printed to console |
| `BIN_CLICK:N,count:C,need:R` | Each bin switch edge for active job | `SerialReaderThread` → pretty-printed progress bar to console |
| `MOTOR_RUNNING` | After `next:` command | Console only |
| `MOTOR_STOPPED` | After `motorStop:` command | Console only |
| `WARN: ...` | Watchdog / error conditions | Console only |

---

## 7. Classification Logic

Classification is a two-dimensional lookup on (finger count, weight range):

```
                     Weight (grams)
                  350  400  466  521  541  621
                   │    │    │    │    │    │   731
3-finger           │IF38│    IF36 │    30TR │
4 or 5-finger           │33BCP│  30BCP │ 25BCP │
```

| Class | Bin | Fingers | Weight Range |
|-------|-----|---------|-------------|
| 33BCP | 1 | 4 or 5 | 400–520g |
| 25BCP | 2 | 4 or 5 | 621–730g |
| 30BCP | 3 | 4 or 5 | 521–620g |
| IF38TR | 4 | 3 | 350–465g |
| IF36TR | 5 | 3 | 466–540g |
| 30TR | 6 | 3 | 541–650g |

**Weight ranges are non-overlapping within each finger-count group.** Any combination outside these ranges returns `BananaClass.UNKNOWN` and skips sorting.

---

## 8. Bin Positioning Logic

The conveyor is circular with 5 plate slots. When a plate is assigned bin N, it may pass several bin switches before reaching its target. The Arduino counts only fires of the **assigned bin's own switch** — other switch fires are irrelevant to that job.

`REQUIRED_CLICKS[6] = {3, 4, 5, 5, 6, 5}` (bins 1–6)

| Bin | Required clicks | Switch pin | Notes |
|-----|:--------------:|:----------:|-------|
| 1 | 3 | 35 | Closest to scale |
| 2 | 4 | 37 | |
| 3 | 5 | 39 | |
| 4 | 5 | 41 | |
| 5 | 6 | 43 | |
| 6 | 5 | 45 | Farthest from scale |

`alignDelay[6] = {200, 0, 0, 0, 0, 0}` ms — bin 1 coasts 200 ms after switch before servo opens, all others fire immediately.

**Servo sequence:**
1. Switch edge detected → count incremented
2. If count == REQUIRED_CLICKS: motor stops, `alignDelay` coast (if any), servo opens (180°)
3. After `rotateDelay` (1500 ms): servo closes (0°)
4. After `2 × rotateDelay` (3000 ms): servo returns to neutral (95°)
5. Motor restarts after `rotateDelay` ms from stop

---

## 9. Threading Model

```
Main Thread (Qt event loop)
│
├── VideoThread                 ← reads camera frames at 30 fps
│   └── frame_signal → MainWindow._showFrame()
│
├── SerialReaderThread          ← reads all Arduino serial output
│   ├── scale_event.set()       (unblocks PipelineThread)
│   └── plate_sorted_signal(N)  → PipelineThread.on_plate_sorted(N)
│
└── PipelineThread              ← classify/sort loop
    ├── classified_signal(job)  → MainWindow._onClassified()
    ├── sorted_signal(job)      → MainWindow._onSorted()
    └── error_signal(msg)       → MainWindow._onError()
```

**Thread safety notes:**
- `VideoThread._frame` is protected by `threading.Lock` — `get_latest_frame()` holds the lock only for a `.copy()` call
- `SerialReaderThread.scale_event` is a `threading.Event` — safe to `set()` from one thread and `wait()` from another
- `PipelineThread._active_jobs` is mutated from two threads: `_process_one_plate()` appends (PipelineThread), `on_plate_sorted()` pops (called via Qt signal from main thread). Qt signals are queued across threads so `on_plate_sorted` is actually dispatched on the main thread, but `_active_jobs` is shared state — currently no explicit lock on this dict
- All Qt UI updates (`setPixmap`, `insertRow`, `setWindowTitle`) happen via signals which are safely queued to the main thread

---

## 10. Known Design Notes & Constraints

1. **Single motor for belt + sort:** The conveyor belt and sorting travel use the same physical motor. This is why the pipeline cannot pipeline multiple plates through classification simultaneously — all movement stops at the scale while Python classifies, then the motor runs again. Python does NOT stop the motor for sorting in the current design; the Arduino stops it per-bin.

2. **5-plate capacity:** `jobs[5]` — the Arduino supports at most 5 plates in transit simultaneously. If a 6th plate arrives while all slots are full, it logs a warning and restarts the motor (plate passes through unsorted).

3. **`assign:N` vs `trayPos:N`:** The current firmware uses `assign:N`. The `trayPos:N` command still exists in `ardcommsTest.py` as legacy but is not used by the current pipeline. Old Arduino firmware used `trayPos:N` with blocking motor control; the current version is fully event-driven.

4. **Weight source:** The Arduino Mega's `weightSensor.ino` is a stub. All weight data comes from the ESP8266 → Firebase → Python path. The `readWt:` command and `reqWeight()` method in `ardcommsTest.py` are legacy and unused.

5. **`SCALE_STOP` vs `CAM_STOP`:** The Arduino now sends `SCALE_STOP` when a plate reaches the weighing station (camera is co-located). The old protocol used `CAM_STOP`. The `waitForCameraStop()` method in `ardcommsTest.py` still looks for `CAM_STOP` — this is dead code in the current pipeline (the `SerialReaderThread` handles `SCALE_STOP` instead).

6. **`servo.ino` commented-out blocks:** Large sections of old servo logic are preserved as comments for rollback reference. The active code is the `fireServo` / `checkServoTimers` non-blocking approach.

7. **Firebase credentials:** WiFi SSID/password and Firebase API key are hardcoded in `esp8266_enhanced.ino`. For production these should be moved to a config header or provisioned differently.

8. **MySQL at startup:** `mysql.connector.connect(...)` is called at module import time in `SortQue.py`. If MySQL is unavailable the application crashes before the window appears. `test_run.py` mocks this for UI testing.

9. **Debounce:** All limit switches use a 300 ms software debounce (`DEBOUNCE_MS`). If the conveyor speed changes and a plate moves through a switch faster than 300 ms, clicks could be dropped. Tune `DEBOUNCE_MS` if click counts are unreliable at high speed.

10. **`alignDelay` for bin 1 only:** `alignDelay[6] = {200, 0, 0, 0, 0, 0}` — only bin 1 has a non-zero coast delay. The REQUIRED_CLICKS for other bins can be fine-tuned via `setAlignDelay:<bin>,<ms>` serial command at runtime without reflashing.
