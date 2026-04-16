"""
bin_travel_test.py  —  Bin Travel & Alignment Tuner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manual testing tool for the circular conveyor.
• Waits for SCALE_STOP, then lets you assign any bin.
• Displays live BIN_CLICK counters for each active job.
• Lets you adjust alignDelay per bin and send it live.
• Servo test buttons (testServo:N) to verify gate movement.

REQUIRED_CLICKS is compiled into the Arduino firmware.
To change it: edit REQUIRED_CLICKS[] in BananaSorting_dict.ino and re-upload.
"""

import sys
import time
import threading
import serial
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QGroupBox, QGridLayout, QSplitter, QFrame, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor

# ─── Config ─────────────────────────────────────────────────────────────────
ARDUINO_PORT = "COM5"
ARDUINO_BAUD = 115200

# Classification labels shown in the assign dropdown (matches CLAUDE.md)
BIN_LABELS = {
    1: "Bin 1 — 33BCP  (4-5f, 400-520g)",
    2: "Bin 2 — 25BCP  (4-5f, 621-730g)",
    3: "Bin 3 — 30BCP  (4-5f, 521-620g)",
    4: "Bin 4 — IF38TR (3f,   350-465g)",
    5: "Bin 5 — IF36TR (3f,   466-540g)",
    6: "Bin 6 — 30TR   (3f,   541-650g)",
}

# Default REQUIRED_CLICKS (mirrors firmware; read-only here — edit firmware to change)
REQUIRED_CLICKS_DEFAULT = [3, 4, 5, 6, 6, 6]


# ─── Serial Reader Thread ────────────────────────────────────────────────────
class SerialReader(QObject):
    line_received = pyqtSignal(str)
    connection_lost = pyqtSignal()

    def __init__(self, port, baud):
        super().__init__()
        self._running = False
        self._port = port
        self._baud = baud
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(self._port, self._baud, timeout=1)
        time.sleep(2)

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def write(self, msg: str):
        if self.ser and self.ser.is_open:
            self.ser.write((msg + '\n').encode('utf-8'))
            self.ser.flush()

    def run(self):
        self._running = True
        while self._running:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='replace').rstrip()
                    if line:
                        self.line_received.emit(line)
                else:
                    time.sleep(0.02)
            except serial.SerialException:
                self.connection_lost.emit()
                break
            except Exception:
                time.sleep(0.05)


# ─── Main Window ─────────────────────────────────────────────────────────────
class BinTravelTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bin Travel & Alignment Tuner")
        self.resize(1100, 700)

        self.reader = None
        self.reader_thread = None
        self._plate_waiting = False

        # Track click counts per bin (reset on PLATE_IN_BIN)
        self.click_counts = [0] * 6
        # Track required clicks (read from printClicks: response)
        self.required_clicks = list(REQUIRED_CLICKS_DEFAULT)
        # Align delays (editable)
        self.align_delays = [200, 200, 0, 0, 200, 100]

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._build_connection_group())
        left_layout.addWidget(self._build_assign_group())
        left_layout.addWidget(self._build_bin_table_group())
        left_layout.addWidget(self._build_controls_group())
        left_layout.addStretch()
        splitter.addWidget(left)

        # Right panel — serial log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Serial Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.log)

        # Manual send box
        send_row = QHBoxLayout()
        self.send_input = QLineEdit()
        self.send_input.setPlaceholderText("Send raw command...")
        self.send_input.returnPressed.connect(self._on_manual_send)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_manual_send)
        send_row.addWidget(self.send_input)
        send_row.addWidget(send_btn)
        right_layout.addLayout(send_row)

        splitter.addWidget(right)
        splitter.setSizes([550, 550])

    def _build_connection_group(self):
        grp = QGroupBox("Connection")
        lay = QHBoxLayout(grp)
        self.port_input = QLineEdit(ARDUINO_PORT)
        self.port_input.setFixedWidth(70)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._on_toggle_connect)
        self.conn_status = QLabel("Disconnected")
        self.conn_status.setStyleSheet("color: red; font-weight: bold;")
        lay.addWidget(QLabel("Port:"))
        lay.addWidget(self.port_input)
        lay.addWidget(self.connect_btn)
        lay.addWidget(self.conn_status)
        lay.addStretch()
        return grp

    def _build_assign_group(self):
        grp = QGroupBox("Manual Assign  (waits for SCALE_STOP)")
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = QVBoxLayout(grp)

        self.scale_status = QLabel("No plate at scale")
        self.scale_status.setAlignment(Qt.AlignCenter)
        self.scale_status.setFont(QFont("Arial", 11))
        lay.addWidget(self.scale_status)

        row = QHBoxLayout()
        self.bin_combo = QComboBox()
        for n in range(1, 7):
            self.bin_combo.addItem(BIN_LABELS[n], n)
        self.bin_combo.setMinimumWidth(260)

        self.assign_btn = QPushButton("Assign Bin")
        self.assign_btn.setEnabled(False)
        self.assign_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.assign_btn.clicked.connect(self._on_assign)

        self.skip_btn = QPushButton("Skip (bin 0)")
        self.skip_btn.setEnabled(False)
        self.skip_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 6px;")
        self.skip_btn.clicked.connect(self._on_skip)

        row.addWidget(self.bin_combo)
        row.addWidget(self.assign_btn)
        row.addWidget(self.skip_btn)
        lay.addLayout(row)
        return grp

    def _build_bin_table_group(self):
        grp = QGroupBox("Per-Bin Config")
        grid = QGridLayout(grp)
        grid.setSpacing(4)

        headers = ["Bin", "Classification", "Req. Clicks\n(firmware)", "Clicks\nSeen", "Align Delay (ms)", ""]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setFont(QFont("Arial", 8, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)

        self.click_labels  = []
        self.delay_spins   = []
        self.req_click_labels = []

        for i in range(6):
            bin_n = i + 1
            row = i + 1

            grid.addWidget(QLabel(f"  {bin_n}"), row, 0)
            grid.addWidget(QLabel(BIN_LABELS[bin_n].split("—")[1].strip()[:14]), row, 1)

            # Required clicks (read-only, mirrors firmware)
            rc_lbl = QLabel(str(self.required_clicks[i]))
            rc_lbl.setAlignment(Qt.AlignCenter)
            rc_lbl.setToolTip("Edit REQUIRED_CLICKS[] in firmware to change")
            self.req_click_labels.append(rc_lbl)
            grid.addWidget(rc_lbl, row, 2)

            # Live click count
            ck_lbl = QLabel("0 / " + str(self.required_clicks[i]))
            ck_lbl.setAlignment(Qt.AlignCenter)
            ck_lbl.setFont(QFont("Consolas", 9))
            self.click_labels.append(ck_lbl)
            grid.addWidget(ck_lbl, row, 3)

            # Align delay spin
            spin = QSpinBox()
            spin.setRange(0, 2000)
            spin.setValue(self.align_delays[i])
            spin.setSuffix(" ms")
            self.delay_spins.append(spin)
            grid.addWidget(spin, row, 4)

            # Set button
            set_btn = QPushButton("Set")
            set_btn.setFixedWidth(40)
            set_btn.clicked.connect(lambda checked, b=bin_n: self._on_set_align(b))
            grid.addWidget(set_btn, row, 5)

        return grp

    def _build_controls_group(self):
        grp = QGroupBox("Controls")
        lay = QGridLayout(grp)

        motor_start = QPushButton("Motor Start  (next:)")
        motor_start.setStyleSheet("background-color: #2196F3; color: white; padding: 4px;")
        motor_start.clicked.connect(lambda: self._send("next:"))

        motor_stop = QPushButton("STOP Motor")
        motor_stop.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 4px;")
        motor_stop.clicked.connect(lambda: self._send("motorStop:"))

        print_clicks = QPushButton("Print Clicks")
        print_clicks.clicked.connect(lambda: self._send("printClicks:"))

        print_align = QPushButton("Print Align Delays")
        print_align.clicked.connect(lambda: self._send("printAlignDelay:"))

        set_all_btn = QPushButton("Push ALL Align Delays")
        set_all_btn.clicked.connect(self._on_set_all_align)

        lay.addWidget(motor_start,   0, 0)
        lay.addWidget(motor_stop,    0, 1)
        lay.addWidget(print_clicks,  1, 0)
        lay.addWidget(print_align,   1, 1)
        lay.addWidget(set_all_btn,   2, 0, 1, 2)

        # Servo test buttons
        servo_lbl = QLabel("Test Servos:")
        servo_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        lay.addWidget(servo_lbl, 3, 0, 1, 2)

        servo_row = QHBoxLayout()
        for n in range(1, 7):
            btn = QPushButton(f"S{n}")
            btn.setFixedWidth(38)
            btn.setToolTip(f"testServo:{n}")
            btn.clicked.connect(lambda checked, b=n: self._send(f"testServo:{b}"))
            servo_row.addWidget(btn)

        servo_widget = QWidget()
        servo_widget.setLayout(servo_row)
        lay.addWidget(servo_widget, 4, 0, 1, 2)

        return grp

    # ── Serial ───────────────────────────────────────────────────────────────
    def _send(self, msg: str):
        if self.reader:
            self.reader.write(msg)
            self._log(f">> {msg}", color="#4FC3F7")
        else:
            self._log("Not connected", color="orange")

    def _on_manual_send(self):
        cmd = self.send_input.text().strip()
        if cmd:
            self._send(cmd)
            self.send_input.clear()

    def _on_toggle_connect(self, checked):
        if checked:
            self._do_connect()
        else:
            self._do_disconnect()

    def _do_connect(self):
        port = self.port_input.text().strip()
        try:
            self.reader = SerialReader(port, ARDUINO_BAUD)
            self.reader.line_received.connect(self._on_line)
            self.reader.connection_lost.connect(self._on_conn_lost)
            self.reader.connect()

            self.reader_thread = threading.Thread(target=self.reader.run, daemon=True)
            self.reader_thread.start()

            self.conn_status.setText(f"Connected ({port})")
            self.conn_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.connect_btn.setText("Disconnect")
            self._log(f"Connected to {port} @ {ARDUINO_BAUD}", color="#4CAF50")

            # Query current state
            time.sleep(0.3)
            self._send("printClicks:")
            self._send("printAlignDelay:")
        except Exception as e:
            self._log(f"Connection failed: {e}", color="red")
            self.connect_btn.setChecked(False)

    def _do_disconnect(self):
        if self.reader:
            self.reader.close()
            self.reader = None
        self.conn_status.setText("Disconnected")
        self.conn_status.setStyleSheet("color: red; font-weight: bold;")
        self.connect_btn.setText("Connect")
        self._log("Disconnected", color="orange")

    def _on_conn_lost(self):
        self._log("Connection lost!", color="red")
        self._do_disconnect()
        self.connect_btn.setChecked(False)

    # ── Incoming serial parser ────────────────────────────────────────────────
    def _on_line(self, line: str):
        self._log(f"<< {line}")

        # Plate arrived at scale — prompt user to assign
        if line == "SCALE_STOP":
            self._plate_waiting = True
            self.scale_status.setText("⬤  PLATE AT SCALE — assign a bin!")
            self.scale_status.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.assign_btn.setEnabled(True)
            self.skip_btn.setEnabled(True)

        # Plate assigned successfully
        elif line.startswith("ASSIGNED:"):
            val = line[9:]
            if val == "skip":
                self.scale_status.setText("Skipped — motor restarting")
                self.scale_status.setStyleSheet("color: gray;")
            else:
                self.scale_status.setText(f"Assigned → Bin {val} — in transit")
                self.scale_status.setStyleSheet("color: #2196F3;")
            self._plate_waiting = False
            self.assign_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)

        # Bin click progress
        elif line.startswith("BIN_CLICK:"):
            # Format: BIN_CLICK:N,count:C,need:R
            try:
                parts = line[len("BIN_CLICK:"):].split(",")
                bin_n  = int(parts[0])
                count  = int(parts[1].split(":")[1])
                needed = int(parts[2].split(":")[1])
                idx    = bin_n - 1
                self.click_counts[idx]   = count
                self.required_clicks[idx] = needed
                self.req_click_labels[idx].setText(str(needed))
                self.click_labels[idx].setText(f"{count} / {needed}")
                # Highlight in-progress bin
                self.click_labels[idx].setStyleSheet("color: #FF9800; font-weight: bold;")
            except Exception:
                pass

        # Plate reached bin — reset counter display
        elif line.startswith("PLATE_IN_BIN:"):
            try:
                bin_n = int(line[len("PLATE_IN_BIN:"):])
                idx   = bin_n - 1
                self.click_counts[idx] = 0
                self.click_labels[idx].setText(f"DONE → {self.required_clicks[idx]}")
                self.click_labels[idx].setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.scale_status.setText(f"Bin {bin_n} sort complete")
                self.scale_status.setStyleSheet("color: #4CAF50;")
            except Exception:
                pass

        # Parse printAlignDelay: response to sync spinboxes
        elif line.startswith("alignDelay["):
            # Format: alignDelay[N] = X ms
            try:
                bracket = line.index("[") + 1
                close   = line.index("]")
                eq      = line.index("=") + 2
                bin_n   = int(line[bracket:close])
                ms_str  = line[eq:].replace(" ms", "").strip()
                ms      = int(ms_str)
                idx     = bin_n - 1
                self.align_delays[idx] = ms
                self.delay_spins[idx].blockSignals(True)
                self.delay_spins[idx].setValue(ms)
                self.delay_spins[idx].blockSignals(False)
            except Exception:
                pass

        # Parse printClicks: required click values
        elif line.startswith("  bin") and ":" in line:
            try:
                colon = line.index(":") + 1
                parts = line.split(":")
                # Format: "  binN: X"
                n_str = parts[0].strip().replace("bin", "")
                bin_n = int(n_str)
                req   = int(parts[1].strip())
                idx   = bin_n - 1
                self.required_clicks[idx] = req
                self.req_click_labels[idx].setText(str(req))
                self.click_labels[idx].setText(f"0 / {req}")
            except Exception:
                pass

    # ── Actions ──────────────────────────────────────────────────────────────
    def _on_assign(self):
        bin_n = self.bin_combo.currentData()
        self._send(f"assign:{bin_n}")
        self.assign_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)

    def _on_skip(self):
        self._send("assign:0")
        self.assign_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.scale_status.setText("Skipping...")
        self.scale_status.setStyleSheet("color: gray;")

    def _on_set_align(self, bin_n: int):
        ms = self.delay_spins[bin_n - 1].value()
        self._send(f"setAlignDelay:{bin_n},{ms}")

    def _on_set_all_align(self):
        for i in range(6):
            ms = self.delay_spins[i].value()
            self._send(f"setAlignDelay:{i + 1},{ms}")
            time.sleep(0.05)

    # ── Log ──────────────────────────────────────────────────────────────────
    def _log(self, text: str, color: str = "white"):
        ts = time.strftime("%H:%M:%S")
        html = f'<span style="color:{color};">[{ts}] {text}</span>'
        self.log.append(html)
        self.log.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        if self.reader:
            self.reader.close()
        event.accept()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = BinTravelTester()
    win.show()
    sys.exit(app.exec_())
