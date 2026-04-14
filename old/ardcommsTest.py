import serial
import time

class arduinoCommunication:
    def __init__(self, port, baud):
        self.portName  = port
        self.baudRate  = baud
        try:
            self.serialComm = serial.Serial(self.portName, self.baudRate, timeout=2)
            time.sleep(2)
            print("Arduino connected successfully")
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            raise

    # ── connection helpers ───────────────────────────────────────────
    def checkConn(self):   
        return self.serialComm.is_open
    
    def close(self):
        if self.serialComm.is_open:
            self.serialComm.close()
            print("Serial connection closed")

    def reconnect(self):
        try:
            self.close()
            self.serialComm = serial.Serial(self.portName, self.baudRate, timeout=2)
            time.sleep(2)
            print("Reconnected to Arduino")
        except Exception as e:
            print(f"Reconnection failed: {e}")
            raise

    def restart(self):  
        self.reconnect()

    def clearInputBuffer(self):
        self.serialComm.reset_input_buffer()
        print("Input buffer cleared")

    # ── low-level I/O ────────────────────────────────────────────────
    def writeSerial(self, msg):
        try:
            self.serialComm.write((msg + '\n').encode('utf-8'))
            self.serialComm.flush()
        except Exception as e:
            print(f"Write failed: {e}")
            raise

    def readSerial(self):
        try:
            # FIX: was decode('ascii') — Arduino sends UTF-8 chars (→, ✓, ✗)
            # which are multi-byte and crash ascii decode. Use utf-8 with
            # errors='replace' so a bad byte never kills the wait loop.
            return self.serialComm.readline().decode('utf-8', errors='replace').rstrip()
        except Exception as e:
            print(f"Read failed: {e}")
            raise

    # ── FIX: two separate wait methods with DISTINCT expected tokens ─
    # Arduino must send "CAM_STOP"  after rotateNextSwitchTrig completes
    # Arduino must send "SORT_DONE" after rotateAndSort completes
    # This eliminates the ambiguity that caused premature cycle starts.

    def waitForCameraStop(self, timeout=25):
        """
        Wait for 'CAM_STOP' — sent by Arduino after tray reaches camera position.
        Was previously waitForMotorStop() listening for generic 'MOTOR_STOP'.
        """
        try:
            start = time.time()
            while time.time() - start < timeout:
                if self.serialComm.in_waiting > 0:
                    line = self.readSerial()
                    print(f"  [serial rx] '{line}'")
                    if line == "CAM_STOP":
                        print("✓ Tray at camera position")
                        return True
                    # Drain any stale SORT_DONE from previous cycle
                    if line == "SORT_DONE":
                        print("  [stale SORT_DONE drained]")
                time.sleep(0.05)
            print("✗ Timeout waiting for CAM_STOP")
            return False
        except Exception as e:
            print(f"waitForCameraStop error: {e}")
            return False

    def waitForSortDone(self, timeout=40):
        """
        Wait for 'SORT_DONE' — sent by Arduino after rotateAndSort + fireServo complete.
        Was previously waitForServoStop() listening for generic 'MOTOR_STOP'.
        """
        try:
            start = time.time()
            while time.time() - start < timeout:
                if self.serialComm.in_waiting > 0:
                    line = self.readSerial()
                    print(f"  [serial rx] '{line}'")
                    if line == "SORT_DONE":
                        print("✓ Sort operation complete")
                        return True
                    # Drain any stale CAM_STOP — should not appear here but just in case
                    if line == "CAM_STOP":
                        print("  [stale CAM_STOP drained]")
                time.sleep(0.05)
            print("✗ Timeout waiting for SORT_DONE")
            return False
        except Exception as e:
            print(f"waitForSortDone error: {e}")
            return False

    # Keep old names as aliases so existing code doesn't break immediately,
    # but they now call the correct disambiguated methods.
    def waitForMotorStop(self, timeout=25):
        print("  [waitForMotorStop → waitForCameraStop]")
        return self.waitForCameraStop(timeout)

    def waitForServoStop(self, timeout=40):
        print("  [waitForServoStop → waitForSortDone]")
        return self.waitForSortDone(timeout)

    # ── commands ─────────────────────────────────────────────────────
    def reqWeight(self):
        try:
            self.writeSerial("readWt:")
            start = time.time()
            while time.time() - start < 2:
                if self.serialComm.in_waiting > 0:
                    msg = self.readSerial()
                    if msg.startswith("readWt:"):
                        return msg[7:]
                time.sleep(0.1)
            print("Timeout waiting for weight response")
            return -1
        except Exception as e:
            print(f"Weight read failed: {e}")
            return -1

    def reqRotateNext(self):
        try:
            self.writeSerial("next:")
            print("Sent next:")
        except Exception as e:
            print(f"Rotate command failed: {e}")
            raise

    def tare(self):
        try:
            self.writeSerial("tare1:")
            print("Sent tare")
        except Exception as e:
            print(f"Tare failed: {e}")
            raise

    def reqStartMotor(self):
        try:
            self.writeSerial("mtrCtrl:1")
        except Exception as e:
            print(f"Start motor failed: {e}")
            raise

    def reqStopMotor(self):
        try:
            self.writeSerial("mtrCtrl:0")
        except Exception as e:
            print(f"Stop motor failed: {e}")
            raise

    def servoRotate1(self): self._trayPos(1)
    def servoRotate2(self): self._trayPos(2)
    def servoRotate3(self): self._trayPos(3)
    def servoRotate4(self): self._trayPos(4)
    def servoRotate5(self): self._trayPos(5)
    def servoRotate6(self): self._trayPos(6)

    def _trayPos(self, n):
        try:
            self.writeSerial(f"trayPos:{n}")
            print(f"Sent trayPos:{n}")
        except Exception as e:
            print(f"trayPos:{n} failed: {e}")
            raise