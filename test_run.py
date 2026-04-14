"""
test_run.py — Smoke-test launcher for SortQue.py
Mocks COM5 (serial) and MySQL so the app can start without hardware.
Run from capstone_codes/:  python test_run.py
"""

import sys
import types
from unittest.mock import MagicMock

# ── 1. Mock pyserial (before any import touches it) ─────────────────
serial_mod = types.ModuleType("serial")

class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.is_open     = True
        self.in_waiting  = 0
        print(f"[MOCK] serial.Serial({args}, {kwargs}) — pretending COM5 is connected")

    def write(self, data):        pass
    def flush(self):              pass
    def readline(self):           return b""
    def read(self, n=1):          return b""
    def reset_input_buffer(self): pass
    def close(self):              self.is_open = False

serial_mod.Serial = FakeSerial
sys.modules["serial"] = serial_mod

# ── 2. Mock mysql.connector (before SortQue module-level connect) ────
mysql_mod          = types.ModuleType("mysql")
mysql_conn_mod     = types.ModuleType("mysql.connector")
mock_db            = MagicMock()
mock_cursor        = MagicMock()
mock_cursor.execute = lambda *a, **k: None
mock_db.cursor.return_value  = mock_cursor
mock_db.commit               = lambda: None
mysql_conn_mod.connect       = MagicMock(return_value=mock_db)
mysql_conn_mod.Error         = Exception
mysql_mod.connector          = mysql_conn_mod
sys.modules["mysql"]             = mysql_mod
sys.modules["mysql.connector"]   = mysql_conn_mod
print("[MOCK] MySQL connector — DB calls will be no-ops")

# ── 3. Add old/ to path (ardcommsTest lives there) ───────────────────
sys.path.insert(0, "old")

# ── 4. Launch the app as __main__ (triggers the if __name__ block) ──
print("[TEST] Launching SortQue...\n")
import runpy
runpy.run_path("old/SortQue.py", run_name="__main__")
