"""
test_yolo.py
────────────────────────────────────────────────────────────────────────────
Unit + integration tests for YOLO capture and banana-classification logic
in old/SortQue.py.  Runs entirely without hardware.

Run:
    python -m pytest test_yolo.py -v
    python test_yolo.py
"""
import sys, os, unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Stub every heavy / hardware dependency BEFORE importing SortQue
# ══════════════════════════════════════════════════════════════════════════════

# ── mysql ─────────────────────────────────────────────────────────────────────
_mc = MagicMock()
_mc.Error = type("MySQLError", (Exception,), {})
_mc.connect.return_value = MagicMock()
_mysql_mod = MagicMock()
_mysql_mod.connector = _mc
sys.modules["mysql"]           = _mysql_mod
sys.modules["mysql.connector"] = _mc

# ── PyQt5 — QThread / QWidget need to be *real classes* so subclasses work ──
class _FakeQThread:
    def __init__(self, *a, **kw): pass
    def start(self): pass
    def wait(self, ms=None): pass
    def msleep(self, ms): pass
    def isRunning(self): return False
    def quit(self): pass

class _FakeQWidget:
    def __init__(self, *a, **kw): pass

def _fake_signal(*args, **kwargs):
    """Stand-in for pyqtSignal; returns a MagicMock with .connect/.emit."""
    return MagicMock()

_qtcore                 = MagicMock()
_qtcore.QThread         = _FakeQThread
_qtcore.pyqtSignal      = _fake_signal
_qtcore.Qt              = MagicMock()

_qtwidgets              = MagicMock()
_qtwidgets.QWidget      = _FakeQWidget

sys.modules["PyQt5"]           = MagicMock()
sys.modules["PyQt5.QtCore"]    = _qtcore
sys.modules["PyQt5.QtWidgets"] = _qtwidgets
sys.modules["PyQt5.QtGui"]     = MagicMock()
sys.modules["PyQt5.uic"]       = MagicMock()

# ── misc hardware / network ───────────────────────────────────────────────────
sys.modules["ardcommsTest"] = MagicMock()
sys.modules["requests"]     = MagicMock()

_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
sys.modules["torch"] = _torch_mock

# ── cv2 — set return values that get tuple-unpacked inside captureImage ───────
_cv2_mock = MagicMock()
_cv2_mock.getTextSize.return_value = ((10, 10), 0)          # (tw, th), baseline
_cv2_mock.addWeighted.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
_cv2_mock.FONT_HERSHEY_SIMPLEX     = 0
_cv2_mock.COLOR_BGR2RGB            = 4
sys.modules["cv2"] = _cv2_mock

# ── ultralytics — YOLO global replaced per-test via patch ────────────────────
sys.modules["ultralytics"] = MagicMock()

# ══════════════════════════════════════════════════════════════════════════════
# Import SortQue from old/
# ══════════════════════════════════════════════════════════════════════════════
_old_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "old")
if _old_dir not in sys.path:
    sys.path.insert(0, _old_dir)

import SortQue  # noqa: E402
from SortQue import (
    captureImage, parseFinger, inferHand, classifyBanana,
    FingerCount, HandSize, BananaClass, CLASS_TO_BIN, Config,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
_BLANK = np.zeros((640, 640, 3), dtype=np.uint8)
# 4-point polygon — satisfies the len(m) >= 3 mask filter in captureImage
_POLY  = np.array([[0., 0.], [200., 0.], [200., 200.], [0., 200.]])


def _yolo_result(n_masks: int, conf: float = 0.90):
    """Minimal fake ultralytics prediction result with *n_masks* hands."""
    r = MagicMock()
    if n_masks == 0:
        r.masks      = None
        r.boxes.conf = np.array([])
        return r
    r.masks      = MagicMock()
    r.masks.xy   = [_POLY.copy() for _ in range(n_masks)]
    r.boxes.conf = np.full(n_masks, conf, dtype=float)
    r.boxes.xyxy = np.tile([10., 20., 200., 210.], (n_masks, 1))
    return r


def _make_pipeline(capture_results, conf=0.90):
    """
    Build (get_frame_fn, mock_model) ready for captureImage().

    capture_results — list of ints: mask-count per CAPTURE_FRAMES call.
    NOTE: model.predict is NOT called during the flush phase, so only
          len(capture_results) results are needed (== Config.CAPTURE_FRAMES).
    """
    results_iter = iter(_yolo_result(n, conf) for n in capture_results)
    mock_model   = MagicMock()
    mock_model.predict.side_effect = (
        lambda *a, **kw: [next(results_iter, _yolo_result(0))]
    )
    def get_frame():
        return (True, _BLANK.copy())
    return get_frame, mock_model


def _run(capture_results, conf=0.90):
    """Patch SortQue.model and call captureImage; return the result dict."""
    gf, mm = _make_pipeline(capture_results, conf)
    with patch.object(SortQue, "model", mm):
        return captureImage(gf)


# ══════════════════════════════════════════════════════════════════════════════
class TestParseFinger(unittest.TestCase):
    """parseFinger: raw string → FingerCount enum."""

    def test_3_finger(self):
        for s in ("3-finger", "3", "hand-3", "count_3"):
            with self.subTest(s=s):
                self.assertEqual(parseFinger(s), FingerCount.THREE)

    def test_4_finger(self):
        for s in ("4-finger", "4", "label4"):
            with self.subTest(s=s):
                self.assertEqual(parseFinger(s), FingerCount.FOUR)

    def test_5_finger(self):
        for s in ("5-finger", "5"):
            with self.subTest(s=s):
                self.assertEqual(parseFinger(s), FingerCount.FIVE)

    def test_unknown(self):
        for s in ("", "banana", "2-finger", "6", "abc", " "):
            with self.subTest(s=s):
                self.assertEqual(parseFinger(s), FingerCount.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════════════
class TestInferHand(unittest.TestCase):
    """inferHand: (FingerCount, weight) → HandSize."""

    # 3-finger valid range: W_IF38TR_MIN(350) .. W_30TR_MAX(650)
    def test_3f_in_range(self):
        for w in (350, 400, 500, 600, 650):
            with self.subTest(w=w):
                self.assertEqual(inferHand(FingerCount.THREE, w), HandSize.REGULAR)

    def test_3f_below_range(self):
        self.assertEqual(inferHand(FingerCount.THREE, 300), HandSize.UNKNOWN)

    def test_3f_above_range(self):
        self.assertEqual(inferHand(FingerCount.THREE, 800), HandSize.UNKNOWN)

    def test_3f_lower_boundary(self):
        self.assertEqual(inferHand(FingerCount.THREE, Config.W_IF38TR_MIN), HandSize.REGULAR)

    def test_3f_upper_boundary(self):
        self.assertEqual(inferHand(FingerCount.THREE, Config.W_30TR_MAX), HandSize.REGULAR)

    # 4/5-finger: 25BCP(621-730)→REGULAR, 30BCP(521-620)→REGULAR, 33BCP(400-520)→SMALL
    def test_4f_25bcp_regular(self):
        self.assertEqual(inferHand(FingerCount.FOUR, 680), HandSize.REGULAR)

    def test_4f_30bcp_regular(self):
        self.assertEqual(inferHand(FingerCount.FOUR, 570), HandSize.REGULAR)

    def test_4f_33bcp_small(self):
        self.assertEqual(inferHand(FingerCount.FOUR, 460), HandSize.SMALL)

    def test_5f_25bcp_regular(self):
        self.assertEqual(inferHand(FingerCount.FIVE, 700), HandSize.REGULAR)

    def test_4f_out_of_range(self):
        for w in (200, 350, 731, 900):
            with self.subTest(w=w):
                self.assertEqual(inferHand(FingerCount.FOUR, w), HandSize.UNKNOWN)

    def test_unknown_finger_always_unknown(self):
        for w in (400, 580, 700):
            with self.subTest(w=w):
                self.assertEqual(inferHand(FingerCount.UNKNOWN, w), HandSize.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════════════
class TestClassifyBanana(unittest.TestCase):
    """classifyBanana: (FingerCount, weight) → (BananaClass, HandSize)."""

    # ── All six valid grades ──────────────────────────────────────────────────
    def test_25bcp(self):
        cls, hand = classifyBanana(FingerCount.FOUR, 680)
        self.assertEqual(cls,  BananaClass.C25BCP)
        self.assertEqual(hand, HandSize.REGULAR)

    def test_30bcp(self):
        cls, hand = classifyBanana(FingerCount.FOUR, 570)
        self.assertEqual(cls,  BananaClass.C30BCP)
        self.assertEqual(hand, HandSize.REGULAR)

    def test_33bcp(self):
        cls, hand = classifyBanana(FingerCount.FOUR, 460)
        self.assertEqual(cls,  BananaClass.C33BCP)
        self.assertEqual(hand, HandSize.SMALL)

    def test_30tr(self):
        cls, hand = classifyBanana(FingerCount.THREE, 600)
        self.assertEqual(cls,  BananaClass.C30TR)
        self.assertEqual(hand, HandSize.REGULAR)

    def test_if36tr(self):
        cls, hand = classifyBanana(FingerCount.THREE, 500)
        self.assertEqual(cls,  BananaClass.CF36TR)
        self.assertEqual(hand, HandSize.REGULAR)

    def test_if38tr(self):
        cls, hand = classifyBanana(FingerCount.THREE, 400)
        self.assertEqual(cls,  BananaClass.CIF38TR)
        self.assertEqual(hand, HandSize.REGULAR)

    def test_5_finger_25bcp(self):
        cls, _ = classifyBanana(FingerCount.FIVE, 650)
        self.assertEqual(cls, BananaClass.C25BCP)

    # ── Boundary values ───────────────────────────────────────────────────────
    def test_25bcp_boundaries(self):
        for w in (Config.W_25BCP_MIN, Config.W_25BCP_MAX):
            with self.subTest(w=w):
                cls, _ = classifyBanana(FingerCount.FOUR, w)
                self.assertEqual(cls, BananaClass.C25BCP)

    def test_30bcp_boundaries(self):
        for w in (Config.W_30BCP_MIN, Config.W_30BCP_MAX):
            with self.subTest(w=w):
                cls, _ = classifyBanana(FingerCount.FOUR, w)
                self.assertEqual(cls, BananaClass.C30BCP)

    def test_33bcp_lower_boundary(self):
        cls, _ = classifyBanana(FingerCount.FOUR, Config.W_33BCP_MIN)
        self.assertEqual(cls, BananaClass.C33BCP)

    def test_if38tr_lower_boundary(self):
        cls, _ = classifyBanana(FingerCount.THREE, Config.W_IF38TR_MIN)
        self.assertEqual(cls, BananaClass.CIF38TR)

    def test_30tr_upper_boundary(self):
        cls, _ = classifyBanana(FingerCount.THREE, Config.W_30TR_MAX)
        self.assertEqual(cls, BananaClass.C30TR)

    # ── Unknown / out-of-range ────────────────────────────────────────────────
    def test_unknown_finger(self):
        cls, hand = classifyBanana(FingerCount.UNKNOWN, 500)
        self.assertEqual(cls, BananaClass.UNKNOWN)

    def test_weight_too_low(self):
        cls, _ = classifyBanana(FingerCount.FOUR, 100)
        self.assertEqual(cls, BananaClass.UNKNOWN)

    def test_weight_too_high(self):
        cls, _ = classifyBanana(FingerCount.FOUR, 1000)
        self.assertEqual(cls, BananaClass.UNKNOWN)

    def test_3f_weight_below_range(self):
        cls, _ = classifyBanana(FingerCount.THREE, 200)
        self.assertEqual(cls, BananaClass.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════════════
class TestClassToBinMapping(unittest.TestCase):
    """CLASS_TO_BIN must cover exactly 6 grades with distinct bins 1–6."""

    def test_all_six_grades_mapped(self):
        expected = {BananaClass.C33BCP, BananaClass.C25BCP, BananaClass.C30BCP,
                    BananaClass.CIF38TR, BananaClass.CF36TR, BananaClass.C30TR}
        self.assertEqual(set(CLASS_TO_BIN.keys()), expected)

    def test_bins_are_1_through_6(self):
        self.assertEqual(sorted(CLASS_TO_BIN.values()), [1, 2, 3, 4, 5, 6])

    def test_bins_are_unique(self):
        vals = list(CLASS_TO_BIN.values())
        self.assertEqual(len(vals), len(set(vals)))

    def test_unknown_not_in_map(self):
        self.assertNotIn(BananaClass.UNKNOWN, CLASS_TO_BIN)


# ══════════════════════════════════════════════════════════════════════════════
@patch("time.sleep")   # skip all sleeps so tests run in <1 s each
class TestCaptureImage(unittest.TestCase):
    """
    captureImage() with a mocked YOLO model.

    model.predict is called once per CAPTURE_FRAME (5×).
    The flush phase (FLUSH_FRAMES=5) calls get_frame_fn but NOT predict.
    """

    def test_uniform_4_finger(self, _sl):
        """5/5 frames detect 4 bananas → '4-finger', non-zero conf."""
        res = _run([4, 4, 4, 4, 4])
        self.assertEqual(res["finger"][1], "4-finger")
        self.assertGreater(res["finger"][0], 0)

    def test_uniform_3_finger(self, _sl):
        res = _run([3, 3, 3, 3, 3])
        self.assertEqual(res["finger"][1], "3-finger")

    def test_uniform_5_finger(self, _sl):
        res = _run([5, 5, 5, 5, 5])
        self.assertEqual(res["finger"][1], "5-finger")

    def test_majority_vote_wins(self, _sl):
        """3 frames → '4-finger', 2 frames → '3-finger': 4-finger wins."""
        res = _run([4, 3, 4, 3, 4])
        self.assertEqual(res["finger"][1], "4-finger")

    def test_weak_consensus_returns_invalid(self, _sl):
        """
        2×4-finger + 2×3-finger + 1 miss → no label reaches
        MIN_VALID_FRAMES (3) → 'invalid'.
        """
        res = _run([4, 4, 3, 3, 0])
        self.assertEqual(res["finger"][1], "invalid")

    def test_no_detections_returns_invalid(self, _sl):
        res = _run([0, 0, 0, 0, 0])
        self.assertEqual(res["finger"][1], "invalid")

    def test_out_of_range_mask_counts_skipped(self, _sl):
        """
        2-banana (frame 1) and 6-banana (frame 3) are outside {3,4,5} → skipped.
        Remaining 3×4-finger frames satisfy the threshold.
        """
        res = _run([2, 4, 6, 4, 4])
        self.assertEqual(res["finger"][1], "4-finger")

    def test_camera_failure_returns_invalid(self, _sl):
        """(False, None) from camera → no predict calls → 'invalid'."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_yolo_result(4)]

        def fail_frame():
            return (False, None)

        with patch.object(SortQue, "model", mock_model):
            res = captureImage(fail_frame)

        self.assertEqual(res["finger"][1], "invalid")
        mock_model.predict.assert_not_called()

    def test_image_path_set_on_success(self, _sl):
        res = _run([4, 4, 4, 4, 4])
        self.assertTrue(res["image_path"].endswith(".jpg"))

    def test_bounding_box_populated(self, _sl):
        """x2 > x1 and y2 > y1 on a successful capture."""
        res = _run([4, 4, 4, 4, 4])
        self.assertGreater(res["x2"], res["x1"])
        self.assertGreater(res["y2"], res["y1"])

    def test_confidence_matches_mock(self, _sl):
        """Returned conf == the value supplied to the mock result."""
        res = _run([3, 3, 3, 3, 3], conf=0.82)
        self.assertAlmostEqual(res["finger"][0], 0.82, places=2)

    def test_predict_called_exactly_capture_frames_times(self, _sl):
        """predict is called once per capture frame, never during flush."""
        gf, mm = _make_pipeline([4] * Config.CAPTURE_FRAMES)
        with patch.object(SortQue, "model", mm):
            captureImage(gf)
        self.assertEqual(mm.predict.call_count, Config.CAPTURE_FRAMES)


# ══════════════════════════════════════════════════════════════════════════════
class TestEndToEndClassificationPipeline(unittest.TestCase):
    """
    Smoke-tests the full Python classification path:
    captureImage result → parseFinger → classifyBanana → CLASS_TO_BIN.
    """

    @patch("time.sleep")
    def _classify(self, n_bananas, weight, _sl):
        res    = _run([n_bananas] * Config.CAPTURE_FRAMES)
        label  = res["finger"][1]
        finger = parseFinger(label)
        cls, _ = classifyBanana(finger, weight)
        return cls, CLASS_TO_BIN.get(cls)

    def test_4_finger_heavy_gives_25bcp_bin2(self):
        cls, bin_n = self._classify(4, 680)
        self.assertEqual(cls,   BananaClass.C25BCP)
        self.assertEqual(bin_n, 2)

    def test_4_finger_mid_gives_30bcp_bin3(self):
        cls, bin_n = self._classify(4, 570)
        self.assertEqual(cls,   BananaClass.C30BCP)
        self.assertEqual(bin_n, 3)

    def test_4_finger_light_gives_33bcp_bin1(self):
        cls, bin_n = self._classify(4, 460)
        self.assertEqual(cls,   BananaClass.C33BCP)
        self.assertEqual(bin_n, 1)

    def test_3_finger_heavy_gives_30tr_bin6(self):
        cls, bin_n = self._classify(3, 600)
        self.assertEqual(cls,   BananaClass.C30TR)
        self.assertEqual(bin_n, 6)

    def test_3_finger_mid_gives_if36tr_bin5(self):
        cls, bin_n = self._classify(3, 500)
        self.assertEqual(cls,   BananaClass.CF36TR)
        self.assertEqual(bin_n, 5)

    def test_3_finger_light_gives_if38tr_bin4(self):
        cls, bin_n = self._classify(3, 400)
        self.assertEqual(cls,   BananaClass.CIF38TR)
        self.assertEqual(bin_n, 4)

    def test_invalid_detection_gives_no_bin(self):
        """No detections → parseFinger('invalid') → UNKNOWN → no bin."""
        cls, bin_n = self._classify(0, 500)
        self.assertEqual(cls,  BananaClass.UNKNOWN)
        self.assertIsNone(bin_n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
