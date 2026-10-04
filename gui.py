"""FSOC Coarse-Alignment Tracker: interactive desktop application (SIH26169).

Run:  python gui.py
Every simulation parameter can be changed live from the left panel. Click the SCREEN map to move
the beacon (tests loss and re-acquisition). 'Load MP4' switches to Benchmark-2 mode (PTZ bypassed).
"""
import json
import os
import sys
import time

import cv2
import numpy as np

# Point Qt at the platform plugins shipped inside the PySide6 wheel. Without this, a machine that
# also has another Qt (e.g. a non-headless opencv-python) can fail with "Could not find the Qt
# platform plugin" on macOS/Windows. Harmless when the environment is already clean.
import PySide6  # noqa: E402
_qt_plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
if os.path.isdir(_qt_plugins):
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(_qt_plugins, "platforms"))
    os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins)
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from fsoctrack.metrics import SPEC, format_report
from fsoctrack.session import SimSession, VideoSession
from fsoctrack.sim import MOTIONS, PLATFORM_MODES, WEATHERS, Config
from fsoctrack.viz import draw_camera, draw_plot, draw_world
from fsoctrack.vision import load_model

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PANEL_W = 400          # left control panel width (px)
CAM_TITLE = "VIRTUAL CAMERA  640 × 480   ·   orange = SEARCH, green = TRACK, yellow = COAST"
MAP_TITLE = "SCREEN 2000 × 2000   ·   red = beacon, green box = camera FOV, click to move the beacon"
WORLD_W = 380          # SCREEN map size (px)

PRESETS = {
    "Custom": {},
    "PS default (clear, light noise)": dict(noise=["gaussian"], gauss_sigma=5, weather="clear", jitter_px=0,
                                            platform_mode="none", platform_px=0, n_decoys=0, turbulence=0.1),
    "Heavy sensor noise (Gauss 20 + S&P 10% + Poisson)": dict(noise=["gaussian", "saltpepper", "poisson"], gauss_sigma=20,
                                                            sp_fraction=0.10, weather="clear", jitter_px=5,
                                                            platform_mode="linear", platform_px=3, n_decoys=0),
    "Fog + all noise": dict(noise=["gaussian", "saltpepper", "poisson"], gauss_sigma=10, weather="fog", jitter_px=5,
                            platform_mode="linear", platform_px=3, n_decoys=0),
    "Rain + 4 decoy beacons": dict(noise=["gaussian"], gauss_sigma=10, weather="rain", n_decoys=4, jitter_px=5,
                                   platform_mode="linear", platform_px=3),
    "Low light": dict(noise=["gaussian", "poisson"], gauss_sigma=10, weather="lowlight", jitter_px=5, n_decoys=0),
    "STRESS (beyond spec)": dict(noise=["gaussian", "saltpepper", "poisson"], gauss_sigma=20, weather="fog", jitter_px=20,
                                 platform_mode="random", platform_px=20, pan_speed_dps=10, tilt_speed_dps=10, n_decoys=0),
}
MODES = [("AI  (robust vision + neural verifier)", "ai"), ("Classical  (robust vision, no AI)", "classical"),
         ("Naive threshold  (typical baseline)", "naive")]


def qimg(bgr):
    h, w, _ = bgr.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(bgr.data, w, h, 3 * w, QtGui.QImage.Format_BGR888).copy())


class ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(float, float)

    def mousePressEvent(self, e):
        p = e.position()
        self.clicked.emit(p.x() / max(1, self.width()), p.y() / max(1, self.height()))


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSOC Coarse-Alignment Virtual Camera Tracker  |  SIH26169  |  ISRO")
        self.model = load_model(os.path.join(ROOT, "models", "beacon_verifier.joblib"))
        self.cfg = Config(duration_s=1e9)
        self.sim = SimSession(self.cfg, self.model, "ai")
        self.video = None
        self.running = False
        self.recorder = None
        self.tick = 0
        self.t_last = time.perf_counter()
        self.loop_fps = 0.0
        self._build()
        self._sync_widgets()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(1000 / 30))
        self.render_static()

    # ================================================================== UI
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(10, 8, 10, 6)
        lay.setSpacing(14)
        self._slider_labels = []

        # ---------- left: controls (fixed width, vertical scroll only if the screen is short)
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 6, 0)
        pl.setSpacing(6)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setFixedWidth(PANEL_W)
        lay.addWidget(scroll)

        self.b_run = QtWidgets.QPushButton("▶  Start")
        self.b_run.clicked.connect(self.toggle_run)
        self.b_run.setStyleSheet("font-weight:bold")
        b_reset = QtWidgets.QPushButton("⟲  Reset")
        b_reset.clicked.connect(self.reset)
        b_occ = QtWidgets.QPushButton("☁  Occlude 1 s")
        b_occ.clicked.connect(lambda: self.sim.occlude(1.0))
        self.b_video = QtWidgets.QPushButton("🎞  Load MP4")
        self.b_video.clicked.connect(lambda: self.load_video())
        b_save = QtWidgets.QPushButton("💾  Save report")
        b_save.clicked.connect(self.save_report)
        self.b_rec = QtWidgets.QPushButton("⏺  Record MP4")
        self.b_rec.setCheckable(True)
        self.b_rec.toggled.connect(self.toggle_record)
        tips = ["Run / pause the 30 Hz loop", "Reset: new random beacon position and trajectory",
                "Hide the beacon for one second: watch COAST and re-acquisition",
                "Benchmark-2: track a video file instead of the simulator (PTZ control bypassed)",
                "Save the performance report: results/live_<time>_performance.txt / .json / _frames.csv",
                "Record this window to an MP4 in results/ (click again to stop)"]
        bar = QtWidgets.QGridLayout()
        bar.setHorizontalSpacing(6)
        bar.setVerticalSpacing(6)
        for i, (b, t) in enumerate(zip([self.b_run, b_reset, b_occ, self.b_video, b_save, self.b_rec], tips)):
            b.setMinimumHeight(30)
            b.setToolTip(t)
            bar.addWidget(b, i // 2, i % 2)
        pl.addLayout(bar)

        def group(title):
            g = QtWidgets.QGroupBox(title)
            f = QtWidgets.QFormLayout(g)
            f.setContentsMargins(10, 8, 10, 8)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(5)
            f.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            pl.addWidget(g)
            return f

        f = group("Scenario")
        self.w_preset = self._combo(PRESETS.keys())
        self.w_preset.setToolTip("One-click benchmark scenarios. Changing any setting below switches this back to Custom.")
        self.w_preset.currentTextChanged.connect(self.apply_preset)
        f.addRow("Preset", self.w_preset)
        self.w_motion = self._combo(MOTIONS)
        f.addRow("Target motion", self.w_motion)
        self.w_speed = self._dspin(0.2, 4.0, 0.1, " °/s")
        f.addRow("Target speed", self.w_speed)
        self.w_size = self._spin(5, 20, " px")
        f.addRow("Target size", self.w_size)
        self.w_decoys = self._spin(0, 6, "")
        f.addRow("Decoy beacons", self.w_decoys)
        self.w_cue = QtWidgets.QCheckBox("Telemetry (GPS) cue for acquisition")
        self.w_cue.setToolTip("Off = blind spiral search over the whole scene")
        f.addRow(self.w_cue)

        f = group("Camera")
        self.w_fov = self._dspin(1.0, 10.0, 0.5, " °")
        f.addRow("FOV (horizontal)", self.w_fov)
        self.w_pan = self._dspin(1.0, 10.0, 0.5, " °/s")
        f.addRow("Max pan speed", self.w_pan)
        self.w_tilt = self._dspin(1.0, 10.0, 0.5, " °/s")
        f.addRow("Max tilt speed", self.w_tilt)

        f = group("Image noise")
        self.w_gauss = QtWidgets.QCheckBox("Gaussian σ")
        self.w_gsig, row = self._slider(0, 20, "{}")
        f.addRow(self.w_gauss, row)
        self.w_sp = QtWidgets.QCheckBox("Salt && pepper")
        self.w_spf, row = self._slider(0, 20, "{} %")
        f.addRow(self.w_sp, row)
        self.w_poi = QtWidgets.QCheckBox("Poisson (shot noise)")
        f.addRow(self.w_poi)

        f = group("Atmosphere && platform")
        self.w_weather = self._combo(WEATHERS)
        f.addRow("Weather", self.w_weather)
        self.w_turb, row = self._slider(0, 60, "{}")
        f.addRow("Turbulence", row)
        self.w_jit, row = self._slider(0, 20, "±{} px")
        f.addRow("Camera jitter", row)
        self.w_pmode = self._combo(PLATFORM_MODES)
        f.addRow("Platform motion", self.w_pmode)
        self.w_ppx, row = self._slider(0, 20, "{} px/f")
        f.addRow("Platform speed", row)

        f = group("Detector")
        self.w_mode = self._combo([lab for lab, _ in MODES])
        f.addRow(self.w_mode)
        self.l_model = QtWidgets.QLabel()
        self.l_model.setWordWrap(True)
        self.l_model.setStyleSheet("color:#9fb3c8;font-size:11px")
        f.addRow(self.l_model)
        hint = QtWidgets.QLabel("Click anywhere on the SCREEN map to move the beacon. Every setting can be changed while it runs.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9fb3c8;font-size:11px")
        pl.addWidget(hint)
        pl.addStretch(1)

        for w in [self.w_motion, self.w_weather, self.w_pmode, self.w_mode]:
            w.currentIndexChanged.connect(self.on_change)
        for w in [self.w_speed, self.w_fov, self.w_pan, self.w_tilt, self.w_size, self.w_decoys]:
            w.valueChanged.connect(self.on_change)
        for w in [self.w_gsig, self.w_spf, self.w_turb, self.w_jit, self.w_ppx]:
            w.valueChanged.connect(self.on_change)
        for w in [self.w_gauss, self.w_sp, self.w_poi, self.w_cue]:
            w.toggled.connect(self.on_change)

        # ---------- centre: camera + plot
        mid = QtWidgets.QVBoxLayout()
        mid.setSpacing(6)
        self.l_title = QtWidgets.QLabel(CAM_TITLE)
        self.l_title.setStyleSheet("font-weight:bold;color:#e6edf3")
        mid.addWidget(self.l_title)
        self.l_cam = QtWidgets.QLabel()
        self.l_cam.setFixedSize(640, 480)
        mid.addWidget(self.l_cam)
        self.l_plot = QtWidgets.QLabel()
        self.l_plot.setFixedSize(640, 150)
        mid.addWidget(self.l_plot)
        mid.addStretch(1)
        lay.addLayout(mid)

        # ---------- right: world map + stats (takes whatever width is left)
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        self.l_wtitle = QtWidgets.QLabel(MAP_TITLE)
        self.l_wtitle.setStyleSheet("font-weight:bold;color:#e6edf3")
        self.l_wtitle.setWordWrap(True)
        right.addWidget(self.l_wtitle)
        self.l_world = ClickLabel()
        self.l_world.setFixedSize(WORLD_W, WORLD_W)
        self.l_world.setCursor(QtCore.Qt.CrossCursor)
        self.l_world.clicked.connect(self.on_world_click)
        right.addWidget(self.l_world, 0, QtCore.Qt.AlignLeft)
        self.t_stats = QtWidgets.QTableWidget(0, 3)
        self.t_stats.setHorizontalHeaderLabels(["Metric", "Value", "PS spec"])
        self.t_stats.verticalHeader().setVisible(False)
        self.t_stats.verticalHeader().setDefaultSectionSize(24)
        hh = self.t_stats.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.t_stats.setColumnWidth(1, 110)
        self.t_stats.setColumnWidth(2, 90)
        self.t_stats.setShowGrid(False)
        self.t_stats.setAlternatingRowColors(True)
        self.t_stats.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.t_stats.setFocusPolicy(QtCore.Qt.NoFocus)
        self.t_stats.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.t_stats.setMinimumWidth(WORLD_W)
        self.t_stats.setToolTip("Cumulative since the last Reset. Green = within the PS limit, red = outside it.")
        right.addWidget(self.t_stats, 1)
        lay.addLayout(right, 1)
        self.statusBar().showMessage("Ready. Press Start.")

    def _combo(self, items):
        w = QtWidgets.QComboBox()
        w.addItems(list(items))
        # keep the long preset / detector names from forcing the whole panel wider
        w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(12)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        return w

    def _spin(self, a, b, suf):
        w = QtWidgets.QSpinBox()
        w.setRange(a, b)
        w.setSuffix(suf)
        w.setMaximumWidth(130)
        return w

    def _dspin(self, a, b, st, suf):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(a, b)
        w.setSingleStep(st)
        w.setSuffix(suf)
        w.setMaximumWidth(130)
        return w

    def _slider(self, a, b, fmt):
        """Slider with its current value shown beside it. Returns (slider, row_widget)."""
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(a, b)
        s.setMinimumWidth(80)
        lab = QtWidgets.QLabel()
        lab.setMinimumWidth(58)
        lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lab.setStyleSheet("color:#9fb3c8")
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(s, 1)
        h.addWidget(lab)
        s.valueChanged.connect(lambda v, l=lab, f=fmt: l.setText(f.format(v)))
        lab.setText(fmt.format(s.value()))
        self._slider_labels.append((s, lab, fmt))
        return s, row

    def _refresh_slider_labels(self):
        for s, lab, fmt in self._slider_labels:
            lab.setText(fmt.format(s.value()))

    # ================================================================== parameters
    def _sync_widgets(self):
        """cfg -> widgets (without triggering change handlers)."""
        c = self.cfg
        ws = [self.w_motion, self.w_speed, self.w_size, self.w_decoys, self.w_cue, self.w_fov, self.w_pan, self.w_tilt,
              self.w_gauss, self.w_gsig, self.w_sp, self.w_spf, self.w_poi, self.w_weather, self.w_turb, self.w_jit,
              self.w_pmode, self.w_ppx, self.w_mode]
        for w in ws:
            w.blockSignals(True)
        self.w_motion.setCurrentText(c.motion)
        self.w_speed.setValue(c.target_speed_dps)
        self.w_size.setValue(c.target_size)
        self.w_decoys.setValue(c.n_decoys)
        self.w_cue.setChecked(c.cued)
        self.w_fov.setValue(c.fov_x_deg)
        self.w_pan.setValue(c.pan_speed_dps)
        self.w_tilt.setValue(c.tilt_speed_dps)
        self.w_gauss.setChecked("gaussian" in c.noise)
        self.w_gsig.setValue(int(c.gauss_sigma))
        self.w_sp.setChecked("saltpepper" in c.noise)
        self.w_spf.setValue(int(round(c.sp_fraction * 100)))
        self.w_poi.setChecked("poisson" in c.noise)
        self.w_weather.setCurrentText(c.weather)
        self.w_turb.setValue(int(c.turbulence * 100))
        self.w_jit.setValue(int(c.jitter_px))
        self.w_pmode.setCurrentText(c.platform_mode)
        self.w_ppx.setValue(int(c.platform_px))
        self.w_mode.setCurrentIndex([m for _, m in MODES].index(self.sim.mode))
        for w in ws:
            w.blockSignals(False)
        self._refresh_slider_labels()
        self._model_label()

    def _model_label(self):
        p = os.path.join(ROOT, "models", "training_report.json")
        if self.model is None:
            self.l_model.setText("⚠ AI model not trained yet: run  python train.py  (falls back to Classical).")
        elif os.path.exists(p):
            r = json.load(open(p))
            self.l_model.setText(f"AI verifier loaded: MLP trained on {r['samples']:,} candidates, "
                                 f"held-out accuracy {100 * r['accuracy']:.1f}%, beacon recall {100 * r['recall']:.1f}%.")
        else:
            self.l_model.setText("AI verifier loaded.")

    def on_change(self, *_):
        c = self.cfg
        c.motion = self.w_motion.currentText()
        c.target_speed_dps = self.w_speed.value()
        c.target_size = self.w_size.value()
        c.n_decoys = self.w_decoys.value()
        c.cued = self.w_cue.isChecked()
        c.fov_x_deg = self.w_fov.value()
        c.pan_speed_dps = self.w_pan.value()
        c.tilt_speed_dps = self.w_tilt.value()
        c.noise = [n for n, w in [("gaussian", self.w_gauss), ("saltpepper", self.w_sp), ("poisson", self.w_poi)] if w.isChecked()]
        c.gauss_sigma = float(self.w_gsig.value())
        c.sp_fraction = self.w_spf.value() / 100.0
        c.weather = self.w_weather.currentText()
        c.turbulence = self.w_turb.value() / 100.0
        c.jitter_px = float(self.w_jit.value())
        c.platform_mode = self.w_pmode.currentText()
        c.platform_px = float(self.w_ppx.value())
        mode = MODES[self.w_mode.currentIndex()][1]
        if mode == "ai" and self.model is None:
            mode = "classical"
            self.statusBar().showMessage("AI model missing: run python train.py first. Using Classical.")
        if mode != self.sim.mode and not self.video:
            self.sim.restart_stats()
            self.statusBar().showMessage(f"Detector switched to {mode.upper()}. Statistics restarted.")
        self.sim.mode = mode
        if self.video:
            self.video.det.mode = mode
        self.sim.apply()
        if self.sender() is not self.w_preset:
            self.w_preset.blockSignals(True)
            self.w_preset.setCurrentText("Custom")
            self.w_preset.blockSignals(False)
        if not self.running:
            self.render_static()

    def apply_preset(self, name):
        for k, v in PRESETS.get(name, {}).items():
            setattr(self.cfg, k, list(v) if isinstance(v, list) else v)
        if name.startswith("STRESS") is False and name != "Custom":
            self.cfg.pan_speed_dps = self.cfg.tilt_speed_dps = 5.0
        self._sync_widgets()
        self.sim.apply()
        if name != "Custom" and not self.video:
            self.sim.restart_stats()
            self.statusBar().showMessage(f"Preset applied: {name}. Statistics restarted for this scenario.")
        else:
            self.statusBar().showMessage(f"Preset applied: {name}")

    # ================================================================== actions
    def toggle_run(self):
        self.running = not self.running
        self.b_run.setText("⏸  Pause" if self.running else "▶  Start")

    def reset(self):
        if self.video:
            self.video = None
            self.cfg.fps = 30.0
            self.l_title.setText(CAM_TITLE)
            self.l_wtitle.setText(MAP_TITLE)
        self.sim.reset(new_seed=True)
        self.sim.apply()
        self.statusBar().showMessage("Reset: new random target position and trajectory.")
        self.render_static()

    def on_world_click(self, fx, fy):
        if self.video:
            return
        self.sim.teleport((fx * self.cfg.world_w, fy * self.cfg.world_h))
        self.statusBar().showMessage("Beacon moved: watch COAST → SEARCH → re-acquisition.")

    def load_video(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open benchmark video", ROOT, "Video (*.mp4 *.avi *.mov)")
        if not path:
            return
        try:
            self.video = VideoSession(path, Config(target_size=self.cfg.target_size), self.model, self.sim.mode)
        except Exception as e:  # noqa
            QtWidgets.QMessageBox.warning(self, "Video", str(e))
            return
        g = self.video.next
        self.l_title.setText(f"BENCHMARK-2: {os.path.basename(path)}  ({g.shape[1]}×{g.shape[0]}, PTZ bypassed, digital 640×480 window)")
        self.l_wtitle.setText("FULL VIDEO FRAME  (green = tracked beacon)")
        tr = "with truth CSV: errors are scored" if self.video.truth is not None else "no truth CSV: centroids logged only"
        self.statusBar().showMessage(f"Video loaded ({tr}). Press Start. Press Reset to return to simulation.")
        self.running = False
        self.toggle_run()

    def save_report(self, silent=False):
        rec = self.video.rec if self.video else self.sim.rec
        if len(rec.rows) < 5:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{rec.label}_{stamp}"
        s = rec.write(RESULTS, prefix)
        if not silent:
            QtWidgets.QMessageBox.information(self, "Performance report saved",
                                              f"Saved to results/{prefix}_performance.txt / .json / _frames.csv\n\n" + format_report(s))

    def toggle_record(self, on):
        if on:
            os.makedirs(RESULTS, exist_ok=True)
            path = os.path.join(RESULTS, f"screen_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            sz = self.centralWidget().size()
            self.rec_size = (sz.width() // 2 * 2, sz.height() // 2 * 2)
            self.recorder = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, self.rec_size)
            self.statusBar().showMessage(f"Recording to {path}")
        else:
            if self.recorder:
                self.recorder.release()
            self.recorder = None
            self.statusBar().showMessage("Recording saved in results/.")

    # ================================================================== loop
    def on_tick(self):
        if not self.running:
            return
        now = time.perf_counter()
        self.loop_fps = 0.9 * self.loop_fps + 0.1 / max(now - self.t_last, 1e-6)
        self.t_last = now
        if self.video:
            r = self.video.step()
            if r is None:
                self.toggle_run()
                self.save_report()
                return
            self._render_video(r)
        else:
            r = self.sim.step()
            self._render_sim(r)
        self.tick += 1
        if self.tick % 10 == 0:
            self._update_stats(r)
        if self.recorder is not None:
            img = self.centralWidget().grab().toImage().convertToFormat(QtGui.QImage.Format_BGR888)
            w, h = self.rec_size
            img = img.scaled(w, h)
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
            self.recorder.write(arr.copy())

    def _render_sim(self, r):
        self.l_cam.setPixmap(qimg(draw_camera(r["frame"], r["out"], self.cfg.target_size, r["occluded"])))
        self.l_world.setPixmap(qimg(draw_world(self.cfg, self.sim.trail, r["tgt"], r["cam"])))
        self.l_plot.setPixmap(qimg(draw_plot(self.sim.err)))

    def _render_video(self, r):
        g, out = r["frame"], r["out"]
        m = out["meas_img"]
        H, W = g.shape
        cx, cy = (m if m is not None else (W / 2, H / 2))
        x0 = int(np.clip(cx - 320, 0, max(0, W - 640)))
        y0 = int(np.clip(cy - 240, 0, max(0, H - 480)))
        view = g[y0:y0 + 480, x0:x0 + 640]
        if view.shape != (480, 640):
            view = cv2.resize(view, (640, 480))
            x0 = y0 = 0
        self.l_cam.setPixmap(qimg(draw_camera(view, out, self.cfg.target_size, False, offset=(x0, y0))))
        thumb = cv2.cvtColor(cv2.resize(g, (WORLD_W, WORLD_W)), cv2.COLOR_GRAY2BGR)
        sx, sy = 380 / W, 380 / H
        for i in range(1, len(self.video.trail)):
            a, b = self.video.trail[i - 1], self.video.trail[i]
            cv2.line(thumb, (int(a[0] * sx), int(a[1] * sy)), (int(b[0] * sx), int(b[1] * sy)), (80, 220, 80), 1)
        if m is not None:
            cv2.circle(thumb, (int(m[0] * sx), int(m[1] * sy)), 5, (80, 220, 80), 2)
        self.l_world.setPixmap(qimg(thumb))
        self.l_plot.setPixmap(qimg(draw_plot(self.video.err, label="centroid error vs truth (px)")))

    def render_static(self):
        if self.video:
            return
        r = self.sim.step() if self.sim.k == 0 else None
        if r:
            self._render_sim(r)
            self._update_stats(r)

    def _update_stats(self, r):
        rec = self.video.rec if self.video else self.sim.rec
        s = rec.summary()
        out = r["out"]
        now = [("State", out["state"], ""),
               ("Pointing error (now)", "-" if r.get("perr") is None else f"{r['perr']:.2f} px", "≤ 10 px"),
               ("Centroid error (now)", "-" if r.get("cerr") is None else f"{r['cerr']:.2f} px", "minimise"),
               ("AI beacon confidence", f"{out['prob']:.2f}", "")]
        spec = {"acquisition_time_s": "≤ 2 s", "pointing_err_mean_px": "≤ 10 px", "centroid_err_rmse_px": "minimise",
                "target_loss_pct": "< 5 %", "reacquisition_max_s": "≤ 1 s", "processing_fps": "≥ 20"}
        rows = now
        t_sim = s["sim_duration_s"]
        for k, lab, _, ok in SPEC:
            v = s.get(k)
            txt, good = ("-" if v is None else f"{v}"), ok(v)
            if k == "acquisition_time_s" and v is None:
                txt, good = ("NOT ACQUIRED" if t_sim > 2 else "searching…"), t_sim <= 2
            elif k == "target_loss_pct" and v is not None and s.get("acquisition_time_s") is None:
                txt = f"{v}  (never locked)"
            elif k == "reacquisition_max_s":
                pend, nev = s.get("reacquisition_pending_s"), s.get("reacquisition_events", 0)
                if pend is not None:
                    txt, good = f"LOST for {pend:.1f} s", pend <= 1
                elif v is None:
                    txt, good = "no loss yet", True
                else:
                    txt = f"{v}  (worst of {nev})"
            rows.append((lab, txt, spec[k], good))
        rows += [("RMSE tracking error (px)", str(s.get("pointing_err_rmse_px", "-")), ""),
                 ("Lock retention (%)", str(s.get("lock_retention_pct", "-")), ""),
                 ("Frames within 10 px (%)", str(s.get("pointing_within_10px_pct", "-")), ""),
                 ("Mean processing time", f"{s['proc_time_mean_ms']} ms", ""),
                 ("Display loop FPS", f"{self.loop_fps:.1f}", "≥ 20"),
                 ("Simulated time", f"{s['sim_duration_s']} s", "")]
        self.t_stats.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j in range(3):
                it = QtWidgets.QTableWidgetItem(str(row[j]))
                if j == 1 and len(row) > 3:
                    it.setForeground(QtGui.QColor("#50dc50" if row[3] else "#ff7a59"))
                self.t_stats.setItem(i, j, it)

    def closeEvent(self, e):
        if self.recorder:
            self.recorder.release()
        self.save_report(silent=True)
        super().closeEvent(e)


STYLE = """
QGroupBox { border: 1px solid #263241; border-radius: 6px; margin-top: 14px; padding-top: 4px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fd3d8; }
QPushButton { padding: 4px 8px; border-radius: 4px; }
QPushButton:checked { background: #7a1f1f; }
QTableWidget { border: 1px solid #263241; gridline-color: #263241; }
QHeaderView::section { background: #1f2937; color: #e6edf3; padding: 4px; border: none; }
QToolTip { color: #e6edf3; background: #1f2937; border: 1px solid #39506b; }
"""


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in [(QtGui.QPalette.Window, "#121820"), (QtGui.QPalette.WindowText, "#e6edf3"), (QtGui.QPalette.Base, "#0d1117"),
                      (QtGui.QPalette.AlternateBase, "#161b22"), (QtGui.QPalette.Text, "#e6edf3"), (QtGui.QPalette.Button, "#1f2937"),
                      (QtGui.QPalette.ButtonText, "#e6edf3"), (QtGui.QPalette.Highlight, "#1b998b")]:
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    w = Main()
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
