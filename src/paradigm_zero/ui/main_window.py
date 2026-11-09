import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QComboBox, QCheckBox, QSlider, QTabWidget, QVBoxLayout, QWidget, QStatusBar,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa

from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.symbolic_regression import SymbolicRegressor
from paradigm_zero.core.testing import HypothesisTester
from paradigm_zero.core.benchmark import Benchmark

DARK, PANEL, CARD, ACCENT, CYAN, GREEN, AMBER, RED = (
    "#0b0f1a", "#111827", "#0f172a", "#7c5cff", "#22d3ee", "#34d399", "#fbbf24", "#f87171",
)

APP_QSS = f"""
QMainWindow {{ background: {DARK}; }}
QTabWidget::pane {{ background: {PANEL}; border: 1px solid #1f2937; border-radius: 10px; }}
QTabBar::tab {{ background: #1f2937; color: #e5e7eb; padding: 9px 18px; border-top-left-radius: 10px; border-top-right-radius: 10px; font-weight: bold; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: white; }}
QTabBar::tab:hover {{ background: #2b3a55; }}
QLabel {{ color: #e5e7eb; font-size: 11pt; }}
QPushButton {{ background: {ACCENT}; color: white; border-radius: 9px; padding: 8px 16px; font-weight: bold; font-size: 10.5pt; }}
QPushButton:hover {{ background: #8b6dff; }}
QPushButton:pressed {{ background: #6d4ce8; }}
QPushButton#ghost {{ background: #1f2937; color: #e5e7eb; }}
QPushButton#cyan {{ background: {CYAN}; color: #04222a; }}
QLineEdit, QTextEdit, QTableWidget, QSpinBox, QDoubleSpinBox {{ background: {CARD}; color: #e5e7eb; border: 1px solid #263248; border-radius: 8px; padding: 5px; selection-background-color: {ACCENT}; }}
QComboBox {{ background: #1f2937; color: #e5e7eb; padding: 5px; border-radius: 8px; border: 1px solid #263248; }}
QCheckBox {{ color: #cbd5e1; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #475569; background: #1f2937; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QSlider::groove:horizontal {{ height: 6px; background: #1f2937; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; background: {CYAN}; border-radius: 8px; margin: -5px 0; }}
QHeaderView::section {{ background: #1f2937; color: #e5e7eb; padding: 8px; border: none; font-weight: bold; }}
QTableWidget {{ gridline-color: #1f2937; }}
QScrollBar:vertical {{ background: {DARK}; width: 12px; }}
QScrollBar::handle:vertical {{ background: #334155; border-radius: 6px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QStatusBar {{ background: #0f172a; color: #94a3b8; }}
"""


def card(title, subtitle=""):
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background: {CARD}; border: 1px solid #263248; border-radius: 12px; }}")
    v = QVBoxLayout(f)
    v.setContentsMargins(14, 12, 14, 12)
    v.setSpacing(8)
    h = QLabel(f"<b style='font-size:13pt;color:white'>{title}</b>")
    v.addWidget(h)
    if subtitle:
        s = QLabel(f"<span style='color:#94a3b8'>{subtitle}</span>")
        s.setWordWrap(True)
        v.addWidget(s)
    return f, v


def header(total, section):
    return QLabel(f"<b style='color:white;font-size:15pt'>PARADIGM ZERO</b>  "
                  f"<span style='color:{CYAN}'>❖</span>  <span style='color:#94a3b8'>{section}</span>")


def badge(text, color):
    l = QLabel(f"<b>{text}</b>")
    l.setStyleSheet(f"background: {color}22; color: {color}; border: 1px solid {color}88; border-radius: 10px; padding: 4px 10px;")
    return l


def r2_color(v):
    try:
        v = float(v)
    except Exception:
        return "#94a3b8"
    return GREEN if v >= 0.9 else AMBER if v >= 0.5 else RED


class UniverseView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.addWidget(header(self, "3D Scientific Universe"))
        ctrl, cv = card("Universe Configuration", "Hidden law, noise and sample budget. Observations omit hidden variables.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Hidden equation:"))
        self.eq_edit = QLineEdit("x1**2 + sin(x2) + 0.5*x1*x2"); row.addWidget(self.eq_edit)
        cv.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Noise:")); self.noise = QDoubleSpinBox(); self.noise.setValue(0.1); self.noise.setSingleStep(0.05); row2.addWidget(self.noise)
        row2.addWidget(QLabel("Samples:")); self.n = QSpinBox(); self.n.setValue(300); self.n.setRange(50, 5000); row2.addWidget(self.n)
        self.gen_btn = QPushButton("✦ Generate Universe"); row2.addWidget(self.gen_btn)
        self.status_badge = badge("IDLE", "#94a3b8"); row2.addWidget(self.status_badge)
        cv.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Colormap:")); self.cmap = QComboBox()
        self.cmap.addItems(["plasma", "viridis", "inferno", "turbo", "ocean"]); row3.addWidget(self.cmap)
        self.surface_chk = QCheckBox("Ground-truth surface"); self.surface_chk.setChecked(True); row3.addWidget(self.surface_chk)
        self.points_chk = QCheckBox("Observations"); self.points_chk.setChecked(True); row3.addWidget(self.points_chk)
        self.wire_chk = QCheckBox("Wireframe"); self.wire_chk.setChecked(True); row3.addWidget(self.wire_chk)
        self.rot_chk = QCheckBox("Auto-rotate"); self.rot_chk.setChecked(True); row3.addWidget(self.rot_chk)
        row3.addWidget(QLabel("Speed:")); self.speed = QSlider(Qt.Horizontal); self.speed.setRange(1, 10); self.speed.setValue(3); row3.addWidget(self.speed)
        row3.addWidget(QLabel("Quality:")); self.quality = QComboBox(); self.quality.addItems(["Ultra", "High", "Balanced", "Performance"]); row3.addWidget(self.quality)
        cv.addLayout(row3)
        root.addWidget(ctrl)
        self.figure = Figure(figsize=(9, 6), dpi=150, facecolor=DARK)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background: {DARK}; border: 1px solid #263248; border-radius: 12px;")
        root.addWidget(self.canvas)
        self.universe = self.train_data = self.test_data = None
        self.azim = 45.0
        self.gen_btn.clicked.connect(self.generate)
        for w in (self.cmap, self.surface_chk, self.points_chk, self.wire_chk, self.quality):
            (w.stateChanged if hasattr(w, "stateChanged") else w.currentIndexChanged).connect(lambda *a: self.plot())
        self.timer = QTimer(self); self.timer.timeout.connect(self._rotate); self.timer.start(60)

    def _rotate(self):
        if self.rot_chk.isChecked():
            self.azim = (self.azim + self.speed.value() * 0.6) % 360
            try:
                self.figure.axes[0].view_init(elev=28, azim=self.azim); self.canvas.draw_idle()
            except Exception:
                pass

    def generate(self):
        cfg = UniverseConfig(hidden_equation=self.eq_edit.text(), noise=self.noise.value(), n_samples=self.n.value())
        self.universe = Universe(cfg)
        self.train_data = self.universe.generate()
        self.test_data = self.universe.generate(n=100, seed=cfg.test_seed)
        self.status_badge.setText("● LIVE — SIMULATED UNIVERSE")
        self.status_badge.setStyleSheet(f"background: {GREEN}22; color: {GREEN}; border: 1px solid {GREEN}88; border-radius: 10px; padding: 4px 10px;")
        self.plot()

    def _grid_n(self):
        return {"Ultra": 90, "High": 64, "Balanced": 40, "Performance": 24}[self.quality.currentText()]

    def plot(self):
        self.figure.clear(); self.figure.set_facecolor(DARK)
        ax = self.figure.add_subplot(111, projection="3d"); ax.set_facecolor(DARK)
        for p in ("xaxis", "yaxis", "zaxis"):
            getattr(ax, p).pane.set_facecolor((0.05, 0.07, 0.12, 1)); getattr(ax, p).pane.set_edgecolor("#263248")
        ax.tick_params(colors="#cbd5e1", labelsize=8); ax.grid(True, color="#263248", alpha=0.6)
        if self.train_data is None:
            ax.set_title("PARADIGM ZERO — 3D Scientific Universe", color="white", fontsize=13, pad=14)
            ax.text2D(0.5, 0.5, "Generate a universe to begin", color="#94a3b8", ha="center"); self.canvas.draw(); return
        cmap = self.cmap.currentText()
        x1, x2, y = self.train_data["x1"], self.train_data["x2"], self.train_data["y"]
        zmin, zmax = float(np.min(y)), float(np.max(y))
        if self.surface_chk.isChecked():
            g = self._grid_n()
            gx, gy = np.meshgrid(np.linspace(float(np.min(x1)), float(np.max(x1)), g),
                                 np.linspace(float(np.min(x2)), float(np.max(x2)), g))
            gz = self.universe.evaluate(self.universe.ground_truth_str(), {"x1": gx.ravel(), "x2": gy.ravel()}).reshape(gx.shape)
            zs = gz - np.ptp(gz) * 0.06
            surf = ax.plot_surface(gx, gy, zs, cmap=cmap, alpha=0.82, linewidth=0, antialiased=True, shade=True, vmin=zmin, vmax=zmax)
            if self.wire_chk.isChecked():
                ax.plot_wireframe(gx, gy, zs, color="#e2e8f0", alpha=0.10, linewidths=0.4)
            try:
                ax.contourf(gx, gy, zs, zdir="z", offset=zmin - np.ptp(gz) * 0.10, cmap=cmap, alpha=0.35)
            except Exception:
                pass
            self.figure.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="y (simulated)")
        if self.points_chk.isChecked():
            ax.scatter(x1, x2, y, c=y, cmap=cmap, s=26, alpha=0.95, edgecolors="#0b0f1a", linewidths=0.5, depthshade=True)
            ax.scatter(x1[:1], x2[:1], y[:1], s=90, c="#22d3ee", edgecolors="white", linewidths=1.2, depthshade=False)
        ax.set_xlabel("x1 — observed", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_ylabel("x2 — observed", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_zlabel("y — simulated response", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_title(f"PARADIGM ZERO — 3D Scientific Universe  •  simulated law: {self.universe.ground_truth_str()}",
                     color="white", fontsize=11, pad=14)
        ax.view_init(elev=28, azim=self.azim)
        ax.text2D(0.02, 0.95, f"n={len(y)}  noise={self.noise.value():.2f}  cmap={cmap}  grid={self._grid_n()}²",
                  color="#94a3b8", fontsize=8, transform=ax.transAxes,
                  bbox=dict(facecolor="#111827", alpha=0.85, edgecolor="#263248", boxstyle="round,pad=0.4"))
        self.canvas.draw()


class EquationEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.addWidget(header(self, "Equation Editor"))
        c, v = card("Candidate Equation", "SymPy syntax — e.g. x1**2 + sin(x2). Live validation with a 2D slice preview.")
        self.edit = QTextEdit("x1**2 + sin(x2) + x1*x2"); self.edit.setMaximumHeight(90); v.addWidget(self.edit)
        row = QHBoxLayout()
        self.validate_btn = QPushButton("✓ Validate"); row.addWidget(self.validate_btn)
        self.load_btn = QPushButton("⇪ Load Best Hypothesis"); self.load_btn.setObjectName("cyan"); row.addWidget(self.load_btn)
        self.clear_btn = QPushButton("Clear"); self.clear_btn.setObjectName("ghost"); row.addWidget(self.clear_btn)
        self.status = badge("AWAITING INPUT", "#94a3b8"); row.addWidget(self.status); v.addLayout(row)
        self.figure = Figure(figsize=(8, 2.6), dpi=130, facecolor=DARK)
        self.canvas = FigureCanvas(self.figure); v.addWidget(self.canvas)
        root.addWidget(c)
        self.edit.textChanged.connect(self.validate)
        self.validate_btn.clicked.connect(self.validate)
        self.clear_btn.clicked.connect(lambda: self.edit.clear())
        self.validate()

    def validate(self, *a):
        import sympy as sp
        expr = self.edit.toPlainText().strip()
        try:
            if not expr:
                raise ValueError("empty")
            sp.sympify(expr)
            self.status.setText("✓ VALID EXPRESSION"); self.status.setStyleSheet(f"background: {GREEN}22; color: {GREEN}; border: 1px solid {GREEN}88; border-radius: 10px; padding: 4px 10px;")
            self._preview(expr)
        except Exception as e:
            self.status.setText(f"✕ INVALID: {e}")
            self.status.setStyleSheet(f"background: {RED}22; color: {RED}; border: 1px solid {RED}88; border-radius: 10px; padding: 4px 10px;")
            self.figure.clear(); self.figure.set_facecolor(DARK)
            ax = self.figure.add_subplot(111); ax.set_facecolor(DARK)
            ax.text(0.5, 0.5, "Fix the expression to preview", color="#94a3b8", ha="center"); self.canvas.draw()

    def _preview(self, expr):
        try:
            xs = np.linspace(-3, 3, 300)
            data = {"x1": xs, "x2": np.zeros_like(xs)}
            import sympy as sp
            f = sp.lambdify([sp.symbols("x1"), sp.symbols("x2")], sp.sympify(expr), "numpy")
            ys = np.asarray(f(xs, np.zeros_like(xs)), dtype=float)
            self.figure.clear(); self.figure.set_facecolor(DARK)
            ax = self.figure.add_subplot(111); ax.set_facecolor(DARK)
            ax.plot(xs, ys, color=ACCENT, lw=2.5); ax.fill_between(xs, ys, color=CYAN, alpha=0.18)
            ax.set_title("Slice preview: y(x1, x2=0)", color="white"); ax.tick_params(colors="#cbd5e1")
            for s in ax.spines.values(): s.set_color("#263248")
            self.canvas.draw()
        except Exception:
            pass


class HypothesisExplorer(QWidget):
    def __init__(self, universe_view, parent=None):
        super().__init__(parent)
        self.uv = universe_view
        root = QVBoxLayout(self); root.addWidget(header(self, "Hypothesis Explorer"))
        c, v = card("Competing Explanations", "Symbolic regression proposes candidates; each is tested on train and unseen data. Green ≥ 0.9 · Amber ≥ 0.5 · Red below.")
        row = QHBoxLayout()
        self.run_btn = QPushButton("✦ Run Discovery"); row.addWidget(self.run_btn)
        self.summary = QLabel("<span style='color:#94a3b8'>No hypotheses yet — generate a universe, then run discovery.</span>")
        row.addWidget(self.summary); v.addLayout(row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Expression", "Train R²", "Test R²", "Valid"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)
        root.addWidget(c)

    def run(self):
        uv = self.uv
        if uv.universe is None:
            uv.generate()
        X = {k: uv.train_data[k] for k in uv.universe.config.observed_vars}
        reg = SymbolicRegressor(uv.universe.config.observed_vars)
        cands = reg.fit(X, uv.train_data["y"])
        tester = HypothesisTester(uv.universe)
        scored = []
        for cand in cands:
            tr, te = tester.test(cand, uv.train_data), tester.test(cand, uv.test_data)
            scored.append((cand, tr, te))
        scored.sort(key=lambda s: s[2]["r2"], reverse=True)
        self.table.setRowCount(0)
        for i, (cand, tr, te) in enumerate(scored, 1):
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(f"#{i}"))
            self.table.setItem(r, 1, QTableWidgetItem(cand))
            for j, val in ((2, tr["r2"]), (3, te["r2"])):
                it = QTableWidgetItem(f"{val:.4f}")
                it.setForeground(QColor(r2_color(val))); it.setTextAlignment(Qt.AlignCenter); self.table.setItem(r, j, it)
            ok = QTableWidgetItem("✓" if tr["valid"] else "✕")
            ok.setForeground(QColor(GREEN if tr["valid"] else RED)); self.table.setItem(r, 4, ok)
        best = scored[0]
        self.summary.setText(f"Best: <b style='color:{r2_color(best[2]['r2'])}'>Test R² = {best[2]['r2']:.4f}</b>"
                             f"  •  {best[0]}  •  {len(scored)} candidates ranked")


class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.addWidget(header(self, "Experiment Timeline"))
        c, v = card("Chronological Experiment Log", "Every generation, discovery and benchmark is timestamped.")
        row = QHBoxLayout()
        self.count = badge("0 EVENTS", CYAN); row.addWidget(self.count)
        row.addStretch()
        self.clear_btn = QPushButton("Clear"); self.clear_btn.setObjectName("ghost"); row.addWidget(self.clear_btn)
        v.addLayout(row)
        self.listw = QTextEdit(); self.listw.setReadOnly(True); v.addWidget(self.listw)
        root.addWidget(c)
        self.n_events = 0
        self.clear_btn.clicked.connect(lambda: (self.listw.clear(), self._reset()))

    def _reset(self):
        self.n_events = 0; self.count.setText("0 EVENTS")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = "✓" if any(k in msg for k in ("Generated", "Benchmark", "Discovery", "completed")) else "•"
        self.listw.append(f"<span style='color:#64748b'>[{ts}]</span> <span style='color:{GREEN if tag=='✓' else CYAN}'><b>{tag}</b></span> <span style='color:#e5e7eb'>{msg}</span>")
        self.n_events += 1; self.count.setText(f"{self.n_events} EVENTS")


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.addWidget(header(self, "Results Dashboard"))
        self.cards = QHBoxLayout()
        self.metric_cards = {}
        for name in ("Equation Recovery", "Prediction Accuracy R²", "Generalization Gap", "Candidates Tested"):
            f, fv = card(name, "")
            big = QLabel("<b style='font-size:20pt;color:white'>—</b>"); fv.addWidget(big)
            self.metric_cards[name] = big; self.cards.addWidget(f)
        root.addLayout(self.cards)
        c, v = card("Benchmark Visuals", "Left: metric bars. Right: simulated truth vs prediction (when available).")
        split = QHBoxLayout()
        self.fig1 = Figure(figsize=(5, 3.2), dpi=130, facecolor=DARK)
        self.canvas1 = FigureCanvas(self.fig1); split.addWidget(self.canvas1)
        self.fig2 = Figure(figsize=(5, 3.2), dpi=130, facecolor=DARK)
        self.canvas2 = FigureCanvas(self.fig2); split.addWidget(self.canvas2)
        v.addLayout(split); root.addWidget(c)
        self.text = QTextEdit(); self.text.setReadOnly(True); self.text.setMaximumHeight(110)
        self.text.append("Run a benchmark to populate the dashboard.")
        root.addWidget(self.text)
        self._empty_charts()

    def _empty_charts(self):
        for fig, can, title in ((self.fig1, self.canvas1, "Benchmark metrics"), (self.fig2, self.canvas2, "True vs predicted")):
            fig.clear(); fig.set_facecolor(DARK)
            ax = fig.add_subplot(111); ax.set_facecolor(DARK)
            ax.set_title(title, color="white"); ax.text(0.5, 0.5, "No data yet", color="#94a3b8", ha="center")
            ax.tick_params(colors="#cbd5e1")
            for s in ax.spines.values(): s.set_color("#263248")
            can.draw()

    def update_metrics(self, metrics):
        for k, val in metrics.items():
            self.text.append(f"<b style='color:{CYAN}'>{k}:</b> {val}")
        if "equation_recovery" in metrics:
            self.metric_cards["Equation Recovery"].setText(str(metrics["equation_recovery"]))
        if "prediction_accuracy_r2" in metrics:
            v = metrics["prediction_accuracy_r2"]
            self.metric_cards["Prediction Accuracy R²"].setText(f"{v:.4f}")
            self.metric_cards["Prediction Accuracy R²"].setStyleSheet(f"color: {r2_color(v)};" if isinstance(v, float) else "")
        if "generalization_gap" in metrics:
            self.metric_cards["Generalization Gap"].setText(f"{metrics['generalization_gap']:.4f}")
        keys = [k for k in metrics if isinstance(metrics[k], (int, float))]
        if keys:
            self.fig1.clear(); self.fig1.set_facecolor(DARK)
            ax = self.fig1.add_subplot(111); ax.set_facecolor(DARK)
            vals = [float(metrics[k]) for k in keys]
            cols = [r2_color(v) if "r2" in k.lower() or "recovery" in k.lower() else CYAN for k, v in zip(keys, vals)]
            ax.bar(range(len(keys)), vals, color=cols, edgecolor="#0b0f1a")
            ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
            ax.set_title("Benchmark metrics", color="white"); ax.tick_params(colors="#cbd5e1")
            for s in ax.spines.values(): s.set_color("#263248")
            self.canvas1.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PARADIGM ZERO — Simulated-Universe Law Discovery")
        self.resize(1360, 860)
        top = QWidget(); tv = QVBoxLayout(top); tv.setContentsMargins(16, 10, 16, 4)
        tv.addWidget(QLabel("<b style='color:white;font-size:17pt'>PARADIGM ZERO</b>  "
                             "<span style='color:#94a3b8'>Can AI agents discover unknown laws inside controlled simulated universes?</span>"))
        tv.addWidget(QLabel("<span style='color:#64748b'>All discoveries concern simulated universes only — not real-world physics.</span>"))
        tabs = QTabWidget()
        self.uv, self.eq, self.hyp, self.tl, self.db = UniverseView(), EquationEditor(), None, Timeline(), Dashboard()
        self.hyp = HypothesisExplorer(self.uv)
        tabs.addTab(self.uv, "◈  Universe"); tabs.addTab(self.eq, "ƒ  Equation Editor")
        tabs.addTab(self.hyp, "✧  Hypotheses"); tabs.addTab(self.tl, "◷  Timeline"); tabs.addTab(self.db, "▦  Dashboard")
        central = QWidget(); cv = QVBoxLayout(central); cv.setContentsMargins(12, 0, 12, 8)
        cv.addWidget(top); cv.addWidget(tabs); self.setCentralWidget(central)
        bar = QStatusBar(); bar.showMessage("Ready — generate a universe to begin  •  PARADIGM ZERO"); self.setStatusBar(bar)
        self.uv.gen_btn.clicked.connect(lambda: (self.tl.log(f"Generated universe: {self.uv.eq_edit.text()} (noise={self.uv.noise.value()}, n={self.uv.n.value()})"), bar.showMessage("Universe generated — run discovery next")))
        self.hyp.run_btn.clicked.connect(lambda: (self.tl.log("Discovery run completed"), bar.showMessage("Discovery complete — see Hypotheses tab")))
        self.tl.log("PARADIGM ZERO initialized")

    def run_benchmark(self):
        if self.uv.universe is None:
            self.uv.generate()
        m = Benchmark(self.uv.universe).run("x1**2 + sin(x2)", self.uv.test_data)
        self.db.update_metrics(m); self.tl.log(f"Benchmark: {m}")
        return m


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion"); app.setStyleSheet(APP_QSS)
    w = MainWindow(); w.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
