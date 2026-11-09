import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QDoubleSpinBox, QTextEdit, QTableWidget, QTableWidgetItem,
                               QComboBox, QCheckBox, QSlider)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa

from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.instruments import Instrument, ObservationSystem
from paradigm_zero.core.symbolic_regression import SymbolicRegressor
from paradigm_zero.core.testing import HypothesisTester
from paradigm_zero.core.benchmark import Benchmark

DARK = "#0b0f1a"
PANEL = "#111827"
ACCENT = "#7c5cff"
CYAN = "#22d3ee"

APP_QSS = f"""
QMainWindow {{ background: {DARK}; }}
QTabWidget::pane {{ background: {PANEL}; border: 1px solid #1f2937; }}
QTabBar::tab {{ background: #1f2937; color: #e5e7eb; padding: 8px 16px; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: white; }}
QLabel {{ color: #e5e7eb; font-size: 11pt; }}
QPushButton {{ background: {ACCENT}; color: white; border-radius: 8px; padding: 7px 14px; font-weight: bold; }}
QPushButton:hover {{ background: #8b6dff; }}
QLineEdit, QTextEdit, QTableWidget {{ background: #0f172a; color: #e5e7eb; border: 1px solid #263248; border-radius: 6px; }}
QComboBox {{ background: #1f2937; color: #e5e7eb; padding: 4px; }}
"""


class UniverseView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Hidden equation:"))
        self.eq_edit = QLineEdit("x1**2 + sin(x2) + 0.5*x1*x2")
        row.addWidget(self.eq_edit)
        self.layout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Noise:"))
        self.noise = QDoubleSpinBox(); self.noise.setValue(0.1); self.noise.setSingleStep(0.05)
        row2.addWidget(self.noise)
        row2.addWidget(QLabel("Samples:"))
        self.n = QSpinBox(); self.n.setValue(300); self.n.setRange(50, 5000)
        row2.addWidget(self.n)
        self.gen_btn = QPushButton("Generate Universe")
        row2.addWidget(self.gen_btn)
        self.layout.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Colormap:"))
        self.cmap = QComboBox(); self.cmap.addItems(["plasma", "viridis", "inferno", "turbo", "ocean"])
        row3.addWidget(self.cmap)
        self.surface_chk = QCheckBox("Ground-truth surface"); self.surface_chk.setChecked(True)
        row3.addWidget(self.surface_chk)
        self.points_chk = QCheckBox("Observations"); self.points_chk.setChecked(True)
        row3.addWidget(self.points_chk)
        self.wire_chk = QCheckBox("Wireframe"); self.wire_chk.setChecked(True)
        row3.addWidget(self.wire_chk)
        self.rot_chk = QCheckBox("Auto-rotate"); self.rot_chk.setChecked(True)
        row3.addWidget(self.rot_chk)
        row3.addWidget(QLabel("Speed:"))
        self.speed = QSlider(Qt.Horizontal); self.speed.setRange(1, 10); self.speed.setValue(3)
        row3.addWidget(self.speed)
        row3.addWidget(QLabel("Quality:"))
        self.quality = QComboBox(); self.quality.addItems(["Ultra", "High", "Balanced", "Performance"])
        row3.addWidget(self.quality)
        self.layout.addLayout(row3)
        self.figure = Figure(figsize=(9, 6), dpi=150, facecolor=DARK)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.universe = None
        self.train_data = None
        self.test_data = None
        self.azim = 45.0
        self.gen_btn.clicked.connect(self.generate)
        for w in (self.cmap, self.surface_chk, self.points_chk, self.wire_chk, self.quality):
            if hasattr(w, "stateChanged"):
                w.stateChanged.connect(lambda: self.plot())
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(lambda: self.plot())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)
        self.timer.start(60)

    def _rotate(self):
        if not self.rot_chk.isChecked():
            return
        self.azim = (self.azim + self.speed.value() * 0.6) % 360
        self._apply_view()

    def _apply_view(self):
        try:
            ax = self.figure.axes[0]
            ax.view_init(elev=28, azim=self.azim)
            self.canvas.draw_idle()
        except Exception:
            pass

    def generate(self):
        cfg = UniverseConfig(hidden_equation=self.eq_edit.text(), noise=self.noise.value(), n_samples=self.n.value())
        self.universe = Universe(cfg)
        self.train_data = self.universe.generate()
        self.test_data = self.universe.generate(n=100, seed=cfg.test_seed)
        self.plot()

    def _grid_n(self):
        return {"Ultra": 90, "High": 64, "Balanced": 40, "Performance": 24}[self.quality.currentText()]

    def plot(self):
        self.figure.clear()
        self.figure.set_facecolor(DARK)
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_facecolor(DARK)
        ax.xaxis.pane.set_facecolor((0.05, 0.07, 0.12, 1))
        ax.yaxis.pane.set_facecolor((0.05, 0.07, 0.12, 1))
        ax.zaxis.pane.set_facecolor((0.08, 0.10, 0.18, 1))
        ax.xaxis.pane.set_edgecolor("#263248"); ax.yaxis.pane.set_edgecolor("#263248"); ax.zaxis.pane.set_edgecolor("#263248")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        ax.grid(True, color="#263248", alpha=0.6)
        if self.train_data is None:
            ax.set_title("PARADIGM ZERO — 3D Scientific Universe", color="white", fontsize=13, pad=14)
            ax.text2D(0.5, 0.5, "Generate a universe to begin", color="#94a3b8", ha="center")
            self.canvas.draw(); return
        cmap = self.cmap.currentText()
        x1 = self.train_data["x1"]; x2 = self.train_data["x2"]; y = self.train_data["y"]
        zmin, zmax = float(np.min(y)), float(np.max(y))
        if self.surface_chk.isChecked():
            lo1, hi1 = float(np.min(x1)), float(np.max(x1))
            lo2, hi2 = float(np.min(x2)), float(np.max(x2))
            g = self._grid_n()
            gx, gy = np.meshgrid(np.linspace(lo1, hi1, g), np.linspace(lo2, hi2, g))
            gz = self.universe.evaluate(self.universe.ground_truth_str(), {"x1": gx.ravel(), "x2": gy.ravel()}).reshape(gx.shape)
            zs = gz - np.ptp(gz) * 0.06
            surf = ax.plot_surface(gx, gy, zs, cmap=cmap, alpha=0.82, linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1, vmin=zmin, vmax=zmax)
            if self.wire_chk.isChecked():
                ax.plot_wireframe(gx, gy, zs, color="#e2e8f0", alpha=0.10, linewidths=0.4)
            try:
                ax.contourf(gx, gy, zs, zdir="z", offset=zmin - np.ptp(gz) * 0.10, cmap=cmap, alpha=0.35)
            except Exception:
                pass
            self.figure.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="y (simulated)")
        if self.points_chk.isChecked():
            sc = ax.scatter(x1, x2, y, c=y, cmap=cmap, s=26, alpha=0.95, edgecolors="#0b0f1a", linewidths=0.5, depthshade=True)
            ax.scatter(x1[:1], x2[:1], y[:1], s=90, c="#22d3ee", edgecolors="white", linewidths=1.2, depthshade=False)
        ax.set_xlabel("x1 — observed", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_ylabel("x2 — observed", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_zlabel("y — simulated response", color="#cbd5e1", fontsize=10, labelpad=8)
        ax.set_title("PARADIGM ZERO — 3D Scientific Universe  •  simulated law: "
                     f"{self.universe.ground_truth_str()}", color="white", fontsize=11, pad=14)
        ax.view_init(elev=28, azim=self.azim)
        # premium overlay metrics
        ax.text2D(0.02, 0.95, f"n={len(y)}  noise={self.noise.value():.2f}  cmap={cmap}  grid={self._grid_n()}²",
                  color="#94a3b8", fontsize=8, transform=ax.transAxes,
                  bbox=dict(facecolor="#111827", alpha=0.85, edgecolor="#263248", boxstyle="round,pad=0.4"))
        self.canvas.draw()


class EquationEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Candidate equation (SymPy syntax):"))
        self.edit = QTextEdit("x1**2 + sin(x2) + x1*x2")
        v.addWidget(self.edit)
        self.validate_btn = QPushButton("Validate")
        v.addWidget(self.validate_btn)
        self.status = QLabel("")
        v.addWidget(self.status)
        self.validate_btn.clicked.connect(self.validate)

    def validate(self):
        import sympy as sp
        try:
            sp.sympify(self.edit.toPlainText())
            self.status.setText("Valid expression.")
        except Exception as e:
            self.status.setText(f"Invalid: {e}")


class HypothesisExplorer(QWidget):
    def __init__(self, universe_view, parent=None):
        super().__init__(parent)
        self.uv = universe_view
        v = QVBoxLayout(self)
        self.run_btn = QPushButton("Run Discovery")
        v.addWidget(self.run_btn)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Expression", "Train R2", "Test R2", "Valid"])
        v.addWidget(self.table)
        self.run_btn.clicked.connect(self.run)

    def run(self):
        uv = self.uv
        if uv.universe is None:
            uv.generate()
        X = {k: uv.train_data[k] for k in uv.universe.config.observed_vars}
        y = uv.train_data["y"]
        reg = SymbolicRegressor(uv.universe.config.observed_vars)
        cands = reg.fit(X, y)
        tester = HypothesisTester(uv.universe)
        self.table.setRowCount(0)
        for c in cands:
            tr = tester.test(c, uv.train_data)
            te = tester.test(c, uv.test_data)
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(c))
            self.table.setItem(r, 1, QTableWidgetItem(f"{tr['r2']:.4f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{te['r2']:.4f}"))
            self.table.setItem(r, 3, QTableWidgetItem(str(tr['valid'])))


class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Experiment Timeline"))
        self.listw = QTextEdit()
        self.listw.setReadOnly(True)
        v.addWidget(self.listw)

    def log(self, msg):
        self.listw.append(msg)


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Results Dashboard"))
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        v.addWidget(self.text)

    def update_metrics(self, metrics):
        for k, val in metrics.items():
            self.text.append(f"{k}: {val}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PARADIGM ZERO")
        self.resize(1280, 820)
        tabs = QTabWidget()
        self.uv = UniverseView()
        tabs.addTab(self.uv, "Universe")
        self.eq = EquationEditor()
        tabs.addTab(self.eq, "Equation Editor")
        self.hyp = HypothesisExplorer(self.uv)
        tabs.addTab(self.hyp, "Hypotheses")
        self.tl = Timeline()
        tabs.addTab(self.tl, "Timeline")
        self.db = Dashboard()
        tabs.addTab(self.db, "Dashboard")
        self.setCentralWidget(tabs)
        self.uv.gen_btn.clicked.connect(lambda: self.tl.log("Generated universe"))

    def run_benchmark(self):
        if self.uv.universe is None:
            self.uv.generate()
        expr = "x1**2 + sin(x2)"
        bm = Benchmark(self.uv.universe)
        m = bm.run(expr, self.uv.test_data)
        self.db.update_metrics(m)
        self.tl.log(f"Benchmark: {m}")
        return m


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
