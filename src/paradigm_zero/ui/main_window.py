import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QDoubleSpinBox, QTextEdit, QTableWidget, QTableWidgetItem,
                               QComboBox)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa

from paradigm_zero.core.universe import Universe, UniverseConfig
from paradigm_zero.core.instruments import Instrument, ObservationSystem
from paradigm_zero.core.symbolic_regression import SymbolicRegressor
from paradigm_zero.core.testing import HypothesisTester
from paradigm_zero.core.benchmark import Benchmark


class UniverseView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.config_edits = {}
        row = QHBoxLayout()
        row.addWidget(QLabel("Equation (hidden):"))
        self.eq_edit = QLineEdit("x1**2 + sin(x2) + 0.5*x1*x2")
        row.addWidget(self.eq_edit)
        self.layout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Noise:"))
        self.noise = QDoubleSpinBox(); self.noise.setValue(0.1); self.noise.setSingleStep(0.05)
        row2.addWidget(self.noise)
        row2.addWidget(QLabel("N samples:"))
        self.n = QSpinBox(); self.n.setValue(200); self.n.setRange(10, 5000)
        row2.addWidget(self.n)
        self.gen_btn = QPushButton("Generate Universe")
        row2.addWidget(self.gen_btn)
        self.layout.addLayout(row2)
        self.figure = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.universe = None
        self.train_data = None
        self.test_data = None
        self.gen_btn.clicked.connect(self.generate)

    def generate(self):
        cfg = UniverseConfig(hidden_equation=self.eq_edit.text(), noise=self.noise.value(), n_samples=self.n.value())
        self.universe = Universe(cfg)
        self.train_data = self.universe.generate()
        self.test_data = self.universe.generate(n=100, seed=cfg.test_seed)
        self.plot()

    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        if self.train_data is None:
            self.canvas.draw(); return
        x1 = self.train_data["x1"]; x2 = self.train_data["x2"]; y = self.train_data["y"]
        ax.scatter(x1, x2, y, s=8, alpha=0.7)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        ax.set_title("3D Scientific Universe")
        self.canvas.draw()


class EquationEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Candidate equation (sympy syntax):"))
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
        self.resize(1100, 750)
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
        # use first hypothesis
        expr = "x1**2 + sin(x2)"
        bm = Benchmark(self.uv.universe)
        m = bm.run(expr, self.uv.test_data)
        self.db.update_metrics(m)
        self.tl.log(f"Benchmark: {m}")
        return m


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
