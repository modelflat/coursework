from collections import Counter

import numpy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QComboBox
from matplotlib import patches
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from config import Config
from core.desk import LabDesk
from core.ui import SimpleApp, createSlider, stack, ParameterizedImageWidget
from core.utils import blank_image, CLImg


numpy.random.seed(Config.seed)


def make_phase_wgt(space_shape, image_shape):
    return ParameterizedImageWidget(
        bounds=space_shape,
        names=("z_real", "z_imag"),
        shape=(True, True),
        textureShape=image_shape,
        targetColor=Qt.gray
    )


def make_param_wgt(h_bounds, alpha_bounds, image_shape):
    return ParameterizedImageWidget(
        bounds=(*h_bounds, *alpha_bounds),
        names=("h", "alpha"),
        shape=(True, True),
        textureShape=image_shape,
        targetColor=Qt.gray
    )


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")

        self.desk = LabDesk(self.ctx, self.queue, Config)

        self.left_image = CLImg(self.ctx, Config.image_shape)
        self.right_image = CLImg(self.ctx, Config.image_shape)

        self.figure = Figure(figsize=Config.attractor_plot_shape, dpi=Config.attractor_plot_dpi)
        self.canvas = FigureCanvas(self.figure)

        self.canvas.setMinimumSize(*Config.image_shape)

        # left widget is for parameter-related stuff
        self.left_wgt = make_param_wgt(Config.h_bounds, Config.alpha_bounds, Config.image_shape)

        # right widget is for phase-related stuff
        self.right_wgt = make_phase_wgt(Config.phase_shape, Config.image_shape)

        self.h_slider, self.h_slider_wgt = \
            createSlider("real", Config.h_bounds, withLabel="h = {:2.3f}", labelPosition="top", withValue=0.5)

        self.alpha_slider, self.alpha_slider_wgt = \
            createSlider("real", Config.alpha_bounds, withLabel="alpha = {:2.3f}", labelPosition="top", withValue=0.0)

        self.left_recompute_btn = QPushButton("Recompute")
        self.left_recompute_btn.clicked.connect(self.draw_left)

        self.right_recompute_btn = QPushButton("Recompute")
        self.right_recompute_btn.clicked.connect(self.draw_right)

        self.random_seq_reset_btn = QPushButton("Reset")
        self.param_map_draw_btn = QPushButton("Draw parameter map")

        self.right_mode_cmb = QComboBox()
        self.left_mode_cmb = QComboBox()

        self.period_label = QLabel()
        self.d_label = QLabel()
        self.period_map = None

        self.root_seq_edit = QLineEdit()

        self.left_wgts = {
            "parameter map": self.draw_param_map,
            # "bif tree (h)": lambda: self.draw_bif_tree(param="h"),
            # "bif tree (alpha)": lambda: self.draw_bif_tree(param="alpha")
        }
        self.left_mode_cmb.addItems(self.left_wgts.keys())
        self.left_mode_cmb.setCurrentText("parameter map")

        self.right_wgts = {
            "phase":  self.draw_phase,
            "basins (attractors)": lambda: self.draw_basins(method="basins"),
            # "basins (periods)": lambda: self.draw_basins(method="periods"),
        }
        self.right_mode_cmb.addItems(self.right_wgts.keys())
        self.right_mode_cmb.setCurrentText("basins (attractors)")

        self.root_seq_edit.setText("0")

        self.setup_layout()
        self.connect_everything()
        self.draw_param_placeholder()
        self.draw_left()
        self.draw_right()

    def setup_layout(self):
        left = stack(
            stack(self.left_mode_cmb, self.left_recompute_btn, kind="h"),
            self.left_wgt,
            self.period_label,
            self.d_label,
            stack(self.root_seq_edit, self.random_seq_reset_btn, kind="h"),
        )
        right = stack(
            stack(self.right_mode_cmb, self.right_recompute_btn, kind="h"),
            self.right_wgt,
            self.alpha_slider_wgt,
            self.h_slider_wgt,
        )
        right_canvas = stack(
            self.canvas
        )
        self.setLayout(stack(left, right, right_canvas, kind="h", cm=(4, 4, 4, 4), sp=4))

    def connect_everything(self):
        def set_sliders_and_draw(val):
            self.draw_right()
            self.set_values_no_signal(*val)

        self.left_wgt.valueChanged.connect(set_sliders_and_draw)

        def set_h_value(h):
            _, alpha = self.left_wgt.value()
            self.set_values_no_signal(h, alpha)
            self.draw_right()
            if "bif" in self.left_mode_cmb.currentText():
                self.draw_left()
        self.h_slider.valueChanged.connect(set_h_value)

        def set_alpha_value(alpha):
            h, _ = self.left_wgt.value()
            self.set_values_no_signal(h, alpha)
            self.draw_right()
            if "bif" in self.left_mode_cmb.currentText():
                self.draw_left()
        self.alpha_slider.valueChanged.connect(set_alpha_value)

        if Config.param_map_draw_on_select and Config.param_map_select_z0_from_phase:
            def select_z0(*_):
                self.draw_left()
            self.right_wgt.valueChanged.connect(select_z0)

        def reset_random_seq_fn(*_):
            self.desk.root_seq = None
        self.random_seq_reset_btn.clicked.connect(reset_random_seq_fn)

        self.right_mode_cmb.currentIndexChanged.connect(self.draw_right)
        self.right_mode_cmb.currentIndexChanged.connect(self.draw_left)

    def parse_root_sequence(self):
        self.desk.update_root_sequence(self.root_seq_edit.text())

    def draw_param_placeholder(self):
        self.left_wgt.setImage(blank_image(Config.image_shape))

    def draw_basins(self, *_, method=None):
        h, alpha = self.left_wgt.value()
        self.parse_root_sequence()

        self.desk.update_basins_params(
            h=h, alpha=alpha, method=method, threshold=10,
            table_size=2 ** 24 - 3
        )

        attractors, image = self.desk.draw_basins(self.right_image)

        self.right_wgt.setImage(image)

        period_counts = Counter()

        self.figure.clf()
        ax = self.figure.subplots(1, 1)

        if Config.attractor_plot_fixed_size:
            ax.set_xlim(Config.phase_shape[0:2])
            ax.set_ylim(Config.phase_shape[2:4])

        known_points = sum((a["occurences"] for a in attractors))
        drawn_points = 0
        total_area = numpy.prod(self.right_image.shape)

        for attractor in sorted(attractors, key=lambda x: (x["period"], -x["occurences"])):
            period = attractor["period"]
            area = 100 * attractor["occurences"] / total_area
            drawn_points += attractor["occurences"]
            period_counts[period] += 1
            if period_counts[period] < 4:
                color = (attractor["color"][0] / 360, *attractor["color"][1:])
                x, y = attractor["attractor"].T
                ax.plot(
                    x, y, "-o",
                    label=f"a #{period_counts[period]} (p {period}) ({area:.2f}%)",
                    alpha=0.8,
                    color=hsv_to_rgb(color)
                )

        ax.grid(which="both")
        self.figure.tight_layout(pad=1.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(patches.Patch(
            color="black",
            label=f"chaos / inf {100 * (total_area - known_points) / total_area:.2f}%",
        ))
        handles.append(patches.Patch(
            label=f"other attr. {100 * (known_points - drawn_points) / total_area:.2f}%",
            alpha=0.0
        ))
        self.figure.legend(handles=handles)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def draw_param_map(self, *_):
        self.parse_root_sequence()

        if Config.param_map_select_z0_from_phase:
            wgt = self.right_wgt
            z0 = complex(*wgt.value())
        else:
            z0 = Config.param_map_z0

        self.desk.update_param_map_params(z0=z0)

        image, periods = self.desk.draw_param_map(self.left_image)

        self.left_wgt.setImage(image)
        self.period_map = periods

    def draw_phase(self, *_):
        h, alpha = self.left_wgt.value()
        self.parse_root_sequence()

        self.desk.update_phase_plot_params(
            h=h,
            alpha=alpha,
            skip=Config.n_skip + Config.n_iter - 1,
            iter=1,
            z0=Config.phase_z0 if not Config.phase_plot_select_point else complex(*self.right_wgt.value()),
        )

        self.right_wgt.setImage(
            self.desk.draw_phase(self.right_image)
        )

        D = self.desk.fbc.compute(self.queue, self.right_image.dev)

        self.d_label.setText("D = {:.3f}".format(D))

    def draw_bif_tree(self, *_, param=None):
        h, alpha = self.left_wgt.value()
        self.parse_root_sequence()

        if param == "h":
            param_properties = {
                "fixed_id": 1,
                "fixed_value": alpha,
                "other_min": Config.h_bounds[0],
                "other_max": Config.h_bounds[1]
            }
        elif param == "alpha":
            param_properties = {
                "fixed_id": 0,
                "fixed_value": h,
                "other_min": Config.alpha_bounds[0],
                "other_max": Config.alpha_bounds[1]
            }
        else:
            raise RuntimeError()

        self.desk.update_bif_tree_params(
            **param_properties
        )

        self.left_wgt.setImage(
            self.desk.draw_bif_tree(self.left_image)
        )

    def set_period_label(self):
        x_px, y_px = self.left_wgt._imageWidget.targetPx()
        if self.period_map is not None:
            shp = self.period_map.shape
            x_px = max(min(shp[0] - 1, x_px), 0)
            y_px = max(min(shp[1] - 1, y_px), 0)
            y, x = int(y_px), int(x_px)
            per = self.period_map[y][x]
            self.period_label.setText(f"Detected period: {per}")

    def set_values_no_signal(self, h, alpha):
        self.left_wgt.blockSignals(True)
        self.left_wgt.setValue((h, alpha))
        self.left_wgt.blockSignals(False)
        self.h_slider.blockSignals(True)
        self.h_slider.setValue(h)
        self.h_slider.blockSignals(False)
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(alpha)
        self.alpha_slider.blockSignals(False)

    def draw_right(self):
        what = self.right_mode_cmb.currentText()
        self.set_period_label()
        self.right_wgts[what]()

    def draw_left(self):
        what = self.left_mode_cmb.currentText()
        self.left_wgts[what]()


if __name__ == '__main__':
    CourseWork().run()
