import time

import numpy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox

import config as cfg
from core.desk import LabDesk
from core.ui import SimpleApp, createSlider, stack, ParameterizedImageWidget
from core.utils import blank_image, CLImg


numpy.random.seed(cfg.seed)


def make_phase_wgt(space_shape, image_shape):
    return ParameterizedImageWidget(
        space_shape, ("z_real", "z_imag"), shape=(True, True), textureShape=image_shape,
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

        self.desk = LabDesk(self.ctx, self.queue)

        self.left_image = CLImg(self.ctx, cfg.default_shape)
        self.right_image = CLImg(self.ctx, cfg.default_shape)

        # left widget is for parameter-related stuff
        self.left_wgt = make_param_wgt(cfg.h_bounds, cfg.alpha_bounds, cfg.param_map_image_shape)

        # right widget is for phase-related stuff
        self.right_wgt = make_phase_wgt(cfg.phase_shape, cfg.phase_image_shape)

        self.h_slider, self.h_slider_wgt = \
            createSlider("real", cfg.h_bounds, withLabel="h = {:2.3f}", labelPosition="top", withValue=0.5)

        self.alpha_slider, self.alpha_slider_wgt = \
            createSlider("real", cfg.alpha_bounds, withLabel="alpha = {:2.3f}", labelPosition="top", withValue=0.0)

        self.left_recompute_btn = QPushButton("Recompute")
        self.left_recompute_btn.clicked.connect(self.draw_left)

        self.right_recompute_btn = QPushButton("Recompute")
        self.right_recompute_btn.clicked.connect(self.draw_right)

        self.random_seq_gen_btn = QPushButton("Generate random (input length)")

        self.random_seq_reset_btn = QPushButton("Reset")
        self.param_map_draw_btn = QPushButton("Draw parameter map")
        self.clear_cb = QCheckBox("Clear image")
        self.clear_cb.setChecked(True)
        self.right_mode_cmb = QComboBox()
        self.left_mode_cmb = QComboBox()

        self.period_label = QLabel()
        self.d_label = QLabel()
        self.period_map = None

        self.root_seq_edit = QLineEdit()

        self.left_wgts = {
            "parameter map": self.draw_param_map,
            "bif tree (h)": lambda: self.draw_bif_tree(param="h"),
            "bif tree (alpha)": lambda: self.draw_bif_tree(param="alpha")
        }
        self.left_mode_cmb.addItems(self.left_wgts.keys())

        self.right_wgts = {
            "phase":  self.draw_phase,
            "basins (dev)": lambda: self.draw_basins(method="dev"),
            "basins (host)": lambda: self.draw_basins(method="host"),
            "basins (periods)": lambda: self.draw_basins(method="periods"),
        }
        self.right_mode_cmb.addItems(self.right_wgts.keys())

        self.setup_layout()
        self.connect_everything()
        self.draw_param_placeholder()
        self.draw_right()

    def setup_layout(self):
        left = stack(
            stack(self.left_mode_cmb, self.left_recompute_btn, kind="h"),
            self.left_wgt,
            self.period_label,
            stack(
                self.random_seq_gen_btn, self.random_seq_reset_btn,
                kind="h"
            ),
            self.root_seq_edit,
        )
        right = stack(
            stack(self.right_mode_cmb, self.right_recompute_btn, self.d_label, kind="h"),
            self.right_wgt,
            stack(self.clear_cb, kind="h"),
            self.alpha_slider_wgt,
            self.h_slider_wgt
        )
        self.setLayout(stack(left, right, kind="h", cm=(4, 4, 4, 4), sp=4))

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

        if cfg.param_map_draw_on_select and cfg.param_map_select_z0_from_phase:
            def select_z0(*_):
                self.draw_left()
            self.right_wgt.valueChanged.connect(select_z0)

        def gen_random_seq_fn(*_):
            try:
                root_seq_size = int(self.root_seq_edit.text())
                self.desk.root_seq = numpy.random.randint(0, 2 + 1, size=root_seq_size, dtype=numpy.int32)
                self.root_seq_edit.setText(" ".join(map(str, self.desk.root_seq)))
                print(self.desk.root_seq)
            except Exception as e:
                print(e)

        self.random_seq_gen_btn.clicked.connect(gen_random_seq_fn)

        def reset_random_seq_fn(*_):
            self.desk.root_seq = None
        self.random_seq_reset_btn.clicked.connect(reset_random_seq_fn)

        self.right_mode_cmb.currentIndexChanged.connect(self.draw_right)
        self.right_mode_cmb.currentIndexChanged.connect(self.draw_left)

    def parse_root_sequence(self):
        self.desk.update_root_sequence(self.root_seq_edit.text())

    def draw_param_placeholder(self):
        self.left_wgt.setImage(blank_image(cfg.default_shape))

    def draw_basins(self, *_, method=None):
        h, alpha = self.left_wgt.value()
        self.parse_root_sequence()

        self.desk.update_basins_params(
            h=h, alpha=alpha, method=method
        )

        self.right_wgt.setImage(
            self.desk.draw_basins(self.right_image)
        )

    def draw_param_map(self, *_):
        self.parse_root_sequence()
        print("Start computing parameter map")
        t = time.perf_counter()

        if cfg.param_map_select_z0_from_phase:
            wgt = self.right_wgt
            z0 = complex(*wgt.value())
        else:
            z0 = cfg.param_map_z0

        self.desk.update_param_map_params(
            z0=z0
        )

        image, periods = self.desk.draw_param_map(self.left_image)

        print("Computed parameter map in {:.3f} s".format(time.perf_counter() - t))

        self.left_wgt.setImage(image)
        self.period_map = periods

    def draw_phase(self, *_):
        h, alpha = self.left_wgt.value()
        self.parse_root_sequence()

        self.desk.update_phase_plot_params(
            h=h,
            alpha=alpha,
            z0=cfg.phase_z0 if not cfg.phase_plot_select_point else complex(
                *self.right_wgt.value()),
            clear=self.clear_cb.isChecked(),
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
                "other_min": cfg.h_bounds[0],
                "other_max": cfg.h_bounds[1]
            }
        elif param == "alpha":
            param_properties = {
                "fixed_id": 0,
                "fixed_value": h,
                "other_min": cfg.alpha_bounds[0],
                "other_max": cfg.alpha_bounds[1]
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
        x_px //= cfg.param_map_resolution
        y_px //= cfg.param_map_resolution
        if self.period_map is not None:
            x_px = max(min(cfg.param_map_image_shape[0] // cfg.param_map_resolution - 1, x_px), 0)
            y_px = max(min(cfg.param_map_image_shape[1] // cfg.param_map_resolution - 1, y_px), 0)
            # print(x_px, y_px)
            y, x = int(y_px), int(x_px)
            per = self.period_map[y][x]
            self.period_label.setText(
                "Detected period: {}".format("<chaos({})>".format(per) if per > 0.25 * cfg.param_map_iter else per)
            )

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
