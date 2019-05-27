import time
from collections import defaultdict

import numpy
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox
from PyQt5.QtCore import Qt

import config as cfg

from core.basins import BasinsOfAttraction
from core.bif_tree import BifTree
from core.fast_box_counting import FastBoxCounting
from core.param_map import ParameterMap
from core.phase_plot import PhasePlot
from core.ui import SimpleApp, createSlider, stack, ParameterizedImageWidget
from core.utils import blank_image, CLImg

numpy.random.seed(42)


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

        self.left_image = CLImg(self.ctx, (512, 512))
        self.param_map = ParameterMap(self.ctx, self.left_image)
        self.bif_tree = BifTree(self.ctx, self.left_image)

        self.right_image = CLImg(self.ctx, (512, 512))
        self.basins = BasinsOfAttraction(self.ctx, self.right_image)
        self.phase_plot = PhasePlot(self.ctx, self.right_image)

        self.fbc = FastBoxCounting(self.ctx)

        self.default_args = defaultdict(lambda: None)

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

        self.random_seq = None

        self.left_wgts = {
            "parameter map": self.draw_param_map,
            "bif tree (h)": lambda: self.draw_bif_tree(param="h"),
            "bif tree (alpha)": lambda: self.draw_bif_tree(param="alpha")
        }
        self.left_mode_cmb.addItems(self.left_wgts.keys())

        self.right_wgts = {
            "phase":  self.draw_phase,
            "basins": lambda: self.draw_basins(algo="b"),
            "basins colored": lambda: self.draw_basins(algo="c")
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
                self.random_seq = numpy.random.randint(0, 2 + 1, size=root_seq_size, dtype=numpy.int32)
                print(self.random_seq)
                self.root_seq_edit.setText(" ".join(map(str, self.random_seq)))
            except Exception as e:
                print(e)

        self.random_seq_gen_btn.clicked.connect(gen_random_seq_fn)

        def reset_random_seq_fn(*_):
            self.random_seq = None
        self.random_seq_reset_btn.clicked.connect(reset_random_seq_fn)

        self.right_mode_cmb.currentIndexChanged.connect(self.draw_right)
        self.right_mode_cmb.currentIndexChanged.connect(self.draw_left)

    def parse_root_sequence(self):
        try:
            raw = self.root_seq_edit.text()
            if self.random_seq is not None:
                return self.random_seq
            else:
                l = list(map(int, raw.split()))
                if len(l) == 0 or not all(map(lambda x: x <= 2, l)):
                    return None
                else:
                    return l
        except:
            return None

    def draw_param_placeholder(self):
        self.left_wgt.setImage(blank_image((512, 512)))

    def draw_basins(self, *_, algo="b"):
        h, alpha = self.left_wgt.value()

        image = self.basins.compute(
            # default_args=self.default_args,
            queue=self.queue,
            skip=cfg.basins_skip,
            h=h, alpha=alpha, c=cfg.C,
            bounds=cfg.phase_shape,
            root_seq=self.parse_root_sequence(),
            method="dev",
            scale_factor=cfg.basins_resolution,
        )
        self.right_wgt.setImage(image)

    def draw_param_map(self, *_):
        print("Start computing parameter map")
        t = time.perf_counter()

        if cfg.param_map_select_z0_from_phase:
            wgt = self.right_wgt
            z0 = complex(*wgt.value())
        else:
            z0 = cfg.param_map_z0

        image, periods = self.param_map.compute(
            queue=self.queue,
            skip=cfg.param_map_skip,
            iter=cfg.param_map_iter,
            z0=z0, c=cfg.C,
            tol=cfg.param_map_tolerance,
            bounds=(*cfg.h_bounds, *cfg.alpha_bounds),
            root_seq=self.parse_root_sequence(),
            method="fast",
            scale_factor=cfg.param_map_resolution,
            img=self.left_image
        )

        print("Computed parameter map in {:.3f} s".format(time.perf_counter() - t))

        self.left_wgt.setImage(image)
        self.period_map = periods

    def draw_phase(self, *_):
        h, alpha = self.left_wgt.value()

        image = self.phase_plot.compute(
            queue=self.queue,
            skip=cfg.phase_skip,
            iter=cfg.phase_iter,
            h=h, alpha=alpha,
            c=cfg.C,
            bounds=cfg.phase_shape,
            grid_size=cfg.phase_grid_size,
            z0=cfg.phase_z0 if not cfg.phase_plot_select_point else complex(
                *self.right_wgt.value()
            ),
            root_seq=self.parse_root_sequence(),
            clear=self.clear_cb.isChecked()
        )
        self.right_wgt.setImage(image)

        D = self.fbc.compute(self.queue, self.right_image.dev)

        self.d_label.setText("D = {:.3f}".format(D))

    def draw_bif_tree(self, *_, param=None):
        h, alpha = self.left_wgt.value()

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

        z0 = cfg.bif_tree_z0

        image = self.bif_tree.compute(
            queue=self.queue,
            skip=cfg.bif_tree_skip,
            iter=cfg.bif_tree_iter,
            z0=z0,
            c=cfg.C,
            var_id=0,
            root_seq=self.parse_root_sequence(),
            var_min=-3,
            var_max=3,
            **param_properties
        )
        self.left_wgt.setImage(image.copy())

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
        # print("draw_right called")
        what = self.right_mode_cmb.currentText()
        self.set_period_label()
        self.right_wgts[what]()

    def draw_left(self):
        what = self.left_mode_cmb.currentText()
        # self.set_period_label()
        self.left_wgts[what]()


if __name__ == '__main__':
    CourseWork().run()
