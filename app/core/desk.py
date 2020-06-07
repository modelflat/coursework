from collections import defaultdict
from multiprocessing import Lock

import numpy

from PyQt5.QtWidgets import QWidget

from core.basins import BasinsOfAttraction
from core.bif_tree import BifTree
from core.fast_box_counting import FastBoxCounting
from core.param_map import ParameterMap
from core.phase_plot import PhasePlot


def make_color_fn():
    assigned_colors = defaultdict(lambda: [])
    assigned_colors_tracker = set()

    max_norm = 100

    def color_fn(attractor):
        period = attractor["period"]
        attr = attractor["attractor"]

        norm = numpy.linalg.norm(attr.flatten())

        if period == 1:
            col = (0, 1.0, 1.0)
        elif period == 3:
            p1, p2, p3 = attr.view(dtype=numpy.complex128).flatten()
            if abs(p1) < 1.0 and abs(p3) < 1.0 and abs(p2) > 20.0:
                n = 0.6 * min(norm / 80, 1.0)
                col = (240, 1.0, 1.0 - n)
            else:
                col = (280, 1.0, 1.0)
        elif norm > max_norm:
            col = (300 * period / 64, 1.0, 0.3)
        else:
            col = (240 * period / 64, 1.0, 1.0)

        i = 0
        while col in assigned_colors_tracker:
            col = (col[0], col[1] - 0.25, col[2])
            i += 1
            if i >= 3:
                break
        
        assigned_colors_tracker.add(col)

        assigned_colors[period].append((col, attractor))

        return col
    
    return color_fn


class LabDesk(QWidget):

    def __init__(self, ctx, queue, cfg):
        super(LabDesk, self).__init__(None)

        param_bounds = (*cfg.h_bounds, *cfg.alpha_bounds)

        self.ctx = ctx
        self.queue = queue

        self.bif_tree = BifTree(self.ctx)
        self.bif_tree_params = {
            "skip": cfg.bif_tree_skip,
            "iter": cfg.bif_tree_iter,
            "z0": cfg.bif_tree_z0,
            "c": cfg.C,
            "var_id": 0,
            "fixed_id": None,
            "fixed_value": None,
            "other_min": None,
            "other_max": None,
            "var_min": -3,
            "var_max": 3,
            "try_rescaling": False
        }

        self.param_map = ParameterMap(self.ctx)
        self.param_map_params = {
            "skip": cfg.param_map_skip,
            "iter": cfg.param_map_iter,
            "z0": cfg.param_map_z0,
            "c": cfg.C,
            "tol": cfg.param_map_tolerance,
            "bounds": param_bounds,
            "seed": None
        }

        self.basins = BasinsOfAttraction(self.ctx)
        self.basins_params = {
            "skip": cfg.basins_skip,
            "iter": cfg.basins_iter,
            "h": None,
            "alpha": None,
            "c": cfg.C,
            "bounds": cfg.phase_shape,
            "method": "periods",
            "color_init": None,
            "threshold": 128
        }

        self.phase_plot = PhasePlot(self.ctx)
        self.phase_params = {
            "skip": cfg.phase_skip,
            "h": None,
            "alpha": None,
            "iter": cfg.param_map_iter,
            "c": cfg.C,
            "bounds": cfg.phase_shape,
            "grid_size": cfg.phase_grid_size,
            "z0": cfg.phase_z0,
            "clear": True,
            "seed": cfg.seed
        }

        self.fbc = FastBoxCounting(self.ctx)

        self.root_seq = None

        self.compute_lock = Lock()

    def update_root_sequence(self, s):
        try:
            l = list(map(int, s.split()))
            if len(l) == 0 or not all(map(lambda x: x <= 2, l)):
                self.root_seq = None
            else:
                self.root_seq = l
        except ValueError as e:
            self.root_seq = None
            print(e)

    def update_phase_plot_params(self, **kwargs):
        self.phase_params = { **self.phase_params, **kwargs }

    def draw_phase(self, img):
        with self.compute_lock:
            return self.phase_plot.compute(self.queue, img, root_seq=self.root_seq, **self.phase_params)

    def update_basins_params(self, **kwargs):
        self.basins_params = {**self.basins_params, **kwargs}

    def draw_basins(self, img):
        with self.compute_lock:
            color_fn = make_color_fn()
            return self.basins.compute(self.queue, img, root_seq=self.root_seq, color_fn=color_fn, **self.basins_params)

    def periods_and_attractors(self):
        pass

    def update_param_map_params(self, **kwargs):
        self.param_map_params = { **self.param_map_params, **kwargs }

    def draw_param_map(self, img):
        with self.compute_lock:
            return self.param_map.compute(self.queue, img, root_seq=self.root_seq, **self.param_map_params)

    def update_bif_tree_params(self, **kwargs):
        self.bif_tree_params = { **self.bif_tree_params, **kwargs }

    def draw_bif_tree(self, img):
        with self.compute_lock:
            return self.bif_tree.compute(self.queue, img, root_seq=self.root_seq, **self.bif_tree_params)
