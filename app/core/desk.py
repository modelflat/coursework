from PyQt5.QtWidgets import QWidget

import config as cfg

from core.basins import BasinsOfAttraction
from core.bif_tree import BifTree
from core.fast_box_counting import FastBoxCounting
from core.param_map import ParameterMap
from core.phase_plot import PhasePlot

from multiprocessing import Lock


class LabDesk(QWidget):

    def __init__(self, ctx, queue):
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
            "method": "fast",
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
            "method": "dev",
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
        self.basins_params = { **self.basins_params, **kwargs }

    def draw_basins(self, img):
        with self.compute_lock:
            return self.basins.compute(self.queue, img, root_seq=self.root_seq, **self.basins_params)

    def periods_and_sttractors(self):
        from research.attractor_miner_param_map import periods_and_attractors_at_point
        import json
        import os
        points_file = "../research/attr_res/points.json"

        if os.path.exists(points_file):
            with open(points_file, "r") as f:
                points = json.load(f)
        else:
            points = []

        periods_and_attractors_at_point(
            self.ctx, self.queue, self.param_map, self.basins,
            self.basins_params['h'], self.basins_params['alpha'],
            attractors=self.precomputed_attractors, colors=[
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                (0.0, 1.0, 1.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
            ] * 3,
            root_seq=self.root_seq,
            output_name="attr", output_dir=f"../research/attr_res/{len(points)}"
        )

        points.append([self.basins_params['h'], self.basins_params['alpha']])

        with open(points_file, "w") as f:
            json.dump(points, f)

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
