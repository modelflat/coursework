from multiprocessing import Lock

from PyQt5.QtCore import Qt

from . import CL_INCLUDE_PATH, CL_SOURCE_PATH

from .utils import *
from .ui import ParameterizedImageWidget
from .fast_box_counting import FastBoxCounting


def get(what, kwargs, default_args):
    return kwargs.get(what, default_args[what])


class IFSFractal:

    def __init__(self, ctx, img_shape):
        self.ctx = ctx
        self.img = alloc_image(self.ctx, img_shape)
        self.img_shape = img_shape

        self.box_counter = FastBoxCounting(self.ctx)

        src = [
            read_file(os.path.join(CL_SOURCE_PATH, "phase_plot.cl")),
            read_file(os.path.join(CL_SOURCE_PATH, "param_map.cl")),
            read_file(os.path.join(CL_SOURCE_PATH, "fast_param_map.cl")),
            read_file(os.path.join(CL_SOURCE_PATH, "basins.cl")),
            read_file(os.path.join(CL_SOURCE_PATH, "bif_tree.cl")),
        ]

        self.prg = cl.Program(ctx, "\n".join(src)).build(
            options=["-I", CL_INCLUDE_PATH, "-w"]
        )

        self.real = numpy.float64

        self.map_points = None
        self.basin_points = numpy.empty((numpy.prod(img_shape), 2), dtype=self.real)
        self.basin_points_dev = alloc_like(self.ctx, self.basin_points)

        self.compute_lock = Lock()

    def compute_d(self, queue):
        return self.box_counter.compute(queue, self.img[1])

    def _compute_map(self, queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution, buf=None):
        elem_size = 8
        reqd_size = iter * numpy.prod(self.img_shape) // resolution ** 2

        if buf is None and (self.map_points is None or self.map_points.size != reqd_size):
            print("Will allocate {} bytes for parameter map".format(elem_size * reqd_size))
            self.map_points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=reqd_size * elem_size)
            buf = self.map_points

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_points(
            queue, (self.img_shape[0] // resolution, self.img_shape[1] // resolution), (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=self.real),
            # c
            numpy.array((c.real, c.imag), dtype=self.real),
            # bounds
            numpy.array(param_bounds, dtype=self.real),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            self.real(random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            buf
        )

    def _render_map(self, queue, num_points, resolution):
        color_scheme = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=4)

        periods = numpy.empty((self.img_shape[0] // resolution,
                               self.img_shape[1] // resolution),
                              dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)

        self.prg.draw_periods(
            queue, self.img_shape, None,
            numpy.int32(resolution),
            numpy.int32(num_points),
            color_scheme,
            self.map_points,
            periods_device,
            self.img[1]
        )

        cl.enqueue_copy(queue, periods, periods_device)

        return read_image(queue, *self.img, self.img_shape), periods

    def _compute_basins(self, queue, skip, h, alpha, c, bounds, root_seq, resolution):
        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_basins(
            queue, (self.img_shape[0] // resolution, self.img_shape[1] // resolution), None,

            numpy.int32(skip),
            numpy.array(bounds, dtype=self.real),
            numpy.array((c.real, c.imag), dtype=self.real),
            self.real(h),
            self.real(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,
            self.basin_points_dev
        )

    def _render_basins(self, queue, bounds, resolution, algo):
        clear_image(queue, self.img[1], self.img_shape)

        if algo == "c":
            cl.enqueue_copy(queue, self.basin_points, self.basin_points_dev)
            unique_points = numpy.unique(self.basin_points, axis=0)
            unique_points_dev = copy_dev(self.ctx, unique_points)

            print("Unique attraction points: {} / {}".format(unique_points.shape[0], self.basin_points.shape[0]))

            self.prg.draw_basins_colored(
                queue, self.img_shape, (1, 1),
                numpy.int32(resolution),
                numpy.int32(unique_points.shape[0]),
                unique_points_dev,
                self.basin_points_dev,
                self.img[1]
            )
        elif algo == "b":
            self.prg.draw_basins(
                queue, self.img_shape, (1, 1),
                numpy.int32(resolution),
                numpy.array(bounds, dtype=self.real),
                self.basin_points_dev,
                self.img[1]
            )
        else:
            raise ValueError("Unknown algo: \"{}\"".format(algo))

        return read_image(queue, *self.img, self.img_shape)

    def _compute_bif_tree(self, queue, skip, iter, z0, c, var_id, fixed_id, fixed_value,
                          other_min, other_max, root_seq=None):

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        res = numpy.empty((self.img_shape[0], iter), dtype=self.real)
        res_dev = alloc_like(self.ctx, res)

        self.prg.compute_points_for_bif_tree(
            queue, (self.img_shape[0],), None,
            # z0
            numpy.array((z0.real, z0.imag), dtype=self.real),
            # c
            numpy.array((c.real, c.imag), dtype=self.real),

            numpy.int32(var_id),
            numpy.int32(fixed_id),
            self.real(fixed_value),
            self.real(other_min),
            self.real(other_max),

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,

            res_dev
        )

        return res, res_dev

    def _render_bif_tree(self, queue, iter, var_min, var_max, result_buf):
        clear_image(queue, self.img[1], self.img_shape)

        self.prg.draw_bif_tree(
            queue, (self.img_shape[0],), None,
            numpy.int32(iter),
            self.real(var_min),
            self.real(var_max),
            numpy.int32(1),
            result_buf,
            self.img[1]
        )

        return read_image(queue, *self.img, self.img_shape)

    def call_draw_phase_portrait(
            self, queue, skip, iter, h, alpha, c, bounds, grid_size,
            z0=None, root_seq=None, clear=True):

        if clear:
            clear_image(queue, self.img[1], self.img_shape)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.newton_fractal(
            queue, (grid_size, grid_size) if z0 is None else (1, 1), None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=self.real),

            numpy.array((c.real, c.imag), dtype=self.real),
            self.real(h),
            self.real(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,

            numpy.int32(1 if z0 is not None else 0),
            numpy.array((0, 0) if z0 is None else (z0.real, z0.imag), dtype=self.real),

            self.img[1]
        )

        return read_image(queue, *self.img, self.img_shape)

    def draw_phase_portrait(self, default_args=None, **kwargs):
        queue = get("queue", kwargs, default_args)
        skip = get("skip", kwargs, default_args)
        iter = get("iter", kwargs, default_args)
        h = get("h", kwargs, default_args)
        alpha = get("alpha", kwargs, default_args)
        c = get("c", kwargs, default_args)
        bounds = get("bounds", kwargs, default_args)
        grid_size = get("grid_size", kwargs, default_args)
        z0 = get("z0", kwargs, default_args)
        root_seq = get("root_seq", kwargs, default_args)
        clear = get("clear", kwargs, default_args)

        with self.compute_lock:
            res = self.call_draw_phase_portrait(
                queue, skip, iter, h, alpha, c, bounds, grid_size, z0, root_seq, clear
            )
        return res

    def call_draw_parameter_map(
            self, queue, skip, iter, z0, c, tol, param_bounds,
            root_seq=None, resolution=1):
        self._compute_map(queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution)
        return self._render_map(queue, iter, resolution)

    def call_fast_parameter_map(
            self, queue, skip, iter, z0, c, tol, param_bounds,
            root_seq=None, resolution=1):

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.fast_param_map(
            queue, (self.img_shape[0] // resolution, self.img_shape[1] // resolution), None,
            numpy.int32(resolution),
            # z0
            numpy.array((z0.real, z0.imag), dtype=self.real),
            # c
            numpy.array((c.real, c.imag), dtype=self.real),
            # bounds
            numpy.array(param_bounds, dtype=self.real),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            self.real(random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            self.img[1]
        )

        read_image(queue, *self.img, self.img_shape)

        return self.img[0], None

    def draw_parameter_map(self, default_args=None, **kwargs):
        queue = get("queue", kwargs, default_args)
        skip = get("skip", kwargs, default_args)
        iter = get("iter", kwargs, default_args)
        z0 = get("z0", kwargs, default_args)
        c = get("c", kwargs, default_args)
        tol = get("tol", kwargs, default_args)
        param_bounds = get("param_bounds", kwargs, default_args)
        resolution = get("resolution", kwargs, default_args)
        root_seq = get("root_seq", kwargs, default_args)
        method = get("method", kwargs, default_args)

        with self.compute_lock:
            if method == "fast":
                res = self.call_fast_parameter_map(
                    queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution
                )
            elif method == "precise":
                res = self.call_draw_parameter_map(
                    queue, skip, iter, z0, c, tol, param_bounds, root_seq, resolution
                )
            else:
                raise RuntimeError("No such method")
        return res

    def call_draw_basins(
            self, queue, skip, h, alpha, c, bounds,
            root_seq=None, resolution=1, algo="c"):
        self._compute_basins(queue, skip, h, alpha, c, bounds, root_seq, resolution)
        return self._render_basins(queue, bounds, resolution, algo)

    def draw_basins(self, default_args=None, **kwargs):
        queue = get("queue", kwargs, default_args)
        skip = get("skip", kwargs, default_args)
        h = get("h", kwargs, default_args)
        alpha = get("alpha", kwargs, default_args)
        c = get("c", kwargs, default_args)
        bounds = get("bounds", kwargs, default_args)
        root_seq = get("root_seq", kwargs, default_args)
        resolution = get("resolution", kwargs, default_args)
        algo = get("algo", kwargs, default_args)

        with self.compute_lock:
            res = self.call_draw_basins(
                queue, skip, h, alpha, c, bounds, root_seq, resolution, algo
            )
        return res

    def call_draw_bif_tree(
            self, queue,
            skip, iter, z0, c,
            var_id, fixed_id, fixed_value, other_min, other_max,
            root_seq=None, var_min=-5, var_max=5):
        res, res_dev = self._compute_bif_tree(queue, skip, iter, z0, c, var_id,
                                              fixed_id, fixed_value, other_min, other_max, root_seq)
        cl.enqueue_copy(queue, res, res_dev)

        # c_var_min, c_var_max = numpy.nanmin(res, axis=1), numpy.nanmax(res, axis=1)
        #
        # print(c_var_min, c_var_max)
        #
        # var_min = max(numpy.nanmedian(c_var_min), var_min)
        # var_max = min(numpy.nanmedian(c_var_max), var_max)

        # print(var_min, var_max)

        # print(res)

        return self._render_bif_tree(queue, iter, var_min, var_max, res_dev)

    def draw_bif_tree(self, default_args=None, **kwargs):
        queue = get("queue", kwargs, default_args)
        skip = get("skip", kwargs, default_args)
        iter = get("iter", kwargs, default_args)
        z0 = get("z0", kwargs, default_args)
        c = get("c", kwargs, default_args)
        fixed_id = get("fixed_id", kwargs, default_args)
        fixed_value = get("fixed_value", kwargs, default_args)
        other_min = get("other_min", kwargs, default_args)
        other_max = get("other_max", kwargs, default_args)
        var_id = get("var_id", kwargs, default_args)
        var_min = get("var_min", kwargs, default_args)
        var_max = get("var_max", kwargs, default_args)
        root_seq = get("root_seq", kwargs, default_args)

        with self.compute_lock:
            res = self.call_draw_bif_tree(
                queue, skip, iter, z0, c, var_id, fixed_id, fixed_value, other_min, other_max,
                root_seq, var_min, var_max
            )
        return res


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
