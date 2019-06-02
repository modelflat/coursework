import numpy
import pyopencl as cl
from tqdm import tqdm

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class ParameterMap:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, ("param_map.cl", "fast_param_map.cl"))
        self.map_points = None

    def _compute_precise(self, queue, skip, iter, z0, c, tol, bounds, root_seq, scale_factor, img, seed=None):

        shape = img.shape

        elem_size = real_type().nbytes
        reqd_size = iter * numpy.prod(shape) // scale_factor ** 2

        if self.map_points is None or self.map_points.size != reqd_size:
            self.map_points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=reqd_size * elem_size)

        buf = self.map_points

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_points(
            queue, (shape[0] // scale_factor, shape[1] // scale_factor), (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),
            # bounds
            numpy.array(bounds, dtype=real_type),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.uint64(seed if seed is not None else random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            buf
        )

        periods = numpy.empty((shape[0] // scale_factor, shape[1] // scale_factor),
                              dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)

        self.prg.draw_periods(
            queue, shape, None,
            numpy.int32(scale_factor),
            numpy.int32(iter),
            buf,
            periods_device,
            img.dev
        )

        cl.enqueue_copy(queue, periods, periods_device)

        return img.read(queue), periods

    def _compute_fast(self, queue, skip, iter, z0, c, tol, bounds, root_seq, img, seed=None):
        shape = img.shape

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)

        self.prg.fast_param_map(
            queue, shape, None,
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),
            # bounds
            numpy.array(bounds, dtype=real_type),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.uint64(seed if seed is not None else random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            periods_device,
            #
            img.dev
        )

        cl.enqueue_copy(queue, periods, periods_device)

        return img.read(queue), periods

    def compute(self, queue, img, skip, iter, z0, c, tol, bounds,
                root_seq=None, method="fast", scale_factor=None, seed=None):
        if method == "fast":
            if scale_factor is not None:
                pass  # TODO show warning that scale factor will be ignored
            return self._compute_fast(queue, skip, iter, z0, c, tol, bounds, root_seq, img, seed)
        if method == "precise":
            return self._compute_precise(queue, skip, iter, z0, c, tol, bounds, root_seq, scale_factor, img, seed)

        raise RuntimeError("Unknown method: '{}'".format(method))

    def compute_tiled(self, queue, img, full_size, skip, iter, z0, c, tol, bounds,
                      root_seq=None, method="fast", scale_factor=None, seed=None):
        if full_size[0] % img.shape[0] != 0 or full_size[0] < img.shape[0] \
                or full_size[1] % img.shape[1] != 0 or full_size[1] < img.shape[1]:
            raise NotImplementedError("full_size is not trivially covered by img.shape")

        xmin, xmax, ymin, ymax = bounds

        nx = full_size[0] // img.shape[0]
        ny = full_size[1] // img.shape[1]

        x_ticks = numpy.linspace(xmin, xmax, nx + 1 if nx != 1 else 0)
        y_ticks = numpy.linspace(ymin, ymax, ny + 1 if ny != 1 else 0)

        image = None
        periods = None
        pbar = tqdm(
            desc="col loop (col={}x{})".format(img.shape[0], full_size[1]),
            total=(len(x_ticks) - 1) * (len(y_ticks) - 1),
            ncols=120
        )
        for tile_x, x_min, x_max in zip(range(len(x_ticks)), x_ticks, x_ticks[1:]):
            col = []
            col_periods = []
            for tile_y, y_min, y_max in zip(range(len(y_ticks)), y_ticks, y_ticks[1:]):
                pbar.set_description("computing tile {:3d},{:3d} -- bounds {:+4.4f}:{:+4.4f}, {:+4.4f}:{:+4.4f}".format(
                    tile_x, tile_y, x_min, x_max, y_min, y_max
                ))
                res = self.compute(queue, img, skip, iter, z0, c, tol,
                                   (x_min, x_max, y_min, y_max),
                                   root_seq, method, scale_factor, seed)
                col.append(res[0].copy())
                col_periods.append(res[1].copy())

                pbar.update()
            col = numpy.vstack(col[::-1])
            col_periods = numpy.vstack(col_periods[::-1])

            if periods is None:
                periods = col_periods
            else:
                periods = numpy.hstack((periods, col_periods))

            if image is None:
                image = col
            else:
                image = numpy.hstack((image, col))

        pbar.close()

        return image, periods

    def compute_colors_for_periods(self, queue, periods, iter):
        periods_dev = copy_dev(self.ctx, periods)

        colors = numpy.empty((len(periods), 3), dtype=numpy.float32)
        colors_dev = alloc_like(self.ctx, colors)

        self.prg.get_color(
            queue, (len(periods),), None,
            numpy.int32(iter),
            periods_dev,
            colors_dev
        )

        cl.enqueue_copy(queue, colors, colors_dev)

        return colors
