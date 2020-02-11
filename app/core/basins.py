import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds,
                root_seq=None, method="sections", scale_factor=1, tolerance=1e-4, seed=None):
        points = numpy.empty((numpy.prod(img.shape), 2), dtype=real_type)
        points_dev = alloc_like(self.ctx, points)

        periods_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4 * points.shape[0])

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.capture_points(
            queue, img.shape, None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),

            real_type(h),
            real_type(alpha),
            real_type(tolerance),

            numpy.uint64(seed if seed is not None else random_seed()),
            numpy.int32(seq_size),
            seq,

            points_dev,
            periods_dev
        )

        if method == "sections":
            self.prg.color_basins_section(
                queue, img.shape, None,
                numpy.int32(scale_factor),
                numpy.array(bounds, dtype=real_type),
                points_dev,
                img.dev
            )
        elif method == "periods":
            self.prg.color_basins_periods(
                queue, img.shape, None,
                numpy.int32(iter),
                periods_dev,
                img.dev,
            )
        else:
            raise ValueError("Unknown method: \"{}\"".format(method))

        return img.read(queue)
