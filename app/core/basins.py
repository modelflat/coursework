import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")

    def compute(self, queue, img, skip, h, alpha, c, bounds,
                root_seq=None, method="dev", scale_factor=1):
        points = numpy.empty((numpy.prod(img.shape), 2), dtype=real_type)
        points_dev = alloc_like(self.ctx, points)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_basins(
            queue, (img.shape[0] // scale_factor, img.shape[1] // scale_factor), None,

            numpy.int32(skip),
            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),
            real_type(h),
            real_type(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,
            points_dev
        )

        if method == "host":
            cl.enqueue_copy(queue, points, points_dev)

            unique_points = numpy.unique(points, axis=0)
            unique_points_dev = copy_dev(self.ctx, unique_points)

            print("Unique attraction points: {} / {}".format(unique_points.shape[0], points.shape[0]))

            self.prg.draw_basins_colored(
                queue, img.shape, None,
                numpy.int32(scale_factor),
                numpy.int32(unique_points.shape[0]),
                unique_points_dev,
                points_dev,
                img.dev
            )
        elif method == "dev":
            self.prg.draw_basins(
                queue, img.shape, None,
                numpy.int32(scale_factor),
                numpy.array(bounds, dtype=real_type),
                points_dev,
                img.dev
            )
        else:
            raise ValueError("Unknown method: \"{}\"".format(method))

        return img.read(queue)
