import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class BasinsOfAttraction:

    def __init__(self, ctx, img):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")
        self.img = img

    def _compute_basins(self, queue, skip, h, alpha, c, bounds, root_seq, scale_factor, buf):
        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_basins(
            queue, (self.img.shape[0] // scale_factor, self.img.shape[1] // scale_factor), None,

            numpy.int32(skip),
            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),
            real_type(h),
            real_type(alpha),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,
            buf
        )

    def _render_basins(self, queue, bounds, scale_factor, method, img, buf, hostbuf):
        if method == "host":
            cl.enqueue_copy(queue, hostbuf, buf)

            unique_points = numpy.unique(hostbuf, axis=0)
            unique_points_dev = copy_dev(self.ctx, unique_points)

            print("Unique attraction points: {} / {}".format(unique_points.shape[0], hostbuf.shape[0]))

            self.prg.draw_basins_colored(
                queue, self.img.shape, None,
                numpy.int32(scale_factor),
                numpy.int32(unique_points.shape[0]),
                unique_points_dev,
                buf,
                self.img.dev
            )
        elif method == "dev":
            self.prg.draw_basins(
                queue, self.img.shape, None,
                numpy.int32(scale_factor),
                numpy.array(bounds, dtype=real_type),
                buf,
                self.img.dev
            )
        else:
            raise ValueError("Unknown algo: \"{}\"".format(method))

        return img.read(queue)

    def compute(self, queue, skip, h, alpha, c, bounds, root_seq,
                method="dev", scale_factor=1, img=None):
        if img is None:
            img = self.img

        basin_points = numpy.empty((numpy.prod(img.shape), 2), dtype=real_type)
        basin_points_dev = alloc_like(self.ctx, basin_points)

        self._compute_basins(queue, skip, h, alpha, c, bounds, root_seq, scale_factor, basin_points_dev)

        return self._render_basins(queue, bounds, scale_factor, method, img, basin_points_dev, basin_points)
