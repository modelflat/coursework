import numpy

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, real_type


class PhasePlot:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "phase_plot.cl",
                                           options=("-DPOINT_RADIUS=0", ))

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds, grid_size,
                z0=None, root_seq=None, clear=True, seed=None):
        if clear:
            img.clear(queue)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.newton_fractal(
            queue, (grid_size, grid_size) if z0 is None else (1, 1), None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=real_type),

            numpy.array((c.real, c.imag), dtype=real_type),
            real_type(h),
            real_type(alpha),

            numpy.uint64(seed if seed is not None else random_seed()),

            numpy.int32(seq_size),
            seq,

            numpy.int32(1 if z0 is not None else 0),
            numpy.array((0, 0) if z0 is None else (z0.real, z0.imag), dtype=real_type),

            img.dev
        )

        return img.read(queue)
