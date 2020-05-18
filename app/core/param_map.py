import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class ParameterMap:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "param_map.cl")
        self.map_points = None

    def compute(self, queue, img, skip, iter, z0, c, tol, bounds, root_seq=None, seed=None):
        shape = img.shape

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)
        self.prg.param_map(
            queue, shape, None,
            # bounds
            numpy.array(bounds, dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),
            # tol
            real_type(tol),
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
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

    def compute_incremental(self, queue, img, skip, skip_batch_size, iter, z0, c, bounds,
                            root_seq, tolerance_decimals=3, seed=None,
                            draw_image=False, capture_points=True):
        shape = img.shape

        z0_dev = numpy.array((z0.real, z0.imag), dtype=real_type)
        c_dev = numpy.array((c.real, c.imag), dtype=real_type)
        bounds_dev = numpy.array(bounds, dtype=real_type)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)
        seq_size = numpy.int32(seq_size)

        seq_pos_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * numpy.prod(shape))
        rng_state_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * numpy.prod(shape))
        _sample = real_type(0)
        points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=2 * _sample.nbytes * numpy.prod(shape))
        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * numpy.prod(shape))
        if capture_points:
            final_points = numpy.empty((*shape, iter, 2), dtype=real_type)
        else:
            final_points = numpy.empty((1,), dtype=real_type)

        final_points_dev = alloc_like(self.ctx, final_points)

        self.prg.incremental_param_map_init(
            queue, shape, None,
            bounds_dev, c_dev, z0_dev,
            numpy.uint64(seed if seed is not None else random_seed()),
            seq_size, seq,
            seq_pos_dev, rng_state_dev, points_dev
        )
        queue.finish()

        skip_batch_dev = numpy.int32(skip_batch_size)
        for _ in range(skip // skip_batch_size):
            self.prg.incremental_param_map_next(
                queue, shape, None,
                bounds_dev, c_dev,
                skip_batch_dev,
                seq_size, seq,
                seq_pos_dev, rng_state_dev, points_dev
            )
            queue.finish()

        self.prg.incremental_param_map_finalize(
            queue, shape, None,
            bounds_dev, c_dev,
            real_type(1 / 10 ** tolerance_decimals),
            numpy.int32(skip % skip_batch_size),
            numpy.int32(iter),
            numpy.int32(1 if capture_points else 0),
            seq_size, seq,
            seq_pos_dev, rng_state_dev, points_dev,
            final_points_dev, periods_dev
        )

        cl.enqueue_copy(queue, final_points, final_points_dev)
        cl.enqueue_copy(queue, periods, periods_dev)

        if draw_image:
            self.prg.color_param_map(
                queue, shape, None,
                numpy.int32(iter),
                periods_dev,
                img.dev
            )
            queue.finish()
            img.read(queue)

        return periods, final_points

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
