import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev

complex_t = numpy.complex64 if real_type == numpy.float32 else numpy.complex128


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")
        self.points = None
        self.points_dev = None
        self.periods = None
        self.periods_dev = None

    def _maybe_allocate_buffers(self, shape):
        new = numpy.prod(shape) * 2
        if self.points is None or new != numpy.prod(self.points.shape) * 2:
            self.points = numpy.empty((numpy.prod(shape), 2), dtype=real_type)
            self.points_dev = alloc_like(self.ctx, self.points)
            self.periods = numpy.empty(shape, dtype=numpy.int32)
            self.periods_dev = alloc_like(self.ctx, self.periods)

    def _compute_points(self, queue, shape, skip, iter, h, alpha, c, bounds,
                        root_seq=None, tolerance_decimals=3, seed=None):
        self._maybe_allocate_buffers(shape)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.capture_points(
            queue, shape, None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),

            real_type(h),
            real_type(alpha),
            real_type(1 / 10 ** tolerance_decimals),

            numpy.uint64(seed if seed is not None else random_seed()),
            numpy.int32(seq_size),
            seq,

            self.points_dev,
            self.periods_dev
        )

    def _find_attractors(self, queue, shape, iter: int, periods: cl.Buffer, points: cl.Buffer, tol):
        assert iter <= 64, "max_iter is  currently static and is 64"

        period_counts = numpy.zeros((iter,), dtype=numpy.int32)
        period_counts_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=period_counts)

        n_points = numpy.prod(shape)

        self.prg.count_periods(
            queue, (n_points,), None,
            periods,
            period_counts_dev
        )

        cl.enqueue_copy(queue, period_counts, period_counts_dev)

        self.prg.rotate_captured_sequences(
            queue, (n_points,), None,
            numpy.int32(iter),
            real_type(tol),
            periods,
            points,
        )

        volume_per_period = numpy.arange(1, iter) * period_counts[:-1]
        seq_positions = numpy.concatenate(
            (
                # shift array by one position to get exclusive cumsum
                numpy.zeros((1,), dtype=numpy.int64),
                numpy.cumsum(volume_per_period, axis=0, dtype=numpy.int64)[:-1]
            )
        )
        seq_positions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=seq_positions)

        current_positions = numpy.zeros_like(seq_positions, dtype=numpy.int32)
        current_positions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=current_positions)

        total_points_count = sum(numpy.arange(1, iter) * period_counts[:-1])
        new_points = numpy.empty((total_points_count, 2), dtype=real_type)
        new_points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=new_points.nbytes)

        self.prg.align_rotated_sequences(
            queue, (n_points,), None,
            numpy.int32(iter),
            real_type(tol),
            periods,
            seq_positions_dev,
            points,
            current_positions_dev,
            new_points_dev
        )

        cl.enqueue_copy(queue, new_points, new_points_dev)

        result = {}
        for period in range(1, iter):
            points_in_period = period_counts[period - 1]
            if points_in_period == 0:
                continue
            start = 0 if period == 1 else seq_positions[period - 1]
            end = start + period * points_in_period
            data = new_points[start:end].reshape((points_in_period, period, 2))
            unique_data = numpy.unique(data, axis=0, return_counts=True)
            result[period] = unique_data

        return result

    def compute_and_color_known(self, queue, img, skip, iter, h, alpha, c, bounds, attractors, colors,
                                root_seq=None, tolerance_decimals=3, seed=None):

        points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * 2 * iter * numpy.prod(img.shape))
        periods_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * numpy.prod(img.shape))

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.capture_points_iter(
            queue, img.shape, None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),

            real_type(h),
            real_type(alpha),
            real_type(1 / 10 ** tolerance_decimals),

            numpy.uint64(seed if seed is not None else random_seed()),
            numpy.int32(seq_size),
            seq,

            points_dev,
            periods_dev
        )

        return self.color_known_attractors(queue, img, iter, attractors, colors, points_dev, periods_dev)

    def compute_periods(self, queue, img, skip, iter, h, alpha, c, bounds,
                        root_seq=None, tolerance_decimals=3, seed=None):
        self._compute_points(
            queue, img.shape, skip, iter, h, alpha, c, bounds, root_seq, tolerance_decimals, seed
        )

        self.prg.color_basins_periods(
            queue, img.shape, None,
            numpy.int32(iter),
            self.periods_dev,
            img.dev,
        )

        return img.read(queue)

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds,
                root_seq=None, method="periods", tolerance_decimals=3, seed=None):
        if method == "periods":
            return self.compute_periods(queue, img, skip, iter, h, alpha, c, bounds,
                                        root_seq, tolerance_decimals, seed)
        raise ValueError("Unknown method: \"{}\"".format(method))

    def color_known_attractors(self, queue, img, iter, attractors, colors, points, periods):
        assert len(attractors) <= len(colors)

        attractors_dev = copy_dev(self.ctx, numpy.array(attractors, dtype=real_type))
        colors_dev = copy_dev(self.ctx, numpy.array(colors, dtype=numpy.float32))

        if isinstance(points, cl.Buffer):
            points_dev = points
        else:
            points_dev = copy_dev(self.ctx, points)

        if isinstance(periods, cl.Buffer):
            periods_dev = periods
        else:
            periods_dev = copy_dev(self.ctx, periods)

        self.prg.color_known_attractors(
            queue, img.shape, None,
            numpy.int32(iter),
            numpy.int32(len(attractors)),
            colors_dev,
            attractors_dev,
            points_dev,
            periods_dev,
            img.dev
        )

        return img.read(queue)

    def find_attractors(self, queue, shape, skip, iter, h, alpha, c, bounds,
                        root_seq=None, tolerance_decimals=3, seed=None):
        _sample = real_type(0)
        points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=2 * _sample.nbytes * numpy.prod(shape) * iter)
        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_dev = alloc_like(self.ctx, periods)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.capture_points_iter(
            queue, shape, None,

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.array(bounds, dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),

            real_type(h),
            real_type(alpha),
            real_type(1 / 10 ** tolerance_decimals),

            numpy.uint64(seed if seed is not None else random_seed()),
            numpy.int32(seq_size),
            seq,

            points_dev,
            periods_dev
        )

        return self._find_attractors(
            queue, shape, iter, periods_dev, points_dev, 1 / 10 ** tolerance_decimals
        )
