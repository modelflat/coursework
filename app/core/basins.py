import time
import warnings

import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev, CLImg


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

    def _find_attractors(self, queue, shape, iter: int, periods: cl.Buffer, points: cl.Buffer, tol,
                         check_collisions=False, table_size=None):
        assert iter <= 64, "max_iter is currently static and is 64"

        n_points = numpy.prod(shape)

        table_size = table_size or (n_points * 2 - 1)
        assert table_size < 2 ** 32, "table size must not exceed max int32 value"

        # TODO avoid allocation of unnecessary host buffers?

        sequence_hashes = numpy.empty(shape, dtype=numpy.uint32)
        sequence_hashes_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=sequence_hashes.nbytes)

        table = numpy.zeros((table_size,), dtype=numpy.uint32)
        table_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=table)

        table_data = numpy.zeros((table_size,), dtype=numpy.uint32)
        table_data_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=table_data)

        period_counts = numpy.zeros((iter,), dtype=numpy.uint32)
        period_counts_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=period_counts)

        self.prg.round_points(
            queue, (n_points * iter,), None,
            real_type(tol),
            points
        )

        self.prg.rotate_sequences(
            queue, (n_points,), None,
            numpy.int32(iter),
            real_type(tol),
            periods,
            points,
        )

        self.prg.hash_sequences(
            queue, (n_points,), None,
            numpy.uint32(iter),
            numpy.uint32(table_size),
            periods,
            points,
            sequence_hashes_dev
        )

        self.prg.count_unique_sequences(
            queue, (n_points,), None,
            numpy.uint32(iter),
            numpy.uint32(table_size),
            periods,
            sequence_hashes_dev,
            table_dev,
            table_data_dev
        )

        if check_collisions:
            collisions = numpy.zeros((1,), dtype=numpy.uint32)
            collisions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=collisions)

            self.prg.check_collisions(
                queue, (n_points,), None,
                real_type(tol),
                numpy.uint32(iter),
                numpy.uint32(table_size),
                periods,
                sequence_hashes_dev,
                points,
                table_dev,
                table_data_dev,
                collisions_dev
            )

            cl.enqueue_copy(queue, collisions, collisions_dev)
            n_collisions, = collisions

            if n_collisions != 0:
                warnings.warn(
                    "Hash collisions were detected while computing basins of attraction.\n"
                    "This may result in non-deterministic results and/or incorrect counts "
                    "of sequences of some of the periods.\nIn some cases, setting `table_size` "
                    f"manually might help. Current `table_size` is {table_size}."
                )

        self.prg.count_periods_of_unique_sequences(
            queue, (table_size,), None,
            periods,
            table_dev,
            table_data_dev,
            period_counts_dev
        )

        cl.enqueue_copy(queue, period_counts, period_counts_dev)

        hash_positions = numpy.concatenate(
            (
                # shift array by one position to get exclusive cumsum
                numpy.zeros((1,), dtype=numpy.uint32),
                numpy.cumsum(period_counts, axis=0, dtype=numpy.uint32)[:-1]
            )
        )
        hash_positions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hash_positions)

        volume_per_period = numpy.arange(1, iter) * period_counts[:-1]
        sequence_positions = numpy.concatenate(
            (
                numpy.zeros((1,), dtype=numpy.uint32),
                numpy.cumsum(volume_per_period, axis=0, dtype=numpy.uint32)[:-1]
            )
        )
        sequence_positions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sequence_positions)

        current_positions = numpy.zeros((iter,), dtype=numpy.uint32)
        current_positions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_positions)

        total_points_count = sum(numpy.arange(1, iter) * period_counts[:-1])

        unique_sequences = numpy.empty((total_points_count, 2), dtype=real_type)
        unique_sequences_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences.nbytes)

        # hashes and counts
        unique_sequences_info = numpy.empty((sum(period_counts[:-1]), 2), dtype=numpy.uint32)
        unique_sequences_info_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences_info.nbytes)

        self.prg.gather_unique_sequences(
            queue, (table_size,), None,
            numpy.uint32(iter),
            periods,
            points,
            table_dev,
            table_data_dev,
            period_counts_dev,
            hash_positions_dev,
            sequence_positions_dev,
            current_positions_dev,
            unique_sequences_dev,
            unique_sequences_info_dev
        )

        cl.enqueue_copy(queue, unique_sequences, unique_sequences_dev)
        cl.enqueue_copy(queue, unique_sequences_info, unique_sequences_info_dev)

        attractors = []
        current_pos = 0
        current_pos_info = 0
        for period, n_sequences in enumerate(period_counts[:-1], start=1):
            if n_sequences == 0:
                continue
            sequences = unique_sequences[current_pos:current_pos + n_sequences * period] \
                .reshape((n_sequences, period, 2))
            current_pos += n_sequences * period
            info = unique_sequences_info[current_pos_info:current_pos_info + n_sequences] \
                .reshape((n_sequences, 2))
            current_pos_info += n_sequences

            for sequence, (sequence_hash, count) in zip(sequences, info):
                attractors.append({
                    "attractor": list(map(tuple, sequence)),
                    "period": int(period),
                    "hash": int(sequence_hash),
                    "occurences": int(count)
                })

        return attractors, sequence_hashes_dev

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
                        root_seq=None, tolerance_decimals=3, seed=None, color_dict=None, img=None):
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

        queue.finish()

        attractors, sequence_hashes_dev = self._find_attractors(
            queue, shape, iter, periods_dev, points_dev, 1 / 10 ** tolerance_decimals,
            check_collisions=True, table_size=None
        )

        if color_dict is not None and img is not None:
            # colors = {<hash>: <color>}
            # color = (H, S, V) // float32
            hashes = numpy.empty((len(attractors),), dtype=numpy.uint32)
            colors = numpy.empty((len(attractors), 3), dtype=numpy.float32)

            for i, attractor in enumerate(attractors):
                hashes[i] = attractor["hash"]
                colors[i] = color_dict.get(attractor["hash"], (0, 0, 0))

            hashes_dev = copy_dev(self.ctx, hashes)
            colors_dev = copy_dev(self.ctx, colors)

            self.prg.color_attractors(
                queue, shape, None,
                numpy.int32(len(attractors)),
                hashes_dev,
                colors_dev,
                sequence_hashes_dev,
                img.dev
            )

            img.read(queue)

        return attractors
