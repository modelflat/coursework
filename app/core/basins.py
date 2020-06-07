import os
import time
import warnings

from collections import defaultdict

import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev, CLImg, clear_image


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")
        self.points_dev = None
        self.periods = None
        self.periods_dev = None
        self.sequence_hashes_dev = None
        self.table_size = None
        self.table_dev = None
        self.table_data_dev = None

    def _maybe_allocate_buffers(self, shape, iter):
        sample = real_type(0)
        new_size = 2 * numpy.prod(shape) * iter * sample.nbytes
        if self.points_dev is None or new_size > self.points_dev.size:
            self.points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=new_size)

        if self.periods is None or numpy.prod(self.periods.shape) != numpy.prod(shape):
            self.periods = numpy.empty(shape, dtype=numpy.int32)
            self.periods_dev = alloc_like(self.ctx, self.periods)

    def _maybe_allocate_hash_table_buffers(self, queue, shape, table_size):
        new_size = numpy.uint32(0).nbytes * numpy.prod(shape)
        if self.sequence_hashes_dev is None or new_size > self.sequence_hashes_dev.size:
            self.sequence_hashes_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=new_size)

        if self.table_dev is None or self.table_size != table_size:
            self.table_size = table_size
            new_size = numpy.uint32(0).nbytes * table_size
            self.table_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=new_size)
            self.table_data_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=new_size)

        cl.enqueue_fill_buffer(queue, self.table_dev, numpy.uint32(0), 0, self.table_dev.size)
        cl.enqueue_fill_buffer(queue, self.table_data_dev, numpy.uint32(0), 0, self.table_data_dev.size)

    def _find_attractors(self, queue, shape, iter, tol, check_collisions=False, table_size=None):
        assert iter <= 64, "max_iter is currently static and is 64"

        n_points = numpy.prod(shape)
        table_size = table_size or (n_points * 2 - 1)

        assert table_size < 2 ** 32, "table size must not exceed max int32 value"

        self._maybe_allocate_hash_table_buffers(queue, shape, table_size)

        queue.finish()

        period_counts = numpy.zeros((iter,), dtype=numpy.uint32)
        period_counts_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=period_counts)

        self.prg.round_points(
            queue, (n_points * iter,), None,
            real_type(tol),
            self.points_dev
        )

        self.prg.rotate_sequences(
            queue, (n_points,), None,
            numpy.int32(iter),
            real_type(tol),
            self.periods_dev,
            self.points_dev
        )

        self.prg.hash_sequences(
            queue, (n_points,), None,
            numpy.uint32(iter),
            numpy.uint32(table_size),
            self.periods_dev,
            self.points_dev,
            self.sequence_hashes_dev
        )

        self.prg.count_unique_sequences(
            queue, (n_points,), None,
            numpy.uint32(iter),
            numpy.uint32(table_size),
            self.periods_dev,
            self.sequence_hashes_dev,
            self.table_dev,
            self.table_data_dev
        )

        if check_collisions:
            collisions = numpy.zeros((1,), dtype=numpy.uint32)
            collisions_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=collisions)

            self.prg.check_collisions(
                queue, (n_points,), None,
                real_type(tol),
                numpy.uint32(iter),
                numpy.uint32(table_size),
                self.periods_dev,
                self.sequence_hashes_dev,
                self.points_dev,
                self.table_dev,
                self.table_data_dev,
                collisions_dev
            )

            cl.enqueue_copy(queue, collisions, collisions_dev)
            n_collisions = int(collisions[0])

            if n_collisions != 0:
                warnings.warn(
                    "Hash collisions were detected while computing basins of attraction.\n"
                    "This may result in non-deterministic results and/or incorrect counts "
                    "of sequences of some of the periods.\nIn some cases, setting `table_size` "
                    f"manually might help. Current `table_size` is {table_size}."
                )
        else:
            n_collisions = None

        self.prg.count_periods_of_unique_sequences(
            queue, (table_size,), None,
            self.periods_dev,
            self.table_dev,
            self.table_data_dev,
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
        if total_points_count == 0:
            # no attractors were found at all
            return [], n_collisions

        unique_sequences = numpy.empty((total_points_count, 2), dtype=real_type)
        unique_sequences_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences.nbytes)

        # hashes and counts
        unique_sequences_info = numpy.empty((sum(period_counts[:-1]), 2), dtype=numpy.uint32)
        unique_sequences_info_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=unique_sequences_info.nbytes)

        self.prg.gather_unique_sequences(
            queue, (table_size,), None,
            numpy.uint32(iter),
            self.periods_dev,
            self.points_dev,
            self.table_dev,
            self.table_data_dev,
            period_counts_dev,
            hash_positions_dev,
            sequence_positions_dev,
            current_positions_dev,
            unique_sequences_dev,
            unique_sequences_info_dev
        )

        cl.enqueue_copy(queue, unique_sequences, unique_sequences_dev)
        cl.enqueue_copy(queue, unique_sequences_info, unique_sequences_info_dev)

        raw_attractors = []
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

            raw_attractors.append((int(period), sequences, *info.T))

        return raw_attractors, n_collisions

    def _color_attractors(self, queue, img, attractors, color_fn):
        # TODO this is also very-very slow!
        t = time.perf_counter()
        attractors.sort(key=lambda a: a["hash"])

        hashes = numpy.array([attractor["hash"] for attractor in attractors], dtype=numpy.uint32)
        colors = numpy.array([color_fn(attractor) for attractor in attractors], dtype=numpy.float32)

        t = time.perf_counter() - t
        # print(f"coloring prep took {t:.3f} s")

        # TODO we can avoid searching inside this kernel if we use PHF
        hashes_dev = copy_dev(self.ctx, hashes)
        colors_dev = copy_dev(self.ctx, colors)

        self.prg.color_attractors(
            queue, img.shape, None,
            numpy.int32(len(attractors)),
            hashes_dev,
            colors_dev,
            self.sequence_hashes_dev,
            img.dev
        )

        return img.read(queue)

    def compute_attractors(self, queue, shape, skip, iter, h, alpha, c, bounds,
                           root_seq=None, tolerance_decimals=3, seed=None, threshold=0):
        self._maybe_allocate_buffers(shape, iter)

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
            self.points_dev,
            self.periods_dev
        )

        queue.finish()

        t = time.perf_counter()
        raw_attractors, n_collisions = self._find_attractors(
            queue, shape, iter, 1 / 10 ** tolerance_decimals,
            check_collisions=False, table_size=None
        )
        t = time.perf_counter() - t
        # print(f"finding attractors took {t:.3f} s")

        attractors = []

        # TODO this is too slow!
        t = time.perf_counter()

        for period, sequences, hashes, counts in raw_attractors:
            mask = counts > threshold
            sequences = sequences[mask]
            hashes = hashes[mask]
            counts = counts[mask]
            for sequence, hash, count in zip(sequences, hashes, counts):
                attractors.append({
                    "attractor": sequence, "hash": hash, "occurences": count, "period": period
                })
            
        t = time.perf_counter() - t
        # print(f"converting attractors took {t:.3f} s, found {len(attractors)} attractors")

        return attractors, n_collisions

    def compute_periods(self, queue, img, skip, iter, h, alpha, c, bounds,
                        root_seq=None, tolerance_decimals=3, seed=None):
        self._maybe_allocate_buffers(img.shape, 1)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.capture_points(
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

            self.points_dev,
            self.periods_dev
        )

        self.prg.color_basins_periods(
            queue, img.shape, None,
            numpy.int32(iter),
            self.periods_dev,
            img.dev,
        )

        return img.read(queue)

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds,
                root_seq=None, tolerance_decimals=3, seed=None, method="basins", 
                color_init=None, color_fn=None, threshold=0):
        # if method == "periods":
        #     return self.compute_periods(queue, img, skip, iter, h, alpha, c, bounds,
        #                                 root_seq, tolerance_decimals, seed)
        if method == "basins":
            if color_fn is None:
                raise ValueError("color_fn should be set for method = 'basins'")
            attractors, n_collisions = self.compute_attractors(
                queue, img.shape, skip, iter, h, alpha, c, bounds, root_seq, tolerance_decimals, seed, threshold
            )

            if color_init is not None:
                color_init(attractors, n_collisions)

            if attractors:
                return attractors, self._color_attractors(queue, img, attractors, color_fn)
            else:
                clear_image(queue, img.dev, img.shape, color=(0.0, 0.0, 0.0, 1.0))
                return [], img.read(queue)
        raise ValueError("Unknown method: \"{}\"".format(method))


def _format_attractor(attractor, format):
    return " ".join(((f"%{format},%{format}" % (re, im)) for (re, im) in attractor.tolist()))


def analyze_attractors(basins, queue, img, skip, iter, h, alpha, c, bounds, root_seq=None, 
                       tolerance_decimals=3, seed=None, threshold=0, color_fn=None, norm=None):
    z = list(range(1, iter))
    base_colors = (numpy.array(z) - 1) * (300 / z[-1])

    attractors = defaultdict(lambda: defaultdict(lambda: dict()))

    current_colors = dict()
    
    def _color_init(attractors_input, n_collisions):
        nonlocal attractors
        nonlocal current_colors
        for attractor in attractors_input:
            period = attractor["period"]

            is_significant = 0.05 * numpy.prod(img.shape) < attractor["occurences"]
            if is_significant:
                # we should give it brighter color
                color = (base_colors[period - 1], 1, 1)
            else:
                # we should give it darker color
                color = (base_colors[period - 1], 0.5, 0.5)
            current_colors[attractor["hash"]] = color
            
        attractors = attractors_input

    def _color_fn(attractor):
        if color_fn is not None:
            color = color_fn(_format_attractor(attractor["attractor"], f"+.{tolerance_decimals}f"), attractor)
            if color is not None:
                return color
        return (0, 0, 0)

    image = basins.compute(
        queue, img, skip, iter, h, alpha, c, bounds, 
        root_seq, tolerance_decimals, seed, "basins", _color_init, _color_fn, threshold
    )

    attractors_index = dict()

    for attractor in attractors:
        period = f"{int(attractor['period']):02d}"
        if not period in attractors_index:
            attractors_index[period] = dict()
        
        stringified = _format_attractor(attractor["attractor"], f"+.{tolerance_decimals}f")
        if not stringified in attractors_index[period]:
            attractors_index[period][stringified] = dict()
        
        attractor_info = attractors_index[period][stringified]
        attractor_info["count"] = attractor_info.get("count", 0) + int(attractor["occurences"])
        attractor_info["attractor"] = attractor["attractor"]

        attractor.pop("hash")

    return attractors_index, attractors, image

