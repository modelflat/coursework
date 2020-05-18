import time

from collections import defaultdict

import numpy
import pyopencl as cl
import cv2

from tqdm import tqdm

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


complex_t = numpy.complex64 if real_type == numpy.float32 else numpy.complex128


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, ("basins.cl", "basins_incremental.cl"))
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

    def deep_capture(self, queue, shape, skip, skip_batch_size, iter, h, alpha, c, bounds,
                     root_seq=None, tolerance_decimals=3, seed=None, show_progress=True):
        h_dev = real_type(h)
        alpha_dev = real_type(alpha)
        c_dev = numpy.array((c.real, c.imag), dtype=real_type)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)
        seq_size = numpy.int32(seq_size)

        seq_pos_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * numpy.prod(shape))
        rng_state_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * numpy.prod(shape))
        _sample = real_type(0)
        points_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=2 * _sample.nbytes * numpy.prod(shape))
        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_dev = alloc_like(self.ctx, periods)
        final_points = numpy.empty((*shape, iter, 2), dtype=real_type)
        final_points_dev = alloc_like(self.ctx, final_points)

        self.prg.capture_points_init(
            queue, shape, None,
            numpy.array(bounds, dtype=real_type),
            c_dev, h_dev, alpha_dev,
            numpy.uint64(seed if seed is not None else random_seed()),
            seq_size, seq,
            seq_pos_dev, rng_state_dev, points_dev
        )

        skip_batch_dev = numpy.int32(skip_batch_size)
        it = range(skip // skip_batch_size)
        if show_progress:
            it = tqdm(it)
        for _ in it:
            self.prg.capture_points_next(
                queue, shape, None,
                skip_batch_dev,
                c_dev, h_dev, alpha_dev,
                seq_size, seq,
                seq_pos_dev, rng_state_dev, points_dev
            )
            queue.finish()

        self.prg.capture_points_finalize(
            queue, shape, None,
            numpy.int32(skip % skip_batch_size),
            numpy.int32(iter),
            c_dev, h_dev, alpha_dev,
            real_type(1 / 10 ** tolerance_decimals),
            seq_size, seq,
            seq_pos_dev, rng_state_dev, points_dev,
            final_points_dev, periods_dev
        )

        cl.enqueue_copy(queue, final_points, final_points_dev)
        cl.enqueue_copy(queue, periods, periods_dev)

        return final_points, periods

    def compute_sections(self, queue, img, skip, iter, h, alpha, c, bounds,
                         root_seq=None, tolerance_decimals=3, seed=None):
        self._compute_points(
            queue, img.shape, skip, iter, h, alpha, c, bounds, root_seq, tolerance_decimals, seed
        )

        self.prg.color_basins_section(
            queue, img.shape, None,
            numpy.array(bounds, dtype=real_type),
            self.points_dev,
            img.dev
        )

        return img.read(queue)

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

    def compute_periods_and_attractors(self, queue, img, skip, iter, h, alpha, c, bounds,
                                       root_seq=None, tolerance_decimals=3, seed=None, verbose=False,
                                       compute_image=True, only_good_attractors=True):
        self._compute_points(
            queue, img.shape, skip, iter, h, alpha, c, bounds, root_seq, tolerance_decimals, seed
        )

        contours = numpy.empty(img.shape, dtype=numpy.uint8)
        contours_dev = alloc_like(self.ctx, contours)

        self.prg.color_basins_periods_attractors(
            queue, img.shape, None,
            numpy.int32(iter),
            real_type(1 / 10 ** tolerance_decimals),
            numpy.array(bounds, dtype=real_type),
            self.periods_dev,
            self.points_dev,
            contours_dev,
            img.dev,
        )

        cl.enqueue_copy(queue, contours, contours_dev)

        min_std = 1 / 10 ** (tolerance_decimals * 2)
        min_points_in_region = 16

        # note: we can also use OpenCV/OpenCL interop here
        n_labels, labels = cv2.connectedComponents(contours)
        labels_dev = copy_dev(self.ctx, labels)

        t = time.perf_counter()

        label_count = numpy.zeros((n_labels,), dtype=numpy.int32)
        label_count_dev = copy_dev(self.ctx, label_count)
        label_end_indexes_dev = alloc_like(self.ctx, label_count)
        points_sorted_dev = alloc_like(self.ctx, self.points)

        self.prg.prepare_region(
            queue, img.shape, None,
            labels_dev, label_count_dev
        )
        cl.enqueue_copy(queue, label_count, label_count_dev)

        label_end_indexes = numpy.cumsum(label_count, dtype=numpy.int32)
        cl.enqueue_copy(queue, label_end_indexes_dev, label_end_indexes)

        self.prg.sort_points_by_region(
            queue, img.shape, None,
            self.points_dev, labels_dev,
            label_end_indexes_dev, label_count_dev,
            points_sorted_dev
        )
        cl.enqueue_copy(queue, self.points, points_sorted_dev)

        points_view = self.points.view(complex_t).reshape((numpy.prod(img.shape),))

        suspicious_regions = defaultdict(lambda: list())
        attractors = defaultdict(lambda: list())
        skipped_labels = set()
        suspicious_labels = set()

        # NOTE: we skip label 0 here, as it will be for contours (brightest part of the image)
        # contour color = 255 (GRAY)
        for i, label_end in enumerate(label_end_indexes):
            if i == 0:
                if verbose:
                    print("Skipping label 0...")
                continue
            else:
                points_in_component = points_view[label_end_indexes[i - 1]:label_end]

            if len(points_in_component) < min_points_in_region:
                if verbose and len(points_in_component) == 0:
                    print(f'Only contours detected for label {i}')
                elif verbose:
                    print(f'Skipping small region {i}: '
                          f'{len(points_in_component)} < {min_points_in_region}')
                skipped_labels.add(i)
                continue

            std = points_in_component.std()

            if std < min_std:
                # can safely assume that this region has converged to an attractor
                mean = numpy.round(numpy.mean(points_in_component), tolerance_decimals)
                attractors[mean].append((i, len(points_in_component)))
                if verbose:
                    print(f'{i:4}: {len(points_in_component):6} converged  / {mean}')
            elif not only_good_attractors:
                # this region has varying values, we need to track this fact
                mn = numpy.round(points_in_component.min(), tolerance_decimals)
                mx = numpy.round(points_in_component.max(), tolerance_decimals)
                points_in_component = numpy.round(points_in_component, tolerance_decimals)
                elements, counts = numpy.unique(points_in_component, return_counts=True)
                most_common_index = numpy.argmax(counts)
                most_common = elements[most_common_index]
                if counts[most_common_index] / len(points_in_component) > 0.75:
                    suspicious_regions[(mn, most_common, mx)].append((i, len(points_in_component)))
                else:
                    weighted_mean = numpy.round(
                        numpy.average(elements, weights=counts/len(points_in_component)),
                        tolerance_decimals)
                    suspicious_regions[(mn, weighted_mean, mx)].append((i, len(points_in_component)))

                suspicious_labels.add(i)
                if verbose:
                    print(f'{i:4}: {len(points_in_component):6} susp. / {mn} : {mx} / {len(elements)} distinct')
        t = time.perf_counter() - t
        if verbose:
            print(f'analysis: {t:.3f} s')

        if not only_good_attractors:
            # todo add more options
            # lets now try to remove some of the suspicious regions based on the fact that they might
            # failed to converge to actual attractors solely due to low iteration effort
            probable_attractors = {
                (mn, val, mx): min(
                    (k for k in attractors.keys() if mn < k < mx),
                    key=lambda attr: abs(val - attr),
                    default=None
                )
                for (mn, val, mx) in suspicious_regions.keys()
            }

            for k, v in probable_attractors.items():
                if v is not None:
                    attractors[v].extend(suspicious_regions.pop(k))

            for k in suspicious_regions.keys():
                attractors[k].extend(suspicious_regions[k])

            if not attractors:
                print('No attractors detected!')
                attractors[complex(numpy.nan)] = [0]

        if verbose:
            print("A:", " ".join(str(attr) for attr in attractors.keys() if not isinstance(attr, tuple)))

        if compute_image:
            attractors_merged = {
                (attr if isinstance(attr, tuple) else (attr, attr, attr)): v
                for attr, v in attractors.items()
            }

            inv_attr_to_label = {
                v[0]: k
                for k, vv in attractors_merged.items()
                for v in vv
            }

            colors_dict = {
                attr: (240 * i / len(attractors_merged), 1, 1)
                for i, attr in enumerate(sorted(
                    attractors_merged.keys(), key=lambda x: x[1] if isinstance(x, tuple) else x
                ), 1)
            }

            flags = numpy.array(
                [
                    # is suspicious?
                    (0b01 if i in suspicious_labels else 0)
                    # is attractor?
                    | (0b10 if i in inv_attr_to_label else 0)
                    for i in range(n_labels)
                ],
                dtype=numpy.int32
            )
            flags_dev = copy_dev(self.ctx, flags)

            colors = numpy.array(
                [
                    col
                    for col, i in (
                        (colors_dict[inv_attr_to_label[i]] if i in inv_attr_to_label else (0, 0, 0), i)
                        for i in range(n_labels)
                    )
                ], dtype=numpy.float32)
            colors_dev = copy_dev(self.ctx, colors)

            data = numpy.array(
                [
                    (*inv_attr_to_label[i], complex(numpy.nan)) if i in inv_attr_to_label else (
                        complex(numpy.nan), complex(numpy.nan), complex(numpy.nan), complex(numpy.nan)
                    ) for i in range(n_labels)
                ], dtype=complex_t
            )
            data_dev = copy_dev(self.ctx, data)

            self.prg.color_basins_attractors(
                queue, img.shape, None,
                self.points_dev,
                data_dev,
                labels_dev,
                colors_dev,
                flags_dev,
                img.dev,
            )

            return img.read(queue)

        return attractors

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds,
                root_seq=None, method="sections", tolerance_decimals=3, seed=None, verbose=False):
        if method == "sections":
            return self.compute_sections(queue, img, skip, iter, h, alpha, c, bounds,
                                         root_seq, tolerance_decimals, seed)
        if method == "periods":
            return self.compute_periods(queue, img, skip, iter, h, alpha, c, bounds,
                                        root_seq, tolerance_decimals, seed)
        if method == "periods+attractors":
            image = self.compute_periods_and_attractors(queue, img, skip, iter, h, alpha, c, bounds,
                                                        root_seq, tolerance_decimals, seed, verbose)
            return image
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

    def _find_attractors(self, queue, shape, iter: int, periods: cl.Buffer, points: cl.Buffer, tol):
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
