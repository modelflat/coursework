import time

from collections import defaultdict
from itertools import chain

import numpy
import pyopencl as cl
import cv2

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type, copy_dev


class BasinsOfAttraction:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "basins.cl")

    def compute(self, queue, img, skip, iter, h, alpha, c, bounds,
                root_seq=None, method="sections", tolerance_decimals=3, seed=None, verbose=False):
        points = numpy.empty((numpy.prod(img.shape), 2), dtype=real_type)
        points_dev = alloc_like(self.ctx, points)

        periods = numpy.empty(img.shape, dtype=numpy.int32)
        periods_dev = alloc_like(self.ctx, periods)

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

            points_dev,
            periods_dev
        )

        if method == "sections":
            self.prg.color_basins_section(
                queue, img.shape, None,
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
        elif method == "periods+attractors":
            contours = numpy.empty(img.shape, dtype=numpy.uint8)
            contours_dev = alloc_like(self.ctx, contours)

            self.prg.color_basins_periods_attractors(
                queue, img.shape, None,
                numpy.int32(iter),
                real_type(1 / 10 ** tolerance_decimals),
                numpy.array(bounds, dtype=real_type),
                periods_dev,
                points_dev,
                contours_dev,
                img.dev,
            )

            cl.enqueue_copy(queue, contours, contours_dev)
            cl.enqueue_copy(queue, points, points_dev)

            min_std = 1 / 10 ** (tolerance_decimals * 2)
            min_points_in_region = 4

            # todo use OpenCV/OpenCL interop
            n_labels, labels = cv2.connectedComponents(contours)

            labels_dev = copy_dev(self.ctx, labels)

            contours = contours.reshape(numpy.prod(img.shape))

            suspicious_regions = defaultdict(lambda: list())
            attractors = defaultdict(lambda: list())
            skipped_labels = set()
            suspicious_labels = set()

            complex_t = numpy.complex64 if real_type == numpy.float32 else numpy.complex128

            points_index = [list() for _ in range(n_labels)]

            t = time.perf_counter()

            # TODO this is too slow
            for label, contour, point in zip(labels.reshape(numpy.prod(img.shape)),
                                             contours,
                                             points.view(complex_t).T[0]):
                if contour == 255:
                    points_index[label].append(point)

            points_index = [numpy.array(v) for v in points_index]

            for i in range(n_labels):
                points_in_component = points_index[i]

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
                    attractors[mean].append(i)
                    if verbose:
                        print(f'{i:4}: {len(points_in_component):6} converged  / {mean}')
                else:
                    # this region has varying values, we need to track this fact
                    mn = numpy.round(points_in_component.min(), tolerance_decimals)
                    mx = numpy.round(points_in_component.max(), tolerance_decimals)
                    points_in_component = numpy.round(points_in_component, tolerance_decimals)
                    elements, counts = numpy.unique(points_in_component, return_counts=True)
                    most_common_index = numpy.argmax(counts)
                    most_common = elements[most_common_index]
                    if counts[most_common_index] / len(points_in_component) > 0.75:
                        suspicious_regions[(mn, most_common, mx)].append(i)
                    else:
                        weighted_mean = numpy.round(
                            numpy.average(elements, weights=counts/len(points_in_component)),
                            tolerance_decimals)
                        suspicious_regions[(mn, weighted_mean, mx)].append(i)

                    suspicious_labels.add(i)
                    if verbose:
                        print(f'{i:4}: {len(points_in_component):6} suspicious / {mn} : {mx}')
            t = time.perf_counter() - t
            print(f'analysis: {t:.3} s')

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

            attractors = {
                (attr if isinstance(attr, tuple) else (attr, attr, attr)): v
                for attr, v in attractors.items()
            }

            inv_attr_to_label = {
                v: k
                for k, vv in attractors.items()
                for v in vv
            }

            colors_dict = {
                attr: (240 * i / len(attractors), 1, 1)
                for i, attr in enumerate(sorted(attractors.keys(), key=lambda x: x[1] if isinstance(x, tuple) else x), 1)
            }

            flags = numpy.array(
                [
                    # is suspicious?
                    (  0b01 if i in suspicious_labels else 0)
                    # is attractor?
                    | (0b10 if i in inv_attr_to_label else 0)
                    for i in range(n_labels)
                ],
                dtype=numpy.int32
            )
            flags_dev = copy_dev(self.ctx, flags)

            colors = numpy.array(
                [
                    col if i not in suspicious_labels else (col[0], col[1] * 0.5, col[2] * 0.5)
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
                points_dev,
                data_dev,
                labels_dev,
                colors_dev,
                flags_dev,
                img.dev,
            )
        else:
            raise ValueError("Unknown method: \"{}\"".format(method))

        return img.read(queue)
