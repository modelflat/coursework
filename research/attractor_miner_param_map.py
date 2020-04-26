import itertools
import os
from datetime import datetime

import numpy
from matplotlib import pyplot
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth

import pyopencl as cl

from config import C
from core.param_map import ParameterMap
from core.basins import BasinsOfAttraction
from core.utils import create_context_and_queue, CLImg

from PIL import Image


def save_with_axes(filename, image, attractors, colors, bounds, dpi=64, legend_bbox_anchor=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
    ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

    for attr, color in zip(attractors, colors):
        ax.scatter(bounds[0] - 1, bounds[2] - 1,
                   marker="o",
                   color=tuple(numpy.clip(color, 0.0, 1.0)),
                   label=f"{attr}")
    if legend_bbox_anchor is not None:
        ax.legend(bbox_to_anchor=legend_bbox_anchor)
    else:
        ax.legend(loc="upper right")

    fig.tight_layout()
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    fig.tight_layout()
    fig.savefig(filename)
    pyplot.close(fig)


def extract_attractors(filename, show=False):
    res_filename = filename + ".temp.res.npy"
    counts_filename = filename + ".temp.counts.npy"

    counts = None
    if os.path.exists(res_filename):
        res = numpy.load(res_filename)
    else:
        res = numpy.load(filename)
        res = res \
            .reshape(numpy.prod(res.shape[:-1]), 2) \
            .round(3)
        res = res[(numpy.abs(res[:, 0]) < 10) & (numpy.abs(res[:, 1]) < 10)]
        res, counts = numpy.unique(res, axis=0, return_counts=True)
        numpy.save(res_filename, res)
        numpy.save(counts_filename, counts)

    if os.path.exists(counts_filename) and counts is None:
        counts = numpy.load(counts_filename)
    elif counts is None:
        _, counts = numpy.unique(res, axis=0, return_counts=True)
        numpy.save(counts_filename, counts)

    thr = 1600
    # thr = 6400 * 1.5
    cond = (numpy.linalg.norm(res, axis=1) > 0.05) & (counts > thr)
    res = res[cond]
    counts = counts[cond]

    x, y = res.T

    print(len(x))

    bandwidth = estimate_bandwidth(res, quantile=0.1)
    clusters = MeanShift(n_jobs=-1, bandwidth=bandwidth).fit_predict(res)

    n_clusters = len(numpy.unique(clusters))

    print(f"found {n_clusters} attractors")

    colors = itertools.cycle('bgrcmyk')

    fig, ax = pyplot.subplots(1, 1)

    attractors = []

    for k, col in zip(range(n_clusters), colors):
        idxs = numpy.argwhere(clusters == k)
        cnts = counts[idxs]
        mx_idx = numpy.argmax(cnts)
        center = res[idxs][mx_idx][0]
        attractors.append((center[0], center[1]))
        ax.plot(*res[idxs].T, col + '.')
        ax.add_artist(pyplot.Circle(center, 0.05, color='black', fill=False))

    if show:
        ax.grid()
        ax.set_title(f"{thr}")
        pyplot.show()

    return attractors


def periods_and_attractors_at_point(ctx, queue, param, basins, h, alpha, attractors, colors, root_seq,
                                    output_name, output_dir):
    image = CLImg(ctx, (1024, 1024))

    bounds = (-2, 2, -2, 2)

    output_name = f"{output_dir}{h:.2f},{alpha:.2f}_{output_name}"

    basins.compute_and_color_known(
        queue, image, skip=1 << 12, iter=1 << 6,
        h=h, alpha=alpha, c=C, bounds=bounds, attractors=attractors, colors=colors,
        root_seq=root_seq, tolerance_decimals=3, seed=42,
    )

    save_with_axes(f"{output_name}_attr.png", image.as_img(), attractors, colors, bounds)

    basins.compute_periods(
        queue, image, skip=1 << 10, iter=1 << 6,
        h=h, alpha=alpha, c=C, bounds=bounds, root_seq=root_seq, tolerance_decimals=3, seed=42,
    )

    cl.enqueue_copy(queue, basins.periods, basins.periods_dev)
    periods, period_counts = numpy.unique(basins.periods, return_counts=True)

    periods = periods[numpy.argsort(period_counts)[::-1]]

    period_colors = param.compute_colors_for_periods(queue, periods, 1 << 6)

    save_with_axes(f"{output_name}_periods.png", image.as_img(), periods[:10], period_colors[:10], bounds)


def main_attractors(ctx, queue):
    basins = BasinsOfAttraction(ctx)
    param = ParameterMap(ctx)

    attractors = extract_attractors("test-map-20200419192630.npy")

    colors = [
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
    ]

    root_seq = (0, 0, 1)

    periods_and_attractors_at_point(ctx, queue, param, basins, -4.30, 0.70, attractors, colors, root_seq, "attr", "attr_res/")
    # periods_and_attractors_at_point(ctx, queue, param, basins, -3.54, 0.84, attractors, colors, root_seq, "attr", "attr_res/")
    # periods_and_attractors_at_point(ctx, queue, param, basins, -3.72, 0.63, attractors, colors, root_seq, "attr", "attr_res/")
    # periods_and_attractors_at_point(ctx, queue, param, basins, -6.00, 0.57, attractors, colors, root_seq, "attr", "attr_res/")
    # periods_and_attractors_at_point(ctx, queue, param, basins, -5.94, 0.70, attractors, colors, root_seq, "attr", "attr_res/")


def draw_points_on_map(map_file, points_file, bounds, output):
    import json
    with open(points_file) as f:
        points = json.load(f)

    from PIL import Image

    image = Image.open(map_file)
    dpi = 64
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 140})

    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
    ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

    for i, point in enumerate(points):
        ax.scatter(*point,
                   marker="v",
                   s=1000,
                   color=(1.0, 1.0, 1.0),
                   label=f"{i}")
        ax.annotate(str(i), point)

    fig.tight_layout()
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    fig.tight_layout()
    fig.savefig(output)
    pyplot.close(fig)


def main_old(ctx, queue):
    param = ParameterMap(ctx)
    bas = BasinsOfAttraction(ctx)

    image = CLImg(ctx, (1920 // 2, 1080))

    # bounds = (-6, -3, 0.5, 1.0)
    bounds = (-2, 2, -2, 2)

    def compute_map(name):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        points, periods = param.compute_incremental(
            queue, image,
            skip=1 << 16,
            skip_batch_size=1 << 12,
            iter=1 << 6,
            z0=complex(0.5, 0.0),
            c=C,
            bounds=bounds,
            root_seq=(0, 0, 1),
            seed=42,
            draw_image=True,
        )
        image_name = f"{name}-{ts}.png"
        periods_name = f"{name}-{ts}-periods.npy"
        points_name = f"{name}-{ts}.npy"
        image.save(image_name)
        numpy.save(periods_name, periods)
        numpy.save(points_name, points)
        return image_name, periods_name, points_name

    def compute_phase(name):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        points, periods = bas.deep_capture(
            queue, image.shape,
            skip=1 << 16,
            skip_batch_size=1 << 12,
            iter=1 << 6,
            h=-4.5, alpha=0.75,
            c=C,
            bounds=bounds,
            root_seq=(0, 0, 1),
            seed=42,
        )
        # image_name = f"{name}-{ts}.png"
        periods_name = f"{name}-{ts}-periods.npy"
        points_name = f"{name}-{ts}.npy"
        # image.save(image_name)
        numpy.save(periods_name, periods)
        numpy.save(points_name, points)
        return None, periods_name, points_name

    # _, _, points_name = compute_map("test-map")
    # _, _, points_name = compute_phase("test-phase")
    # points_name = "test-map-20200418134726.npy"
    # points_name = "test-map-20200418145033.npy"
    # points_name = "test-map-20200419192630.npy"
    points_name = "test-map-20200419192630.npy"
    # points_name = "test-phase-20200419203953.npy"
    print(points_name)
    attractors = extract_attractors(points_name)

    b = BasinsOfAttraction(ctx)

    points = numpy.load(points_name)

    colors = [
                 (1.0, 0.0, 0.0, 1.0),
                 (0.0, 1.0, 0.0, 1.0),
                 (0.0, 0.0, 1.0, 1.0),
                 (1.0, 0.0, 1.0, 1.0),
                 (0.0, 1.0, 1.0, 1.0),
                 (1.0, 0.0, 1.0, 1.0),
                 (1.0, 1.0, 0.0, 1.0),
             ] * 3

    b.color_known_attractors(queue, image, points.shape[2], attractors, colors, points)

    attrs_to_colors = dict()
    for a, c in zip(attractors, colors):
        attrs_to_colors[a] = c

    save_with_axes("feelsdankman-.png", image.as_img(), attrs_to_colors, bounds)

    image.save("feelsdankman.png")


def main(ctx, queue):
    main_attractors(ctx, queue)

if __name__ == '__main__':
    # main(*create_context_and_queue())

    Image.MAX_IMAGE_PIXELS = 192000000
    draw_points_on_map("final_001_inc_20200422-013434.png", "attr_res/points.json", (-6, 0, 0.5, 1.0),
                       "attr_res/map.png")
