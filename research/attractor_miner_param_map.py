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
import json

from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = 192000000


BOUNDS = (-6, 0, 0.5, 1.0)
ROOT_SEQ = (0, 0, 0, 1)
ROOT_SEQ_STR = "".join(map(str, ROOT_SEQ))
POINTS_SOURCE = "attr_res/points.json"
ATTR_SOURCE = "attr_res_00001/map_00001-20200507184909.npy"
MAP_SOURCE = "final_00001_inc_20200422-023811.png"
MAP_SOURCE = "final_0001_inc_20200422-020632.png"
COLORS = [
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (0.5, 0.5, 0.5, 1.0),
]


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


def extract_attractors(filename, attr_image_output=None, show=False, data=None):
    res_filename = filename + ".temp.res.npy"
    counts_filename = filename + ".temp.counts.npy"

    counts = None
    if os.path.exists(res_filename) and not data:
        res = numpy.load(res_filename)
    else:
        if data:
            res = data
        else:
            res = numpy.load(filename)
        res = res \
            .reshape(numpy.prod(res.shape[:-1]), 2) \
            .round(3)
        res = res[(numpy.abs(res[:, 0]) < 10) & (numpy.abs(res[:, 1]) < 10)]
        res, counts = numpy.unique(res, axis=0, return_counts=True)
        numpy.save(res_filename, res)
        numpy.save(counts_filename, counts)

    if data is not None:
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

    if attr_image_output is not None:
        fig.tight_layout()
        fig.savefig(attr_image_output)

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


def draw_attractors_at_points(ctx, queue):
    basins = BasinsOfAttraction(ctx)
    param = ParameterMap(ctx)

    with open(POINTS_SOURCE, "r") as f:
        points = json.load(f)

    attractors = extract_attractors(ATTR_SOURCE, f"attr_res_{ROOT_SEQ_STR}/all_attractors.png")

    for i, point in enumerate(tqdm(points)):
        break
        periods_and_attractors_at_point(ctx, queue, param, basins, *point,
                                        attractors, COLORS, ROOT_SEQ, "attr", f"attr_res_{ROOT_SEQ_STR}/{i}_")

    draw_points_on_map(MAP_SOURCE, "attr_res/points.json", BOUNDS, f"attr_res_{ROOT_SEQ_STR}/map.png")


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


def compute_and_extract(basins: BasinsOfAttraction, queue, image, h, alpha):
    points, periods = basins.deep_capture(
        queue, image.shape,
        skip=1 << 16,
        skip_batch_size=1 << 12,
        iter=1 << 6,
        h=h, alpha=alpha, c=C,
        bounds=BOUNDS,
        root_seq=ROOT_SEQ,
        seed=42
    )
    attractors = extract_attractors("", data=points)

    basins.color_known_attractors(queue, image, points.shape[2], attractors, COLORS, points, periods)

    attrs_to_colors = dict()
    for a, c in zip(attractors, COLORS):
        attrs_to_colors[a] = c

    # save_with_axes(f"attr_res_{ROOT_SEQ_STR}/map_attractors.png", image.as_img(), attrs_to_colors, COLORS, BOUNDS)


def main_new(ctx, queue):
    basins = BasinsOfAttraction(ctx)
    image = CLImg(ctx, (1024, 1024))

    with open(POINTS_SOURCE, "r") as f:
        points = json.load(f)

    for (h, alpha) in tqdm(points):
        compute_and_extract(basins, queue, image, h, alpha)


def main_old(ctx, queue):
    param = ParameterMap(ctx)
    basins = BasinsOfAttraction(ctx)

    image = CLImg(ctx, (1920 // 2, 1080))

    def compute_map(name):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")

        periods, points = basins.deep_capture(
            queue, image.shape,
            skip=1 << 16,
            skip_batch_size=1 << 12,
            iter=1 << 6,
            z0=complex(0.5, 0.0),
            c=C,
            bounds=BOUNDS,
            root_seq=ROOT_SEQ,
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

    _, periods_name, points_name = compute_map(f"attr_res_{ROOT_SEQ_STR}/map_{ROOT_SEQ_STR}")
    # points_name = "attr_res_00001/map_00001-20200507184909.npy"
    # periods_name = "attr_res_00001/map_00001-20200507184909-periods.npy"

    print(points_name)
    attractors = extract_attractors(points_name)

    b = BasinsOfAttraction(ctx)

    points = numpy.load(points_name)
    periods = numpy.load(periods_name)

    b.color_known_attractors(queue, image, points.shape[2], attractors, COLORS, points, periods)

    attrs_to_colors = dict()
    for a, c in zip(attractors, COLORS):
        attrs_to_colors[a] = c

    save_with_axes(f"attr_res_{ROOT_SEQ_STR}/map_attractors.png", image.as_img(), attrs_to_colors, COLORS, BOUNDS)


def main(ctx, queue):
    # main_old(ctx, queue)
    # return

    main_new(ctx, queue)

    # draw_attractors_at_points(ctx, queue)
    # draw_points_on_map("final_001_inc_20200422-013434.png", "attr_res/points.json", (-6, 0, 0.5, 1.0), "attr_res/map_updated.png")


if __name__ == '__main__':
    main(*create_context_and_queue())
