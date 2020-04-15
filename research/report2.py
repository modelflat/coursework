import os
from datetime import datetime

import numpy
from PIL import Image

from config import C
from core.phase_plot import PhasePlot
from core.bif_tree import BifTree
from core.param_map import ParameterMap
from core.bounding_box import BoundingBox
from core.fast_box_counting import FastBoxCounting
from core.utils import create_context_and_queue, CLImg

from core.basins import BasinsOfAttraction
from tqdm import tqdm

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

NOW = datetime.now()


def bgra2rgba(arr):
    arr.T[numpy.array((0, 1, 2, 3))] = arr.T[numpy.array((2, 1, 0, 3))]
    return arr


def make_img(arr):
    return Image.fromarray(bgra2rgba(arr), mode="RGBA")


def to_file(arr, bounds, filename, save_with_axes=True, periods=None, iter=None, dpi=64, legend_bbox_anchor=None):
    image = make_img(arr)
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if save_with_axes:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
        ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
        ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

        if periods is not None:
            for p, p_c, col in periods:
                ax.scatter(bounds[0] - 1, bounds[2] - 1,
                           marker="o",
                           color=tuple(numpy.clip(col, 0.0, 1.0)),
                           label="{} ({:3.2f}%)".format("chaos" if p >= iter - 1 else p,
                                                        100 * p_c / numpy.prod(image.size)))
            if legend_bbox_anchor is not None:
                ax.legend(bbox_to_anchor=legend_bbox_anchor)
                fig.tight_layout()
            else:
                ax.legend(loc="upper right")

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    else:
        with open(filename, "wb") as f:
            image.save(f)


def basins(ctx, queue, filename, **params):
    p = BasinsOfAttraction(ctx)
    img = CLImg(ctx, params['full_size'])

    params = dict(
        skip=params['skip'],
        h=params["h"],
        alpha=params["alpha"],
        c=C,
        bounds=params["bounds"],
        root_seq=params["root_seq"],
        method="dev",
        scale_factor=1,
        seed=42
    )

    image = p.compute(queue, img, **params)

    # filename = "{}_{}x{}_h=({})_alpha=({})_r=({})__{}.png".format(
    #     filename,
    #     *params.get("full_size", img.shape),
    #     params["h"],
    #     params["alpha"],
    #     ",".join(map(str, params["root_seq"])),
    #     NOW.strftime("%Y%m%d-%H%M%S")
    # )

    filename = f"h={params['h']}_alpha={params['alpha']}_{filename}.png"

    to_file(image, params["bounds"], os.path.join(OUTPUT_ROOT, "basins", filename))


def phase(ctx, queue, filename, **params):
    p = PhasePlot(ctx)
    img = CLImg(ctx, params['full_size'])

    params = dict(
        skip=params['skip'],
        iter=params['iter'],
        h=params["h"],
        alpha=params["alpha"],
        c=C,
        bounds=params["bounds"],
        root_seq=params["root_seq"],
        grid_size=32,
        seed=42
    )

    image = p.compute(queue, img, **params)

    # filename = "{}_{}x{}_h=({})_alpha=({})_r=({})__{}.png".format(
    #     filename,
    #     *params.get("full_size", img.shape),
    #     params["h"],
    #     params["alpha"],
    #     ",".join(map(str, params["root_seq"])),
    #     NOW.strftime("%Y%m%d-%H%M%S")
    # )

    filename = f"h={params['h']}_alpha={params['alpha']}_{filename}.png"

    to_file(image, params["bounds"], os.path.join(OUTPUT_ROOT, "phase", filename))


def question_1(ctx, queue):
    points = [
        # periodic
        (-3.8906, 0.7627),
        # chaotic
        (-4.9102, 0.7383)
    ]

    for point in points[:0]:
        h, alpha = point
        kwargs = dict(
            full_size=(512, 512),
            bounds=(-2, 2, -2, 2),
            skip=1 << 9,
            h=h,
            alpha=alpha
        )

        def sequence_of_zeros():
            nz = [0, 1, 2, 3, 4, 5, 16]
            other_root = 1
            for z in tqdm(nz):
                if z == 0:
                    basins(ctx, queue, "zseq_0", root_seq=numpy.array([0]), **kwargs)
                else:
                    basins(ctx, queue, "zseq_{}".format(z), root_seq=numpy.array(z*[0] + [other_root]), **kwargs)

        # basins(ctx, queue, "test", root_seq=[0, 0, 1], **kwargs)
        sequence_of_zeros()

    for point in points:
        h, alpha = point
        kwargs = dict(
            full_size=(512, 512),
            bounds=(-2, 2, -2, 2),
            skip=1 << 9,
            iter=1 << 9,
            h=h,
            alpha=alpha
        )

        def sequence_of_zeros():
            nz = [0, 1, 2, 3, 4, 5, 16]
            other_root = 1
            for z in tqdm(nz):
                if z == 0:
                    phase(ctx, queue, "zseq_0", root_seq=numpy.array([0]), **kwargs)
                else:
                    phase(ctx, queue, "zseq_{}".format(z), root_seq=numpy.array(z*[0] + [other_root]), **kwargs)

        # basins(ctx, queue, "test", root_seq=[0, 0, 1], **kwargs)
        sequence_of_zeros()


def question_3(ctx, queue):
    points = [
        (-2.0, x)
        for x in numpy.linspace(0.5, 1.0, num=10)
    ]

    for point in tqdm(points):
        h, alpha = point
        kwargs = dict(
            full_size=(512, 512),
            bounds=(-2, 2, -2, 2),
            skip=1 << 9,
            iter=1 << 9,
            h=h,
            alpha=alpha,
            root_seq=[0, 0, 1],
        )

        kwargs_basins = dict(
            full_size=(512, 512),
            bounds=(-2, 2, -2, 2),
            skip=1 << 9,
            h=h,
            alpha=alpha,
            root_seq=[0, 0, 1],
        )

        phase(ctx, queue, "I", **kwargs)
        basins(ctx, queue, "I", **kwargs_basins)

        kwargs['alpha'] = 1.0 - kwargs['alpha']
        kwargs['h'] = - kwargs['h']
        kwargs_basins['alpha'] = 1.0 - kwargs_basins['alpha']
        kwargs_basins['h'] = - kwargs_basins['h']

        phase(ctx, queue, "II", **kwargs)
        basins(ctx, queue, "II", **kwargs_basins)


if __name__ == '__main__':
    ctx, queue = create_context_and_queue()

    # question_1(ctx, queue)
    question_3(ctx, queue)
