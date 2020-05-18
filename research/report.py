import os
from datetime import datetime

import numpy
from PIL import Image
from tqdm import tqdm

from app.config import Config
from app.core.phase_plot import PhasePlot
from app.core.bif_tree import BifTree
from app.core.param_map import ParameterMap
from app.core.bounding_box import BoundingBox
from app.core.fast_box_counting import FastBoxCounting
from app.core.utils import create_context_and_queue, CLImg
from app.core.basins import BasinsOfAttraction


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


def param_map(ctx, queue, filename, full_size=(640, 480), tile_size=(160, 160), **kwargs):
    p = ParameterMap(ctx)
    img = CLImg(ctx, tile_size)

    params = dict(
        full_size=full_size,
        skip=1 << 8,
        iter=1 << 8,
        z0=complex(0.001, 0.001),
        c=Config.C,
        tol=1e-6,
        seed=42,
        method="fast",
        scale_factor=1
    )

    params.update(kwargs)

    image, periods = p.compute_tiled(
        queue, img, **params
    )

    # print(periods.min(), periods.shape, periods.dtype)

    filename = "{}_{}x{}_b=({})_r=({})__{}.png".format(
        # _z0=({:+.2f},{:+.2f})
        filename,
        *params["full_size"],
        "_".join(map(str, params["bounds"])),
        ",".join(map(str, params["root_seq"])) if len(params["root_seq"]) < 6 else "r=(too long...)",
        # params["z0"].real, params["z0"].imag,
        NOW.strftime("%Y%m%d-%H%M%S")
    )

    # print("\ncomputing periods statistics")
    periods, periods_counts = numpy.unique(periods, return_counts=True)

    ind = periods_counts.argsort()[::-1]

    n_top = kwargs.get("ntop", 10)

    top_periods = periods[ind[:n_top]]
    top_periods_counts = periods_counts[ind[:n_top]]

    colors = p.compute_colors_for_periods(queue, top_periods, params["iter"])

    show_stats = False
    if show_stats:
        for _, i in zip(top_periods, range(n_top)):
            print("#{} -- {:3d} (x{:6d}) -- {}".format(
                i + 1, top_periods[i], top_periods_counts[i], colors[i]
            ))

    to_file(image, params["bounds"], os.path.join(OUTPUT_ROOT, "param_maps", filename),
            periods=zip(top_periods, top_periods_counts, colors),
            iter=params["iter"], dpi=kwargs.get("dpi", 80),
            legend_bbox_anchor=kwargs.get("legend_bbox_anchor", (1.3, 1.012)))


def bif_tree(ctx, queue, filename, **params):
    p = BifTree(ctx)
    # img = CLImg(ctx, (1 << 14, 9 * (1 << 10) // 16))
    img = CLImg(ctx, (1 << 13, 1 << 13))

    params = dict(
        skip=1 << 16,
        iter=1 << 14,
        z0=complex(0.001, 0.001),
        c=Config.C,
        var_id=0,
        fixed_id=params["fixed_id"],
        fixed_value=params["h"] if params["fixed_id"] == 0 else params["alpha"],
        other_min=params["h_min"] if params["fixed_id"] == 1 else params["alpha_min"],
        other_max=params["h_max"] if params["fixed_id"] == 1 else params["alpha_max"],
        root_seq=params["root_seq"],
        var_min=params["var_min"],
        var_max=params["var_max"],
        try_rescaling=False,
        seed=42
    )

    image = p.compute(queue, img, **params)

    filename = "{}_{}x{}_{}=({})_r=({})__{}.png".format(
        filename,
        *params.get("full_size", img.shape),
        "h" if params["fixed_id"] == 1 else "alpha",
        ",".join(map(str, (params["other_min"], params["other_max"]))),
        ",".join(map(str, params["root_seq"])),
        NOW.strftime("%Y%m%d-%H%M%S")
    )

    to_file(image, (params["other_min"], params["other_max"], params["var_min"], params["var_max"]),
            os.path.join(OUTPUT_ROOT, "btree", filename))


def basins(ctx, queue, filename, **params):
    p = BasinsOfAttraction(ctx)
    # img = CLImg(ctx, (1 << 14, 9 * (1 << 14) // 16))
    img = CLImg(ctx, (1 << 12, 1 << 12))

    params = dict(
        skip=1 << 10,
        h=params["h"],
        alpha=params["alpha"],
        c=Config.C,
        bounds=params.get("bounds", (-2, 2, -2, 2)),
        root_seq=params["root_seq"],
        method="dev",
        scale_factor=1,
        seed=42
    )

    image = p.compute(queue, img, **params)

    filename = "{}_{}x{}_h=({})_alpha=({})_r=({})__{}.png".format(
        filename,
        *params.get("full_size", img.shape),
        params["h"],
        params["alpha"],
        ",".join(map(str, params["root_seq"])),
        NOW.strftime("%Y%m%d-%H%M%S")
    )

    to_file(image, params["bounds"], os.path.join(OUTPUT_ROOT, "basins", filename))


def all_param_maps(ctx, queue):
    root0_z0 = complex(+0.3, -0.1)
    root1_z0 = complex(-0.1, +0.3)
    root2_z0 = complex(-0.3, -0.3)

    kwargs = dict(
        full_size=(640, 480),
        tile_size=(80, 80),
        bounds=(-6, 0, 0.5, 1.0),
        skip=100000,
        iter=500,
    )

    name = "pmap_"

    def basic():
        param_map(ctx, queue, "single_root0", root_seq=numpy.array([0]), z0=root0_z0, **kwargs)
        param_map(ctx, queue, "single_root1", root_seq=numpy.array([1]), z0=root1_z0, **kwargs)
        param_map(ctx, queue, "single_root2", root_seq=numpy.array([2]), z0=root2_z0, **kwargs)
        param_map(ctx, queue, "simple_02", root_seq=numpy.array([0, 2]), z0=root0_z0, **kwargs)
        param_map(ctx, queue, "simple_12", root_seq=numpy.array([1, 2]), z0=root1_z0, **kwargs)
        param_map(ctx, queue, "simple_012", root_seq=numpy.array([0, 1, 2]), z0=root0_z0, **kwargs)

    def sequence_of_zeros():
        nz = [1, 2, 3, 4, 5, 16, 32, 64, 128]
        other_root = 1
        param_map(ctx, queue, "zseq_0", root_seq=numpy.array([0]), z0=root0_z0, **kwargs)
        for z in nz:
            param_map(ctx, queue, "zseq_{}".format(z),
                      root_seq=numpy.array(z*[0] + [other_root]), z0=root0_z0, **kwargs)

    # basic()
    sequence_of_zeros()


def all_other_stuff(ctx, queue):
    bif_tree(ctx, queue, "btree", fixed_id=1, h=0.0, h_min=-6, h_max=0, alpha=1.0, alpha_min=0, alpha_max=1, root_seq=numpy.array([0]), var_min=-4, var_max=4)

    for h in tqdm(numpy.linspace(-3, 0, 4)):
        basins(ctx, queue, "basins_along_alpha1", h=h, alpha=1.0, root_seq=numpy.array([0]))

    basins(ctx, queue, "basins", h=-5.0, alpha=1.0, root_seq=numpy.array([0]))
    basins(ctx, queue, "basins", h=2.22, alpha=0.0, root_seq=numpy.array([0]))


def plot_D(ctx, queue):
    h_min, h_max = -0.01, 0.01
    alpha = 0.0

    skip = 1 << 14
    iter = 1 << 12

    img = CLImg(ctx, (512, 512))
    phase = PhasePlot(ctx)
    fbc = FastBoxCounting(ctx)
    bbox = BoundingBox(ctx)

    bounds = (-1, 1, -1, 1)

    hs = numpy.linspace(h_min, h_max, 10000)
    from time import perf_counter

    ds = []
    tm = []
    for h in tqdm(hs):
        t = perf_counter()
        phase.compute(queue, img, skip=skip, iter=iter, h=h, alpha=alpha, c=Config.C, bounds=bounds, grid_size=4, seed=42)
        tm.append(perf_counter() - t)

        box, new_bounds = bbox.compute(queue, img, bounds=bounds)
        # print("h = {:.3f}:".format(h), bounds, "->", new_bounds)

        phase.compute(queue, img, skip=skip, iter=iter, h=h, alpha=alpha, c=Config.C, bounds=new_bounds, grid_size=4, seed=42)
        d = fbc.compute(queue, img.dev)
        ds.append(d)

    print(numpy.mean(tm), "s")

    import matplotlib.pyplot as plt

    plt.plot(hs, ds)
    plt.show()


if __name__ == '__main__':
    ctx, queue = create_context_and_queue()

    all_param_maps(ctx, queue)
    # all_other_stuff(ctx, queue)
