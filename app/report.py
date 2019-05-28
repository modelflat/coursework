import os
from datetime import datetime

import numpy
from PIL import Image

from config import C
from core.bif_tree import BifTree
from core.param_map import ParameterMap
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


def to_file(arr, bounds, filename, save_with_axes=True, periods=None, iter=None):
    image = make_img(arr)
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if save_with_axes:
        import matplotlib.pyplot as plt
        dpi = 64
        fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
        ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
        ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
        ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

        if periods is not None:
            for p, p_c, col in periods:
                ax.scatter(bounds[0] - 1, bounds[2] - 1,
                           marker="o",
                           color=tuple(numpy.clip(col, 0.0, 1.0)),
                           label="{} ({:3.2f}%)".format(p if p > iter / 1.5 else "chaos",
                                                        100 * p_c / numpy.prod(image.size)))
            ax.legend(loc="upper right")

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    else:
        with open(filename, "wb") as f:
            image.save(f)


ctx, queue = create_context_and_queue()


def param_map(filename, **params):
    p = ParameterMap(ctx)
    img = CLImg(ctx, (1 << 6, 1 << 6))

    defaults = dict(
        full_size=(1 << 12, 9 * (1 << 8)),
        skip=1 << 10,
        iter=1 << 8,
        z0=complex(0.001, 0.001),
        c=C,
        tol=1e-6,
        seed=42,
        method="precise",
        scale_factor=1,
    )

    params = { **defaults, **params }

    image, periods = p.compute_tiled(
        queue, img, **params
    )

    print(periods.min(), periods.shape, periods.dtype)

    filename = "{}_{}x{}_b=({})_r=({})__{}.png".format(
        filename,
        *params["full_size"],
        "_".join(map(str, params["bounds"])),
        ",".join(map(str, params["root_seq"])),
        NOW.strftime("%Y%m%d-%H%M%S")
    )

    print("\ncomputing periods statistics")
    periods, periods_counts = numpy.unique(periods, return_counts=True)

    ind = periods_counts.argsort()[::-1]

    n_top = 20

    top_periods = periods[ind[:n_top]]
    top_periods_counts = periods_counts[ind[:n_top]]

    colors = p.compute_colors_for_periods(queue, top_periods, params["iter"])

    for _, i in zip(top_periods, range(n_top)):
        print("#{} -- {:3d} (x{:6d}) -- {}".format(
            i + 1, top_periods[i], top_periods_counts[i], colors[i]
        ))

    to_file(image, params["bounds"], os.path.join(OUTPUT_ROOT, "param_maps", filename),
            periods=zip(top_periods, top_periods_counts, colors),
            iter=params["iter"])


def bif_tree(filename, **params):
    p = BifTree(ctx)
    # img = CLImg(ctx, (1 << 14, 9 * (1 << 10) // 16))
    img = CLImg(ctx, (1 << 13, 1 << 13))

    params = dict(
        #full_size=(1 << 12, 9 * (1 << 8)),
        skip=1 << 16,
        iter=1 << 14,
        z0=complex(0.001, 0.001),
        c=C,
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


def basins(filename, **params):
    p = BasinsOfAttraction(ctx)
    # img = CLImg(ctx, (1 << 14, 9 * (1 << 14) // 16))
    img = CLImg(ctx, (1 << 12, 1 << 12))

    params = dict(
        skip=1 << 10,
        h=params["h"],
        alpha=params["alpha"],
        c=C,
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


bounds = (-6, 0, 0.5, 1.0)
name = "test__"

# param_map(name, bounds=bounds, root_seq=numpy.array([0]))
# param_map(name, bounds=bounds, root_seq=numpy.array([1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([1, 2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 0, 0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 0, 0, 0, 1]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 0, 2]))
# param_map(name, bounds=bounds, root_seq=numpy.array([0, 0, 0, 0, 0, 0, 2]))


# bif_tree("btree", fixed_id=1, h=0.0, h_min=-6, h_max=0, alpha=1.0, alpha_min=0, alpha_max=1, root_seq=numpy.array([0]), var_min=-4, var_max=4)

for h in tqdm(numpy.linspace(-3, 0, 4)):
    basins("basins_along_alpha1", h=h, alpha=1.0, root_seq=numpy.array([0]))

basins("basins", h=-5.0, alpha=1.0, root_seq=numpy.array([0]))
# basins("basins", h=2.22, alpha=0.0, root_seq=numpy.array([0]))
