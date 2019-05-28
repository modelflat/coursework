import os
from datetime import datetime

import numpy
from PIL import Image

from config import C
from core.bif_tree import BifTree
from core.param_map import ParameterMap
from core.utils import create_context_and_queue, CLImg

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def bgra2rgba(arr):
    arr.T[numpy.array((0, 1, 2, 3))] = arr.T[numpy.array((2, 1, 0, 3))]
    return arr


def make_img(arr):
    return Image.fromarray(bgra2rgba(arr), mode="RGBA")


def to_file(arr, filename):
    image = make_img(arr)
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(filename, "wb") as f:
        image.save(f)


ctx, queue = create_context_and_queue()


def param_map(filename, **params):
    p = ParameterMap(ctx)
    img = CLImg(ctx, (1 << 8, 1 << 8))

    defaults = dict(
        full_size=(1 << 12, 9 * (1 << 8)),
        skip=1 << 12,
        iter=1 << 8,
        z0=complex(0.001, 0.001),
        c=C,
        tol=1e-5,
        seed=42
    )

    image = p.compute_tiled(
        queue, img, **{
            **params,
            **defaults
        }
    )

    filename = "{}_{}x{}_b=({})_r=({})__{}.png".format(
        filename,
        *params["full_size"],
        "_".join(map(str, params["bounds"])),
        ",".join(map(str, params["root_seq"])),
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    to_file(image, os.path.join(OUTPUT_ROOT, "param_maps", filename))


def bif_tree(filename, **params):
    p = BifTree(ctx)
    img = CLImg(ctx, (1 << 12, 9 * (1 << 8)))

    params = dict(
        #full_size=(1 << 12, 9 * (1 << 8)),
        skip=1 << 16,
        iter=1 << 8,
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
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    to_file(image, os.path.join(OUTPUT_ROOT, "btree", filename))


# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([1]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([2]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 1]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 2]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([1, 2]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 1]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 0, 1]))
# param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 0, 0, 1]))


bif_tree("btree",
         fixed_id=1,
         h=0.0, h_min=-6, h_max=6,
         alpha=1.0, alpha_min=0, alpha_max=1,
         root_seq=numpy.array([0]),
         var_min=-4, var_max=4,
         )
