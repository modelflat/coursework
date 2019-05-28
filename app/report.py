import numpy

from PIL import Image

from core.param_map import ParameterMap
from core.utils import create_context_and_queue, CLImg
from config import C

import os


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
    img = CLImg(ctx, (1 << 6, 1 << 6))

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

    filename = "{}_b=({})_r=({}).png".format(
        filename,
        "_".join(map(str, params["bounds"])),
        ",".join(map(str, params["root_seq"]))
    )

    to_file(image, os.path.join(OUTPUT_ROOT, "param_maps", filename))


param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([1]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([2]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 1]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 2]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([1, 2]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 1]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 0, 1]))
param_map("map", bounds=(-6, 6, 0, 1), root_seq=numpy.array([0, 0, 0, 0, 1]))
