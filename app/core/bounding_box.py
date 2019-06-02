import pyopencl as cl
import numpy

from core import build_program_from_file
from core.utils import CLImg, alloc_like


def interpolate(val, smin, smax, tmin, tmax):
    return (val - smin) / (smax - smin) * (tmax - tmin) + tmin


class BoundingBox:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "bounding_box.cl")

    def compute(self, queue, img: CLImg, bounds=None):
        bbox = numpy.empty((4,), numpy.int32)
        bbox_dev = alloc_like(self.ctx, bbox)

        self.prg.compute_bounding_box(
            queue, (4,), None,
            img.dev, bbox_dev
        )

        cl.enqueue_copy(queue, bbox, bbox_dev)

        if bounds is None:
            return bbox
        else:
            min_x = interpolate(bbox[0], 0, img.shape[0] - 1, bounds[0], bounds[1])
            max_x = interpolate(bbox[1], 0, img.shape[0] - 1, bounds[0], bounds[1])
            min_y = interpolate(bbox[3], 0, img.shape[1] - 1, bounds[2], bounds[3])
            max_y = interpolate(bbox[2], 0, img.shape[1] - 1, bounds[2], bounds[3])
            return bbox, (min_x, max_x, min_y, max_y)
