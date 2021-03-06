"""
Implementation of algorithm described in https://www.sciencedirect.com/science/article/pii/S0169260712001794

Changes that were made:
- Images are assumed to be 2D - trivial change from general 3D approach described
- For simplicity, sorting happens on CPU. TODO change to GPU impl
- For simplicity (and due to temporary constraints regarding OpenCL on Intel(R) Core i5), black and gray boxes
 are counted using atomics, and not local groups + scans.
- Both black and gray boxes are counted, but only filled boxes are used in final computation of D
 (n_black + n_gray)

Note, that due to performance-affecting changes (2 and 3) the implementation becomes much less performant,
 but is still efficient due to the 1st assumption.

Note, that this algorithm was tested only against images that are square, and have a side that is power of 2
 (e.g. 512x512)

TODO looks like it works, but needs more testing
TODO add customizable is_not_empty function
"""


import numpy
import pyopencl as cl
from matplotlib.pyplot import imread
from scipy.stats import linregress
from time import perf_counter

from core import build_program_from_file
from .utils import alloc_image, create_context_and_queue


class FastBoxCounting:

    def __init__(self, ctx: cl.Context):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "fast_box_counting.cl")

    def compute(self, queue, image, verbose=False):
        empty_box_inted_value = 2 ** 64 - 1
        max_significands = int(numpy.floor(numpy.log2(numpy.prod(image.shape)))) // 2

        intercalated_coords = numpy.empty(numpy.prod(image.shape), dtype=numpy.uint64)
        intercalated_coords_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                                            size=intercalated_coords.nbytes)
        self.prg.intercalate_coord(
            queue, image.shape, None,
            numpy.uint64(empty_box_inted_value),
            image,
            intercalated_coords_dev
        )

        cl.enqueue_copy(queue, intercalated_coords, intercalated_coords_dev)

        intercalated_coords.sort()

        cl.enqueue_copy(queue, intercalated_coords_dev, intercalated_coords)

        black_count = numpy.empty((1,), dtype=numpy.int32)
        black_count_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=black_count.nbytes)
        gray_count = numpy.empty((1,), dtype=numpy.int32)
        gray_count_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=gray_count.nbytes)

        boxes = []

        for i in range(0, max_significands):
            strip_bits = 2 * i
            box_size = 2 ** i

            black_count[0] = 0
            gray_count[0] = 0
            cl.enqueue_copy(queue, black_count_dev, black_count)
            cl.enqueue_copy(queue, gray_count_dev, gray_count)

            self.prg.count_non_empty(
                queue, image.shape, None,
                numpy.uint64(empty_box_inted_value),
                numpy.int32(box_size),
                numpy.int32(strip_bits),
                numpy.int32(len(intercalated_coords)),
                intercalated_coords_dev,
                black_count_dev,
                gray_count_dev
            )

            cl.enqueue_copy(queue, black_count, black_count_dev)
            cl.enqueue_copy(queue, gray_count, gray_count_dev)

            total_boxes = numpy.prod(image.shape) // box_size
            filled_boxes = black_count[0] + gray_count[0]

            if verbose:
                print("For box size {} ({} total boxes, {} bits stripped) - {} black, {} gray boxes {:.2f}% boxes "
                      "occupied".format(box_size, total_boxes, strip_bits, black_count[0], gray_count[0],
                                        100 * filled_boxes / total_boxes))

            if filled_boxes == 0:
                # it makes no sense to check boxes of bigger size if filled_boxes is 0
                # also, this can happen only in the first iteration, so image is empty according to
                # is_not_empty function
                assert i == 0, i
                # TODO output a warning about empty image
                return numpy.nan

            boxes.append((numpy.log(1 / box_size), numpy.log(black_count[0] + gray_count[0])))

        X, y = numpy.array(boxes).T
        return linregress(X, y).slope


# TODO proper testing?
def test_gen_image(shape=(512, 512), kind="rand"):
    img, img_dev = alloc_image(ctx, shape)

    if kind == "rand":
        values = numpy.random.randint(0, 2, shape, dtype=numpy.uint8)
    elif kind == "ones":
        values = numpy.ones_like(img)
    elif kind == "zeros":
        values = numpy.zeros_like(img)
    else:
        raise ValueError(kind)

    cl.enqueue_copy(queue, img_dev, values, origin=(0, 0), region=img.shape[:-1])

    box_counter = FastBoxCounting(ctx)

    t = perf_counter()
    D = box_counter.compute(queue, img_dev)
    t = perf_counter() - t

    print("D = {:.3f}; computed in {:.3f} s".format(D, t))


# TODO proper testing?
def test_image(uri):
    values = imread(uri)
    print(values.shape)

    img, img_dev = alloc_image(ctx, (512, 512))

    cl.enqueue_copy(queue, img_dev, values, origin=(0, 0), region=values.shape[:-1])

    box_counter = FastBoxCounting(ctx)

    t = perf_counter()
    D = box_counter.compute(queue, img_dev)
    t = perf_counter() - t

    print("D = {:.3f}; computed in {:.3f} s".format(D, t))


if __name__ == '__main__':
    ctx, queue = create_context_and_queue()
    test_gen_image(kind="zeros")
    test_gen_image(kind="ones")
    test_gen_image(kind="rand")
