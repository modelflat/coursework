import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_image, alloc_like, real_type


class ParameterMap:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, ("param_map.cl", "fast_param_map.cl"))
        self.map_points = None
        self._dummy_img = alloc_image(ctx, (1, 1))[1]

    def _compute_precise(
            self, queue, skip, iter, z0, c, tol, bounds, root_seq, scale_factor, img,
            periods_shape=None, seed=None):

        if periods_shape is not None:
            shape = periods_shape
        else:
            shape = img.shape

        elem_size = real_type().nbytes
        reqd_size = iter * numpy.prod(shape) // scale_factor ** 2

        if self.map_points is None or self.map_points.size != reqd_size:
            self.map_points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=reqd_size * elem_size)

        buf = self.map_points

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.prg.compute_points(
            queue, (shape[0] // scale_factor, shape[1] // scale_factor), (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),
            # bounds
            numpy.array(bounds, dtype=real_type),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.uint64(seed if seed is not None else random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            buf
        )

        periods = numpy.empty((shape[0] // scale_factor, shape[1] // scale_factor),
                              dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)

        self.prg.draw_periods(
            queue, shape, None,
            numpy.int32(scale_factor),
            numpy.int32(iter),
            buf,
            periods_device,
            img.dev
        )

        cl.enqueue_copy(queue, periods, periods_device)

        return img.read(queue), periods

    def _compute_fast(
            self, queue, skip, iter, z0, c, tol, bounds, root_seq, img,
            periods_shape=None, seed=None):

        if periods_shape is not None:
            shape = periods_shape
            render_image = False
        else:
            shape = img.shape
            render_image = True

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        periods = numpy.empty(shape, dtype=numpy.int32)
        periods_device = alloc_like(self.ctx, periods)

        self.prg.fast_param_map(
            queue, shape, None,
            #
            numpy.int32(1 if render_image else 0),
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),
            # bounds
            numpy.array(bounds, dtype=real_type),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.uint64(seed if seed is not None else random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            periods_device,
            #
            img.dev if render_image else self._dummy_img
        )

        cl.enqueue_copy(queue, periods, periods_device)

        return img.read(queue), periods

    def compute(self, queue, img, skip, iter, z0, c, tol, bounds, root_seq,
                method="fast", scale_factor=None, periods_shape=None, seed=None):
        if method == "fast":
            if scale_factor is not None:
                print("[warn] scale_factor is ignored")
                pass # TODO show warning that scale factor will be ignored
            return self._compute_fast(queue, skip, iter, z0, c, tol, bounds, root_seq,
                                      img, periods_shape, seed)
        if method == "precise":
            return self._compute_precise(queue, skip, iter, z0, c, tol, bounds, root_seq,
                                         scale_factor,
                                         img, periods_shape, seed)

        raise RuntimeError("Unknown method: '{}'".format(method))
