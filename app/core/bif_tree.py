import numpy
import pyopencl as cl

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type


class BifTree:

    def __init__(self, ctx, img):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "bif_tree.cl")
        self.img = img

    def _compute_bif_tree(self, queue, skip, iter, z0, c, var_id, fixed_id, fixed_value, other_min, other_max, img,
                          root_seq=None):

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        res = numpy.empty((img.shape[0], iter), dtype=real_type)
        res_dev = alloc_like(self.ctx, res)

        self.prg.compute_points_for_bif_tree(
            queue, (img.shape[0],), None,
            # z0
            numpy.array((z0.real, z0.imag), dtype=real_type),
            # c
            numpy.array((c.real, c.imag), dtype=real_type),

            numpy.int32(var_id),
            numpy.int32(fixed_id),

            real_type(fixed_value),
            real_type(other_min),
            real_type(other_max),

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.uint64(random_seed()),

            numpy.int32(seq_size),
            seq,

            res_dev
        )

        return res, res_dev

    def _render_bif_tree(self, queue, iter, var_min, var_max, result_buf, img):
        img.clear(queue)

        self.prg.draw_bif_tree(
            queue, (img.shape[0],), None,
            numpy.int32(iter),
            real_type(var_min),
            real_type(var_max),
            numpy.int32(1),
            result_buf,
            img.dev
        )

        return img.read(queue)

    def compute(self, queue, skip, iter, z0, c, var_id, fixed_id, fixed_value, other_min, other_max,
                img=None, root_seq=None, var_min=-4, var_max=4, try_rescaling=False):
        if img is None:
            img = self.img

        res, res_dev = self._compute_bif_tree(queue, skip, iter, z0, c, var_id,
                                              fixed_id, fixed_value, other_min, other_max,
                                              img,
                                              root_seq)
        if try_rescaling:
            cl.enqueue_copy(queue, res, res_dev)
            # c_var_min, c_var_max = numpy.nanmin(res, axis=1), numpy.nanmax(res, axis=1)
            #
            # print(c_var_min, c_var_max)
            #
            # var_min = max(numpy.nanmedian(c_var_min), var_min)
            # var_max = min(numpy.nanmedian(c_var_max), var_max)

            # print(var_min, var_max)
            # print(res)
            raise NotImplementedError()

        return self._render_bif_tree(queue, iter, var_min, var_max, res_dev, img)