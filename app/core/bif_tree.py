import numpy
import pyopencl as cl
from tqdm import tqdm

from . import build_program_from_file
from .utils import prepare_root_seq, random_seed, alloc_like, real_type


class BifTree:

    def __init__(self, ctx):
        self.ctx = ctx
        self.prg = build_program_from_file(ctx, "bif_tree.cl")

    def compute(self, queue, img, skip, iter, z0, c, var_id, fixed_id, fixed_value, other_min, other_max,
                root_seq=None, var_min=-4, var_max=4, try_rescaling=False, seed=None):
        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        res = numpy.empty((img.shape[0], iter), dtype=real_type)
        res_dev = alloc_like(self.ctx, res)

        self.prg.compute_points_for_bif_tree(
            queue, (img.shape[0],), None,

            numpy.array((z0.real, z0.imag), dtype=real_type),
            numpy.array((c.real, c.imag), dtype=real_type),

            numpy.int32(var_id),
            numpy.int32(fixed_id),

            real_type(fixed_value),
            real_type(other_min),
            real_type(other_max),

            numpy.int32(skip),
            numpy.int32(iter),

            numpy.uint64(seed if seed is not None else random_seed()),

            numpy.int32(seq_size),
            seq,

            res_dev
        )

        if try_rescaling:
            cl.enqueue_copy(queue, res, res_dev)
            c_var_min, c_var_max = numpy.nanmin(res), numpy.nanmax(res)
            var_min = max(c_var_min, var_min)
            var_max = min(c_var_max, var_max)

        img.clear(queue)

        self.prg.draw_bif_tree(
            queue, (img.shape[0],), None,
            numpy.int32(iter),
            real_type(var_min),
            real_type(var_max),
            numpy.int32(1),
            res_dev,
            img.dev
        )

        return img.read(queue)

    def compute_tiled(self, queue, img, full_size,
                      skip, iter, z0, c, var_id, fixed_id, fixed_value, other_min, other_max,
                      root_seq=None, var_min=-4, var_max=4, try_rescaling=False, seed=None):
        if full_size[0] % img.shape[0] != 0 or full_size[0] < img.shape[0] or full_size[1] != img.shape[1]:
            raise NotImplementedError("full_size is not trivially covered by img.shape")

        n_other = full_size[0] // img.shape[0]
        other_ticks = numpy.linspace(other_min, other_max, n_other + 1 if n_other != 1 else 0)

        rest = None
        pbar = tqdm(
            desc="col loop (col={}x{})".format(img.shape[0], full_size[1]),
            total=(len(other_ticks) - 1),
            ncols=120
        )
        for tile_x, x_min, x_max in zip(range(len(other_ticks)), other_ticks, other_ticks[1:]):
            pbar.set_description("computing tile {:3d} -- bounds {:+4.4f}:{:+4.4f}".format(
                tile_x, x_min, x_max
            ))
            col = (
                self.compute(queue, img, skip, iter, z0, c, var_id, fixed_id, fixed_value,
                             x_min, x_max, root_seq, var_min, var_max, seed=seed).copy()
            )
            pbar.update()

            print(col.shape)

            if rest is None:
                rest = col
            else:
                rest = numpy.hstack((rest, col))

        return rest