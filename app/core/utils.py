import os
import sys

import numpy
import pyopencl as cl

from PIL import Image


os.environ["PYOPENCL_NO_CACHE"] = "1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def get_endianness(ctx: cl.Context):
    de = ((dev, dev.get_info(cl.device_info.ENDIAN_LITTLE)) for dev in ctx.get_info(cl.context_info.DEVICES))
    if all(map(lambda x: x[1], de)):
        return "little"
    if all(map(lambda x: not x[1], de)):
        return "big"
    return "both"


def alloc_image(ctx: cl.Context, dim: tuple, flags=cl.mem_flags.READ_WRITE):
    endianness = get_endianness(ctx)
    if endianness == "both":
        raise RuntimeError("Context has both little and big endian devices, which is not currently supported")
    elif endianness == sys.byteorder:
        order = cl.channel_order.BGRA
    else:
        if endianness == "little":
            order = cl.channel_order.BGRA
        else:
            order = cl.channel_order.ARGB
    fmt = cl.ImageFormat(order, cl.channel_type.UNORM_INT8)
    return numpy.empty((*dim, 4), dtype=numpy.uint8), cl.Image(ctx, flags, fmt, shape=dim)


def blank_image(shape):
    return numpy.full((*shape, 4), 255, dtype=numpy.uint8)


def get_alternatives(d: dict, *alternatives):
    for alt in alternatives:
        val = d.get(alt)
        if val is not None:
            return val
    raise RuntimeError("No alternative key were found in given dict (alt: {})".format(str(alternatives)))


def create_context_and_queue(json_config: dict = None):
    if json_config is None or json_config.get("autodetect"):
        ctx = cl.create_some_context(interactive=False)
        print("Using auto-detected device:", ctx.get_info(cl.context_info.DEVICES))
    else:
        pl = cl.get_platforms()[get_alternatives(json_config, "pid", "platform", "platformId")]
        dev = pl.get_devices()[get_alternatives(json_config, "did", "device", "deviceId")]
        print("Using specified device:", dev)
        ctx = cl.Context([dev])
    return ctx, cl.CommandQueue(ctx)


def clear_image(queue, img, shape, color=(1.0, 1.0, 1.0, 1.0)):
    cl.enqueue_fill_image(
        queue, img,
        color=numpy.array(color, dtype=numpy.float32), origin=(0,)*len(shape), region=shape
    )


def read_image(queue, host_img, dev_img, shape):
    cl.enqueue_copy(
        queue, host_img, dev_img, origin=(0,)*len(shape), region=shape
    )
    return host_img


def copy_dev(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def alloc_like(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)


def read_file(path):
    with open(path) as file:
        return file.read()


def random_seed():
    return numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),


def prepare_root_seq(ctx, root_seq):
    if root_seq is None:
        seq = numpy.empty((1,), dtype=numpy.int32)
        seq[0] = -1
    else:
        seq = numpy.array(root_seq, dtype=numpy.int32)

    seq_buf = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seq)
    return seq.size if root_seq is not None else 0, seq_buf


def bgra2rgba(arr):
    arr.T[numpy.array((0, 1, 2, 3))] = arr.T[numpy.array((2, 1, 0, 3))]
    return arr


def make_img(arr):
    return Image.fromarray(bgra2rgba(arr), mode="RGBA")


class CLImg:

    host = property(lambda self: self.img[0])
    dev = property(lambda self: self.img[1])

    def __init__(self, ctx, shape):
        self.ctx = ctx
        self.img = alloc_image(ctx, shape)
        self.shape = shape

    def read(self, queue):
        read_image(queue, self.img[0], self.img[1], self.shape)
        return self.img[0]

    def clear(self, queue):
        clear_image(queue, self.img[1], self.shape)

    def as_img(self):
        return make_img(self.host.reshape((*self.shape[::-1], self.host.shape[-1])))

    def save(self, path):
        self.as_img().save(path)


real_type = numpy.float64
