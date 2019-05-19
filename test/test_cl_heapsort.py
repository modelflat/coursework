
# TODO setup proper testing for heap_sort, count_unique

import pyopencl as cl
import numpy

from core.ifs_fractal import ROOT_DIR, read_file

numpy.random.seed(42)

SOURCE = read_file(ROOT_DIR + "/include/heapsort.clh")

test_kernel = r"""

kernel void test_count_unique(
    int n,
    global ulong* data,
    global int* out
) {
    *out = count_unique(data, n, 0.0);
}

"""

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context(devices=[dev])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, SOURCE + test_kernel).build(options=["-I", ROOT_DIR + "/include", "-w"])

n = 1 << 20
arr = numpy.random.randint(0, n, n, numpy.uint64)
arr_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
out_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=arr.nbytes)

prg.test_count_unique(
    queue, (1,), None, numpy.int32(n),
    arr_dev, out_dev
)

arr_res = arr.copy()
out = numpy.empty_like(arr_res)

cl.enqueue_copy(queue, arr_res, arr_dev)
cl.enqueue_copy(queue, out, out_dev)

count = len(numpy.unique(arr))
print(arr_res)
print(count, out[0])
assert count == out[0]



test_kernel = r"""

kernel void test_heap_sort(
    int n,
    global ulong* data
) {
    heap_sort(data, n);
}

"""
