import numpy
import pyopencl as cl
import unittest


from app.core import CL_INCLUDE_PATH
from app.core.utils import read_file


TEST_KERNEL = r"""

kernel void test_count_unique(
    int n,
    global ulong* data,
    global int* out
) {
    *out = count_unique(data, n, 0.0);
}

kernel void test_heap_sort(
    int n,
    global ulong* data
) {
    heap_sort(data, n);
}


"""


class TestHeapSort(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(42)

        src = read_file(CL_INCLUDE_PATH + "/heapsort.clh")

        self.ctx = cl.Context(devices=[cl.get_platforms()[0].get_devices()[0]])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, src + TEST_KERNEL).build(options=["-I", CL_INCLUDE_PATH, "-w"])

    def test_count_unique(self):
        n = 1 << 20

        arr = numpy.random.randint(0, n, n, numpy.uint64)
        arr_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        out_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=arr.nbytes)

        self.prg.test_count_unique(
            self.queue, (1,), None, numpy.int32(n),
            arr_dev, out_dev
        )

        arr_res = arr.copy()
        out = numpy.empty_like(arr_res)

        cl.enqueue_copy(self.queue, arr_res, arr_dev)
        cl.enqueue_copy(self.queue, out, out_dev)

        count = len(numpy.unique(arr))

        # print(arr_res)
        # print(count, out[0])

        assert count == out[0]

    def test_heap_sort(self):
        n = 1 << 20

        arr = numpy.random.randint(0, n, n, numpy.uint64)
        arr_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        self.prg.test_heap_sort(
            self.queue, (1,), None,
            numpy.int32(n), arr_dev
        )

        arr_res = arr.copy()
        cl.enqueue_copy(self.queue, arr_res, arr_dev)

        assert (arr_res == numpy.sort(arr)).all()


if __name__ == '__main__':
    unittest.main()
