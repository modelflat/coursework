import numpy
import pyopencl as cl

from core.ifs_fractal import CL_INCLUDE_PATH, read_file, copy_dev, alloc_like

numpy.random.seed(42)

SOURCE = read_file(CL_INCLUDE_PATH + "/complex_v2.clh")

TEST_KERNEL = r"""
kernel void test_cmul(
global real2* a, global real2* b, global real2* c
) {
    int id = get_global_id(0);
    c[id] = cmul(a[id], b[id]);
}

kernel void test_cdiv(
global real2* a, global real2* b, global real2* c
) {
    int id = get_global_id(0);
    c[id] = cdiv(a[id], b[id]);
}

kernel void test_csq(
global real2* a, global real2* c
) {
    int id = get_global_id(0);
    c[id] = csq(a[id]);
}

kernel void test_ccb(
global real2* a, global real2* c
) {
    int id = get_global_id(0);
    c[id] = ccb(a[id]);
}

kernel void test_csqrt1(
global real2* a, global real2* c
) {
    int id = get_global_id(0);
    c[id] = csqrt1(a[id]);
}

kernel void test_ccbrt1(
global real2* a, global real2* c
) {
    int id = get_global_id(0);
    c[id] = ccbrt1(a[id]);
}

kernel void test_solve_equations(
global real* a,
global real* b, 
global real* c,
global real* result,
global int* err_code
) {
    const int id = get_global_id(0);
    a += 2 * id;
    b += 2 * id;
    c += 2 * id;
    result += 6 * id;
    err_code += id;
    {
        real2 a_ = {a[0], a[1]};
        real2 b_ = {b[0], b[1]};
        real2 c_ = {c[0], c[1]};
        real2 roots[3];
        if (id == 0) {
            // printf("%f,%f %f,%f %f,%f\n", a_.x, a_.y, b_.x, b_.y, c_.x, c_.y);
        }
        *err_code = solve_cubic(a_, b_, c_, roots);
        // solve_depressed_cubic(b_, c_, roots);
        result[0] = roots[0].x;
        result[1] = roots[0].y;
        result[2] = roots[1].x;
        result[3] = roots[1].y;
        result[4] = roots[2].x;
        result[5] = roots[2].y;
    }
}
"""

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context(devices=[dev])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, SOURCE + TEST_KERNEL).build(options=["-I", CL_INCLUDE_PATH, "-w"])


def call_solve_equations(a, b, c):
    assert len(a) == len(b) == len(c)
    n = len(a)
    a_dev = copy_dev(ctx, a.view(numpy.float64))
    b_dev = copy_dev(ctx, b.view(numpy.float64))
    c_dev = copy_dev(ctx, c.view(numpy.float64))

    res = numpy.empty((n, 3, 2), dtype=numpy.float64)
    res_dev = alloc_like(ctx, res)
    err = numpy.empty((n,), dtype=numpy.int32)
    err_dev = alloc_like(ctx, err)
    prg.test_solve_equations(
        queue, (n,), None,
        a_dev, b_dev, c_dev, res_dev, err_dev
    )
    cl.enqueue_copy(queue, res, res_dev)
    cl.enqueue_copy(queue, err, err_dev)
    return res.view(numpy.complex128), err


def test_single(a: complex, b: complex, c: complex):
    res, err = call_solve_equations(numpy.array([a]), numpy.array([b]), numpy.array([c]))
    res = res.flatten()
    return res[0], res[1], res[2]


def test_cmul(n=1 << 16):
    a_arr = numpy.random.random(size=(n, 2))
    a_arr_dev = copy_dev(ctx, a_arr)
    b_arr = numpy.random.random(size=(n, 2))
    b_arr_dev = copy_dev(ctx, b_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_cmul(
        queue, (n,), None,
        a_arr_dev, b_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    b = b_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)

    assert numpy.allclose(c, a * b)


def test_cdiv(n=1 << 16):
    a_arr = 2*numpy.random.random(size=(n, 2)) - 1
    a_arr_dev = copy_dev(ctx, a_arr)
    b_arr = 0.01 + numpy.random.random(size=(n, 2))
    b_arr_dev = copy_dev(ctx, b_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_cdiv(
        queue, (n,), None,
        a_arr_dev, b_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    b = b_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)

    assert numpy.allclose(c, a / b)


def test_csq(n=1 << 16):
    a_arr = 16 * numpy.random.random(size=(n, 2)) - 8
    a_arr_dev = copy_dev(ctx, a_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_csq(
        queue, (n,), None,
        a_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)

    assert numpy.allclose(c, a * a)


def test_ccb(n=1 << 16):
    a_arr = 16 * numpy.random.random(size=(n, 2)) - 8
    a_arr_dev = copy_dev(ctx, a_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_ccb(
        queue, (n,), None,
        a_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)

    assert numpy.allclose(c, a * a * a)


def test_csqrt1(n=1 << 16):
    a_arr = 16 * numpy.random.random(size=(n, 2)) - 8
    a_arr_dev = copy_dev(ctx, a_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_csqrt1(
        queue, (n,), None,
        a_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)
    #
    # print(numpy.sqrt(a))
    # print(c)

    assert numpy.allclose(c, numpy.sqrt(a))


def test_ccbrt1(n=1 << 16):
    a_arr = 16 * numpy.random.random(size=(n, 2)) - 8
    a_arr_dev = copy_dev(ctx, a_arr)
    c_arr = numpy.empty((n, 2), numpy.float64)
    c_arr_dev = alloc_like(ctx, c_arr)

    prg.test_ccbrt1(
        queue, (n,), None,
        a_arr_dev, c_arr_dev
    )

    cl.enqueue_copy(queue, c_arr, c_arr_dev)

    a = a_arr.view(numpy.complex128)
    c = c_arr.view(numpy.complex128)

    # print(numpy.sqrt(a))
    # print(c)

    assert numpy.allclose(c, (a)**(1/3))


def format_eq(eq):
    return "z{p}3 + {}{m}z{p}2 + {}{m}z + {} = 0".format(*eq, p="^", m=" ")


def compare_solutions(s1, s2, print_result=True):
    s1 = numpy.sort(numpy.array(s1).flatten())
    s2 = numpy.sort(numpy.array(s2).flatten())
    assert len(s1) == len(s2) == 3, "solutions should be equal in length and have 3 elements"
    if print_result:
        print("1> " + str(s1), "2> " + str(s2), sep="\n")
    assert numpy.allclose(s1, s2), "solutions are not close enough"


def test_solve_cubic_arbitrary(n=1 << 10):
    a_arr = 10*numpy.random.random(size=(n, 2)) - 5
    b_arr = 10*numpy.random.random(size=(n, 2)) - 5
    c_arr = 10*numpy.random.random(size=(n, 2)) - 5

    res, err = call_solve_equations(a_arr, b_arr, c_arr)

    a_arr = a_arr.view(numpy.complex128).flatten()
    b_arr = b_arr.view(numpy.complex128).flatten()
    c_arr = c_arr.view(numpy.complex128).flatten()

    for a, b, c, z, e in zip(a_arr, b_arr, c_arr, res, err):
        eq = a, b, c
        # print(format_eq(eq), "//", e)
        sol = numpy.roots([complex(1.0), a, b, c])
        compare_solutions(z, sol, print_result=False)


def test_solve_cubic():
    s1 = test_single(complex(-6.0), complex(11.0), complex(-6.0))
    s2 = numpy.roots([complex(1.0), complex(-6.0), complex(11.0), complex(-6.0)])

    compare_solutions(s1, s2, print_result=False)

    s1 = test_single(complex(0), -complex(36, 12), complex(126, -117))
    s2 = numpy.roots([complex(1.0), complex(0), -complex(36, 12), complex(126, -117)])

    compare_solutions(s1, s2, print_result=False)

test_cmul()
test_cdiv()
test_csq()
test_ccb()
test_csqrt1()
test_ccbrt1()
test_solve_cubic_arbitrary()
test_solve_cubic()

