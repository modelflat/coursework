#include "newton.clh"

#define POINT_RADIUS 1
#define POINT_COLOR (float4)(0.0, 0.0, 0.0, 1.0)
#define DETECTION_PRECISION 1e-4
#include "util.clh"

inline int near_zero(real2 p, real tol) {
    return length(p) < tol;
}

// Draw Newton fractal (phase plot)
kernel void newton_fractal(
    // algo parameters
    const int skip,
    const int iter,

    // plane bounds
    const real4 bounds,

    // fractal parameters
    const real2 c,
    const real h,
    const real alpha,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    // starting point
    const int use_single_point,
    const real2 z0,

    // output
    write_only image2d_t out
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    INIT_VARIABLES(use_single_point ? z0 : point_from_id(bounds) / 1.5, c, h, alpha);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    const int2 image_size = get_image_dim(out);

    for (int i = 0, frozen = 0; i < iter; ++i) {
        for (int i = 0; i < 3; ++i) {
            const int2 coord = to_size(_ROOTS[i], bounds, image_size);

            if (in_image(coord, image_size) && !near_zero(_ROOTS[i], 1e-6)) {
                put_point(out, coord, image_size);
                frozen = 0;
            } else {
                if (++frozen > 32) {
                    // this likely means that solution is going to approach infinity
                    // printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                    break;
                }
            }
        }

        NEXT_POINT(c, seq_size, &rng_state);
    }
}

