#include "complex_v2.clh"
#include "random.clh"

#define POINT_RADIUS 1
#define POINT_COLOR (float4)(0.0, 0.0, 0.0, 1.0)
#include "util.clh"

typedef struct {
    real2 z, c;
    real h, alpha;
    uint2 rng_state;
    int seq_pos, seq_size;
    const global int* seq;
} newton_state;

#define RANDOM_ROOT(rng_state) ((as_uint(random(rng_state)) >> 3) % 3)

void next_point_v2(newton_state* state, real2 roots[3]) {
    real eps = -state->h * (1 - state->alpha) / 3;

    real2 a = -(state->z + state->alpha * state->h * (
        (ccb(state->z) + state->c) / (3 * csq(state->z))
    )) / (1 + eps);
    real2 b = (real2)(0, 0);
    real2 c = state->c * eps / (1 + eps);

    solve_cubic(a, b, c, roots);

    int root;
    if (state->seq_size > 0) {
        state->seq_pos %= state->seq_size;
        root = state->seq[state->seq_pos++];
    } else {
        root = RANDOM_ROOT(&state->rng_state);
    }

    state->z = roots[root];
}

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

    newton_state state = {
        use_single_point ? z0 : point_from_id(bounds) / 1.5,
        c, h, alpha, rng_state, 0, seq_size, seq
    };
    real2 roots[3];
    roots[0] = 0; roots[1] = 0; roots[2] = 0;

    for (int i = 0; i < skip; ++i) {
        next_point_v2(&state, roots);
    }

    const int2 image_size = get_image_dim(out);

    for (int i = 0, frozen = 0; i < iter; ++i) {
        for (int i = 0; i < 3; ++i) {
            const int2 coord = to_size(roots[i], bounds, image_size);

            if (in_image(coord, image_size) && !near_zero(roots[i], 1e-6)) {
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

        next_point_v2(&state, roots);
    }
}

