#include "newton.clh"
#include "util.clh"

#define POINT_RADIUS 1
#define POINT_COLOR (float4)(0.0, 0.0, 0.0, 1.0)


void put_point(write_only image2d_t, const int2, const int2);
void put_point(write_only image2d_t image, const int2 coord, const int2 image_size) {
    // brute-force non-zero radius for point
    if (POINT_RADIUS > 1) {
        for (int x = -POINT_RADIUS; x <= POINT_RADIUS; ++x) {
            for (int y = -POINT_RADIUS; y <= POINT_RADIUS; ++y) {
                const int2 coord_new = (int2)(coord.x + x, coord.y + y);
                if (in_image(coord_new, image_size) && (x*x + y*y <= POINT_RADIUS*POINT_RADIUS)) {
                    write_imagef(image, coord_new, POINT_COLOR);
                }
            }
        }
    } else if (POINT_RADIUS == 1) {
        int2 coord_new = (int2)(coord.x, coord.y);
        write_imagef(image, coord_new, POINT_COLOR);

        coord_new.x = coord.x - 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, POINT_COLOR);
        }
        coord_new.x = coord.x + 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, POINT_COLOR);
        }
        coord_new.x = coord.x;
        coord_new.y = coord.y - 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, POINT_COLOR);
        }
        coord_new.y = coord.y + 1;
        if (in_image(coord_new, image_size)) {
            write_imagef(image, coord_new, POINT_COLOR);
        }
    } else {
        write_imagef(image, coord, POINT_COLOR);
    }
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
    newton_state state;
    ns_init(
        &state,
        use_single_point ? z0 : 3 * point_from_id(bounds) / 4,
        c, h, alpha, seed, seq_size, seq
    );

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const int2 image_size = get_image_dim(out);

    for (int i = 0, frozen = 0; i < iter; ++i) {
        for (int i = 0; i < 3; ++i) {
            const int2 coord = to_size(state.roots[i], bounds, image_size);

            if (in_image(coord, image_size)) {
                put_point(out, coord, image_size);
                frozen = 0;
            } else {
                if (++frozen > 32 * 3) {
                    // this likely means that solution is going to approach infinity
                    // printf("[OCL] error at slave %d: frozen!\n", get_global_id(0));
                    break;
                }
            }
        }

        ns_next(&state);
    }
}
