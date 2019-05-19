#include "newton.clh"
#include "util.clh"

//
kernel void compute_points_for_bif_tree(
    const real2 z0,
    const real2 c,

    const int var_id,
    const int fixed_param_id,

    const real fixed_param,
    const real other_param_min,
    const real other_param_max,

    const int skip,
    const int iter,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    global real* result
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    const int id = get_global_id(0);
    result += id * iter;

    const real other_param = other_param_min + id * (other_param_max - other_param_min) / get_global_size(0);

    INIT_VARIABLES(z0, c,
        fixed_param_id == 0 ? fixed_param : other_param,
        fixed_param_id == 1 ? fixed_param : other_param
    );

//    printf("%f %f %f %f")

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    for (int i = 0; i < iter; ++i) {
        result[i] = (var_id == 0) ? Z_VAR.x : Z_VAR.y;
        NEXT_POINT(c, seq_size, &rng_state);
    }
}

inline void write_point(
    const real x, const real x_min, const real x_max, const int l_id,
    const int flip_y,
    write_only image2d_t out
) {
    if (x > x_max || x < x_min || isnan(x)) return;
    const int h = get_image_height(out) - 1;

    int x_coord = convert_int_rtz((x - x_min) / (x_max - x_min) * h);
    write_imagef(out, (int2)(l_id, (flip_y) ? h - x_coord : x_coord), (float4)(0.0, 0.0, 0.0, 1.0));
}

kernel void draw_bif_tree(
    const int iter,
    const real x_min, const real x_max,
    const int flip_y,
    const global real* data,
    write_only image2d_t img
) {
    const int id = get_global_id(0);
    data += id * iter;
    for (int i = 0; i < iter; ++i) {
        write_point(data[i], x_min, x_max, id, flip_y, img);
    }
}
