#include "newton.clh"
#include "heapsort.clh"
#include "util.clh"

inline float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.25;
    }
    const float d = 1.0 / count * 8;
    switch(count % 8) {
        case 1:
            return (float3)(1.0, 0.0, 0.0)*d;
        case 2:
            return (float3)(0.0, 1.0, 0.0)*d;
        case 3:
            return (float3)(0.0, 0.0, 1.0)*d;
        case 4:
            return (float3)(1.0, 0.0, 1.0)*d;
        case 5:
            return (float3)(1.0, 1.0, 0.0)*d;
        case 6:
            return (float3)(0.0, 1.0, 1.0)*d;
        case 7:
            return (float3)(0.5, 0.0, 0.0)*d;
        default:
            return count == 8 ? .5 : d;
    }
}

// Compute samples for parameter map
kernel void compute_points(
    const real2 z0,
    const real2 c,

    const real4 bounds,

    const int skip,
    const int iter,
    const float tol,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    global ulong* result
) {
    uint2 rng_state;
    init_state(seed, &rng_state);

    // NOTE flipped y
    const int2 coord = COORD_2D_INV_Y;
    result += (coord.y * get_global_size(0) + coord.x) * iter;

    const real2 param = point_from_id_dense(bounds);

    INIT_VARIABLES(z0, c, param.x, param.y);
//    INIT_VARIABLES(z0, c, param.x, 1);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    for (int i = 0; i < iter; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
//        result[i] = float2(Z_VAR);
        result[i] = as_ulong(convert_int2_rtz(Z_VAR / tol));
    }
}

// Draw parameter map using computed samples
kernel void draw_periods(
    const int scale_factor,
    const int num_points,
    const global float* color_scheme,
    global ulong* points,
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y to correspond compute_periods
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;

    points += (coord.y * size_x + coord.x) * num_points;

    int unique = count_unique(points, num_points, 1e-4);
    float h = (unique % 23) / (float)(22);
    float v = 1.0 - (unique / (float)(num_points));

    float3 hsvcolor = (float3)(240.0 * h, 0.8, v);

    float3 color = color_for_count(unique, num_points);//hsv2rgb(hsvcolor);

    periods[coord.y * size_x + coord.x] = unique;
    // NOTE flipped y to correspond to image coordinates (top left (0,0))
    write_imagef(out, COORD_2D_INV_Y, (float4)(color, 1.0));
}


//
kernel void compute_points_lossless(
    const real2 z0,
    const real2 c,

    const real4 bounds,

    const int skip,
    const int iter,
    const float tol,

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

    // NOTE flipped y
    const int2 coord = COORD_2D_INV_Y;
    result += 2 * (coord.y * get_global_size(0) + coord.x) * iter;

    const real2 param = point_from_id_dense(bounds);

    INIT_VARIABLES(z0, c, param.x, param.y);
//    INIT_VARIABLES(z0, c, param.x, 0);

    for (int i = 0; i < skip; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
    }

    for (int i = 0; i < iter; ++i) {
        NEXT_POINT(c, seq_size, &rng_state);
        vstore2(Z_VAR, i, result);
    }
}

//
kernel void draw_periods_lossless(
    const int scale_factor,
    const int num_points,
    const global float* color_scheme,
    global real* points,
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y to correspond compute_periods
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;

    int period = periods[coord.y * size_x + coord.x];

    float h = (period % 23) / (float)(22);
    float v = 1.0 - (period / (float)(num_points));
    float3 hsvcolor = (float3)(240.0 * h, 0.8, v);

    float3 color = hsv2rgb(hsvcolor);

    // NOTE flipped y to correspond to image coordinates (top left (0,0))
    write_imagef(out, COORD_2D_INV_Y, (float4)(color, 1.0));
}
