#include "newton.clh"
#include "util.clh"


// capture points and periods after `skip + iter` iterations
kernel void capture_points(
    const int skip,
    const int iter,

    const real4 bounds,

    const real2 c,

    const real h,
    const real alpha,
    const real tol,

    const ulong seed,
    const int seq_size,
    const global int* seq,

    global real* points,
    global int* periods
) {
    newton_state state;
    ns_init(
        &state,
        point_from_id_dense(bounds), c, h, alpha,
        seed, seq_size, seq
    );

    const int2 coord = COORD_2D_INV_Y;

    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    periods += seq_start_coord;

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const real2 base = state.z;

    int p = iter;

    for (int i = 0; i < iter; ++i) {
        ns_next(&state);

        if (all(fabs(base - state.z) < tol)) {
            p = i + 1;
            break;
        }
    }

    *periods = p;

    vstore2(round_point(state.z, 4), seq_start_coord, points);
}


kernel void color_basins_section(
    const int scale_factor,
    const real4 bounds,
    const global real* endpoints,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;

    const real2 origin = point_from_id_dense(bounds);
    const real2 end = vload2(coord.y * size_x + coord.x, endpoints);

    real x_gran = (real)(1) / (get_global_size(0) - 1);
    real y_gran = (real)(1) / (get_global_size(1) - 1);
    real av_len = 0.0;
    float edge = 1.0;

    if (coord.x > 0) {
        const real2 west_end = vload2(coord.y * size_x + coord.x - 1, endpoints);
        const real dst = length(west_end - end);
        av_len += dst;
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.x < get_global_size(1) - 1) {
        const real2 east_end = vload2(coord.y * size_x + coord.x + 1, endpoints);
        const real dst = length(east_end - end);
        av_len += dst;
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y > 0) {
        const real2 north_end = vload2((coord.y - 1) * size_x + coord.x, endpoints);
        const real dst = length(north_end - end);
        av_len += dst;
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y < get_global_size(1) - 1) {
        const real2 south_end = vload2((coord.y + 1) * size_x + coord.x, endpoints);
        const real dst = length(south_end - end);
        av_len += dst;
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

    av_len /= 4;

    float mod = 0.005 / length(end - origin);
    float col = 240;
    float value = 0.8;

    real arg = atan2(end.y, end.x) + PI;

    if        (0 <= arg && arg < 2 * PI / 3) {
        col = 0;
    } else if (2 * PI / 3 <= arg && arg < 4 * PI / 3) {
        col = 60;
    } else if (4 * PI / 3 <= arg && arg < 2 * PI) {
        col = 120;
    }

    float4 color = hsv2rgb((float3)(col, value, edge));

    write_imagef(image, COORD_2D_INV_Y, color);
}


kernel void color_basins_periods(
    const int iter,
    const global int* periods,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y;

    const int size_x = get_global_size(0);

    const int end = periods[coord.y * size_x + coord.x];

    float edge = 1.0;

//    if (coord.x > 0) {
//        const int west_end = periods[coord.y * size_x + coord.x - 1];
//        if (west_end != end) {
//            edge -= 0.25f;
//        }
//    }
//
//    if (coord.x < get_global_size(1) - 1) {
//        const int east_end = periods[coord.y * size_x + coord.x + 1];
//        if (east_end != end) {
//            edge -= 0.25f;
//        }
//    }
//
//    if (coord.y > 0) {
//        const int north_end = periods[(coord.y - 1) * size_x + coord.x];
//        if (north_end != end) {
//            edge -= 0.25f;
//        }
//    }
//
//    if (coord.y < get_global_size(1) - 1) {
//        const int south_end = periods[(coord.y + 1) * size_x + coord.x];
//        if (south_end != end) {
//            edge -= 0.25f;
//        }
//    }

    float4 color = (float4)(color_for_count_old(end, iter) * edge, 1.0);

    write_imagef(image, COORD_2D_INV_Y, color);
}
