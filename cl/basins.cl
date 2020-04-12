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

// color computed basins by section of a phase surface to which attractor belongs to
kernel void color_basins_section(
    const real4 bounds,
    const global real* points,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y;
    const int size_x = get_global_size(0);

    const real2 end = vload2(coord.y * size_x + coord.x, points);

    real x_gran = (real)(1) / (get_global_size(0) - 1);
    real y_gran = (real)(1) / (get_global_size(1) - 1);
    float edge = 1.0;

    if (coord.x > 0) {
        const real2 west_end = vload2(coord.y * size_x + coord.x - 1, points);
        const real dst = length(west_end - end);
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.x < (int)get_global_size(1) - 1) {
        const real2 east_end = vload2(coord.y * size_x + coord.x + 1, points);
        const real dst = length(east_end - end);
        if (dst > x_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y > 0) {
        const real2 north_end = vload2((coord.y - 1) * size_x + coord.x, points);
        const real dst = length(north_end - end);
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

    if (coord.y < (int)get_global_size(1) - 1) {
        const real2 south_end = vload2((coord.y + 1) * size_x + coord.x, points);
        const real dst = length(south_end - end);
        if (dst > y_gran) {
            edge -= 0.25f;
        }
    }

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

// color computed basins by period
kernel void color_basins_periods(
    const int iter,
    const global int* periods,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y;
    const int size_x = get_global_size(0);

    const int end = periods[coord.y * size_x + coord.x];
    float4 color = (float4)(color_for_count_old(end, iter), 1.0);

    write_imagef(image, COORD_2D_INV_Y, color);
}

// color computed basins by period, and draw bounds between different attractors
kernel void color_basins_periods_attractors(
    const int iter,
    const real attractor_min_dist,
    const real4 bounds,
    const global int* periods,
    const global real* points,
    global uchar* contours,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y;
    const int size_x = get_global_size(0);

    const real2 attractor = vload2(coord.y * size_x + coord.x, points);

    contours += coord.y * size_x + coord.x;

    float edge = 1.0;

    if (coord.x > 0) {
        const real2 west_end = vload2(coord.y * size_x + coord.x - 1, points);
        const real dst = length(west_end - attractor);
        if (attractor_min_dist < dst) {
            edge -= 0.25f;
        }
    }

    if (coord.x < (int)get_global_size(1) - 1) {
        const real2 east_end = vload2(coord.y * size_x + coord.x + 1, points);
        const real dst = length(east_end - attractor);
        if (attractor_min_dist < dst) {
            edge -= 0.25f;
        }
    }

    if (coord.y > 0) {
        const real2 north_end = vload2((coord.y - 1) * size_x + coord.x, points);
        const real dst = length(north_end - attractor);
        if (attractor_min_dist < dst) {
            edge -= 0.25f;
        }
    }

    if (coord.y < (int)get_global_size(1) - 1) {
        const real2 south_end = vload2((coord.y + 1) * size_x + coord.x, points);
        const real dst = length(south_end - attractor);
        if (attractor_min_dist < dst) {
            edge -= 0.25f;
        }
    }

    float4 color;
    if (edge < 1.0) {
        *contours = 0;
        color = (float4)(0.95, 0.95, 0.95, 1.0);
    } else {
        *contours = 255;
        const int period = periods[coord.y * size_x + coord.x];
        color = (float4)(color_for_count_old(period, iter), 1.0);
    }

    write_imagef(image, COORD_2D_INV_Y, color);
}

//
kernel void color_basins_attractors(
    const global real* points,
    const global real* data,
    const global int* labels,
    const global float* colors,
    const global int* flags,
    write_only image2d_t image
) {
    // NOTE y flipped to correspond to compute_basins
    const int2 coord = COORD_2D_INV_Y;
    const int size_x = get_global_size(0);

    const int label = labels[coord.y * size_x + coord.x];

    const int flag = flags[label];

    float4 color;
    if (flag == 0) {
        // skipped
        color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    } else if ((flag & 1) != 0) {
        if ((flag & 2) != 0) {
            // suspicious attractor
            // TODO
            float3 hsv = vload3(label, colors);
            color = hsv2rgb(hsv);
        } else {
            // suspicious region
            real8 d = vload8(label, data);
            (void)d;
            color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        }
    } else {
        // attractor
        float3 hsv = vload3(label, colors);
        color = hsv2rgb(hsv);
    }

    write_imagef(image, COORD_2D_INV_Y, color);
}

kernel void prepare_region(
    const global int* labels,
    global int* label_counts
) {
    const int2 coord = COORD_2D_INV_Y;
    const int id = coord.y * get_global_size(0) + coord.x;
    const int label = labels[id];
    // NOTE: the contention of the lock here is extremely high in skewed cases!
    atomic_inc(label_counts + label);
}

kernel void sort_points_by_region(
    const global real* points,
    const global int* labels,
    const global int* label_end_indexes,
    global int* label_counts_current,
    global real* points_sorted
) {
    const int2 coord = COORD_2D_INV_Y;
    const int id = coord.y * get_global_size(0) + coord.x;
    const real2 point = vload2(id, points);
    const int label = labels[id];
    const int shift = (label == 0) ? 0 : label_end_indexes[label - 1];
    // NOTE: the contention of the lock here is extremely high in skewed cases!
    const int point_location = atomic_dec(label_counts_current + label) - 1;
    vstore2(point, shift + point_location, points_sorted);
}
