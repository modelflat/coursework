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

// capture points and periods after `skip` iterations
kernel void capture_points_iter(
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

    size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    periods += seq_start_coord;

    seq_start_coord *= iter;

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const real2 base = state.z;

    int p = iter;

    for (int i = 0; i < iter; ++i) {
        ns_next(&state);

        vstore2(state.z, seq_start_coord + i, points);

        if (p == iter && all(fabs(base - state.z) < tol)) {
            p = i + 1;
        }
    }

    *periods = p;
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
            float3 hsv = vload3(label, colors);
            color = hsv2rgb(hsv);
            const int line_thickness = 10;
            int d = (int)((coord.y + coord.x) * M_SQRT1_2_F);
            if (d < 0) {
                d = abs(d) + line_thickness;
            }
            if (d % (line_thickness << 1) > line_thickness) {
                color.s012 *= 0.7f;
            } else {
                color.s012 *= 0.3f;
            }
        } else {
            // suspicious region
            color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        }
    } else {
        // attractor
        float3 hsv = vload3(label, colors);
        color = hsv2rgb(hsv);
    }

    write_imagef(image, coord, color);
}

kernel void prepare_region(
    const global int* labels,
    global int* label_counts
) {
    const int2 coord = COORD_2D_INV_Y;
    const int id = coord.y * get_global_size(0) + coord.x;
    const int label = labels[id];
    // NOTE: the contention for the lock here is extremely high in skewed cases!
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
    // NOTE: the contention for the lock here is extremely high in skewed cases!
    const int point_location = atomic_dec(label_counts_current + label) - 1;
    vstore2(point, shift + point_location, points_sorted);
}

kernel void color_known_attractors(
    const int iter,
    const int n_attractors,
    const global float* colors,
    const global real* attractors,
    const global real* points,
    const global int* periods,
    write_only image2d_t image
) {
    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = (coord.y * get_global_size(0) + coord.x) * iter;

    const int period = periods[(coord.y * get_global_size(0) + coord.x)];

    for (int i = 0; i < period; ++i) {
        const real2 p = vload2(seq_start_coord + i, points);
        if (any(isnan(p))) {
            break;
        }
        for (int j = 0; j < n_attractors; ++j) {
            const real2 attr = vload2(j, attractors);
            if (length(p - attr) < 0.05) {
                write_imagef(image, coord, vload4(j, colors));
                return;
            }
        }
    }

    write_imagef(image, coord, (float4)(0.0f, 0.0f, 0.0f, 1.0f));
}

kernel void count_periods(
    const global int* periods,
    global int* period_counts
) {
    const int id = get_global_id(0);
    const int period = periods[id];
    // lock contention is over 9000
    atomic_inc(period_counts + period - 1);
}

#define MAX_ITER 64

// TODO these might be overcomplicated, maybe some simpler comparison will also work?
#define PT_LESS(a, b) (length(a) < length(b))
#define PT_GREATER(a, b) (length(a) > length(b))
#define PT_NOT_EQUAL(a, b) (PT_LESS(a, b) || PT_GREATER(a, b))


// https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation#Booth's_Algorithm
int find_minimal_rotation(int n_points, size_t shift, const global real* points, real tol) {
    char failure[MAX_ITER];
    for (int i = 0; i < n_points; ++i) {
        failure[i] = -1;
    }

    int k = 0;

    for (int j = 1; j < n_points; ++j) {
        const real2 sj = vload2(shift + j, points);
        int i = failure[j - k - 1];
        real2 sj_next = vload2(shift + ((k + i + 1) % n_points), points);
        while (i != -1 && PT_NOT_EQUAL(sj, sj_next)) {
            if (PT_LESS(sj, sj_next)) {
                k = j - i - 1;
            }
            i = failure[i];
            sj_next = vload2(shift + ((k + i + 1) % n_points), points);
        }
        if (PT_NOT_EQUAL(sj, sj_next)) {
            const real2 sk = vload2(shift + (k % n_points), points);
            if (PT_LESS(sj, sk)) {
                k = j;
            }
            failure[j - k] = -1;
        } else {
            failure[j - k] = i + 1;
        }
    }

    return k;
}

void rotate_sequence(int n_points, size_t shift, global real* points, int k) {
    for (int c = 0, v = 0; c < n_points; ++v) {
        ++c;
        int target = v;
        int target_next = v + k;
        real2 tmp = vload2(shift + v, points);
        while (target_next != v) {
            ++c;
            vstore2(
                vload2(shift + target_next, points), shift + target, points
            );
            target = target_next;
            target_next = (target_next + k) % n_points;
        }
        vstore2(tmp, shift + target, points);
    }
}

kernel void rotate_captured_sequences(
    const int iter,
    const real tol,
    const global int* periods,
    global real* points
) {
    const int id = get_global_id(0);
    const int period = periods[id];

    if (period <= 1 || period >= iter) {
        return;
    }

    int start = find_minimal_rotation(period, id * iter, points, tol);

    if (start < 0) {
        printf("W: rotation was found to be -1, which likely is a result of some bug!\n");
        start = 0;
    }

    if (start == 0) {
        return;
    }

    rotate_sequence(period, id * iter, points, start);
}

kernel void align_rotated_sequences(
    const int iter,
    const real tol,
    const global int* periods,
    const global ulong* sequence_positions,
    const global real* points,
    global int* current_positions,
    global real* points_output
) {
    const int id = get_global_id(0);
    const int period = periods[id];

    if (period <= 0 || period == iter) {
        return;
    }

    const size_t shift = id * iter;
    const int seq_no = atomic_inc(current_positions + period - 1);
    const size_t position = seq_no * period + (period == 1 ? 0 : sequence_positions[period - 1]);

    for (int i = 0; i < period; ++i) {
        const real2 t = vload2(shift + i, points);
        vstore2(round_point_tol(t, tol), position + i, points_output);
    }
}
