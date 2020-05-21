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
        if (any(isnan(state.z)) || any(fabs(state.z) > 1e6)) {
            p = 0;
            break;
        }
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
    int period_ready = 0;

    for (int i = 0; i < iter; ++i) {
        ns_next(&state);

        vstore2(state.z, seq_start_coord + i, points);

        if (!period_ready && (any(isnan(state.z)) || any(fabs(state.z) > 1e4))) {
            p = 0;
            period_ready = 1;
        }

        if (!period_ready && (all(fabs(base - state.z) < tol))) {
            p = i + 1;
            period_ready = 1;
        }
    }

    *periods = p;
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

//
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

// TODO this should be dynamic
#define MAX_ITER 64

#define PT_LESS(a, b) (length(a) < length(b))
#define PT_GREATER(a, b) (length(a) > length(b))
#define PT_NOT_EQUAL(a, b) (PT_LESS(a, b) || PT_GREATER(a, b))

//
kernel void count_periods(
    const global int* periods,
    global int* period_counts
) {
    const int id = get_global_id(0);
    const int period = periods[id];
    // lock contention is over 9000
    atomic_inc(period_counts + period - 1);
}

// https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation#Booth's_Algorithm
int find_minimal_rotation(int, size_t, const global real*, real);
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

void rotate_sequence(int, size_t, global real*, int);
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

kernel void round_points(
    const real tol,
    global real* points
) {
    const size_t id = get_global_id(0);
    vstore2(
        round_point_tol(vload2(id, points), tol),
        id, points
    );
}

kernel void align_rotated_sequences(
    const int iter,
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
        vstore2(vload2(shift + i, points), position + i, points_output);
    }
}

#define FNV_OFFSET_BASIS 0xcbf29ce484222325
#define FNV_PRIME 0x00000100000001B3

inline ulong fnv_1a_64(size_t len, size_t shift, const global real* points) {
    ulong hash = FNV_OFFSET_BASIS;

    for (size_t i = 0; i < len; ++i) {
        const real2 point = vload2(shift + i, points);
        for (size_t j = 0; j < sizeof(real2); ++j) {
            hash ^= *((uchar*)(&point) + j);
            hash *= FNV_PRIME;
        }
    }

    return hash;
}

kernel void hash_sequences(
    const int len,
    const global ulong* sequence_positions,
    const global real* points,
    global ulong* hashed_points
) {
    const int id = get_global_id(0);
    const size_t position = id * len + (len == 1 ? 0 : sequence_positions[len - 1]);

    const ulong hash = fnv_1a_64(len, position, points);

    hashed_points[id] = hash;
}

