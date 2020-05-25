#include "newton.clh"
#include "hash.clh"
#include "rotations.clh"
#include "util.clh"

// capture points and periods after `skip + iter` iterations
// TODO deprecate this
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

    vstore2(state.z, seq_start_coord, points);
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

// TODO deprecate this
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

// Round points to a certain precision
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

// Rotate all sequences to a "minimal rotation"
kernel void rotate_sequences(
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

kernel void hash_sequences(
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global real* points,
    global uint* sequence_hashes
) {
    const uint id = get_global_id(0);
    const size_t shift = id * iter;
    const uint period = periods[id];
    sequence_hashes[id] = fnv_hash(table_size - 1, period, shift, points);
}

// Count sequences using hash table approach
kernel void count_unique_sequences(
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global uint* sequence_hashes,
    global uint* table,
    global uint* table_data
) {
    const uint id = get_global_id(0);
    const uint hash = sequence_hashes[id];
    if (!atomic_inc(table + hash)) {
        table_data[hash] = id;
    }
}

// TODO current implementation does not really calculate the exact number of hash collisions,
// but rather it calculates the number of sequences with colliding hashes. Therefore it cannot
// be relied upon for the purposes other than verifying whether there were *any* collisions or not.
kernel void check_collisions(
    const real tol,
    const uint iter,
    const uint table_size,
    const global uint* periods,
    const global uint* sequence_hashes,
    const global real* points,
    const global uint* table,
    const global uint* table_data,
    global uint* collisions
) {
    const uint id = get_global_id(0);
    const uint period = periods[id];
    const size_t shift = id * iter;
    const uint hash = sequence_hashes[id];

    size_t shift_by_hash = table_data[hash] * iter;
    for (int i = 0; i < (int)period; ++i) {
        const real2 point1 = vload2(shift + i, points);
        const real2 point2 = vload2(shift_by_hash + i, points);
        if (any(fabs(point1 - point2) >= tol)) {
            // insane lock contention, but this is more of a debug code
            atomic_inc(collisions);
            break;
        }
    }
}

// Find out how many sequences of each period there are
kernel void count_periods_of_unique_sequences(
    const global uint* periods,
    const global uint* table,
    const global uint* table_data,
    global uint* period_counts
) {
    // TODO this kernel has very high parallelism but very little actual work to do
    const uint hash = get_global_id(0);
    if (table[hash] == 0) {
        return;
    }
    const int period = periods[table_data[hash]];
    if (period < 1) {
        return;
    }

    atomic_inc(period_counts + period - 1);
}

// Extract and align unique sequences into chunks sorted by period
kernel void gather_unique_sequences(
    const uint iter,
    const global uint* periods,
    const global real* points,
    const global uint* table,
    const global uint* table_data,
    const global uint* period_counts,
    const global uint* hash_positions,
    const global uint* sequence_positions,
    global uint* current_positions,
    global real* unique_sequences,
    global uint* unique_sequences_info
) {
    const uint hash = get_global_id(0);
    const uint count = table[hash];
    if (count == 0) {
        return;
    }

    const uint id = table_data[hash];
    const int period = periods[id];
    if (period < 1 || period > (int)iter - 1) {
        return;
    }

    const uint shift = id * iter;

    const uint base = (period == 1) ? 0 : sequence_positions[period - 1];
    const uint position = atomic_inc(current_positions + period - 1);

    for (int i = 0; i < period; ++i) {
        vstore2(vload2(shift + i, points), base + position * period + i, unique_sequences);
    }

    const uint hash_base = (period == 1) ? 0 : hash_positions[period - 1];
    vstore2(
        (uint2)(hash, count),
        hash_base + position, unique_sequences_info
    );
}

// Color attractors relying on their hashes
kernel void color_attractors(
    const int n,
    const global uint* hashes,
    const global float* colors,
    const global uint* hashed_points,
    write_only image2d_t image
) {
    const int2 coord = COORD_2D_INV_Y;
    const int id = coord.y * get_global_size(0) + coord.x;

    const uint hash = hashed_points[id];

    const int color_no = binary_search(n, hashes, hash);

    if (color_no != -1) {
        const float4 color = hsv2rgb(vload3(color_no, colors));
        write_imagef(image, coord, color);
    } else {
        write_imagef(image, coord, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
    }
}
