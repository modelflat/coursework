#include "newton.clh"
#include "util.clh"

// TODO this duplicates much of the code from basins_incremental.cl, what can we do about it?

kernel void capture_points_init(
    const real4 bounds,
    const real2 c,
    const real2 z0,
    const ulong seed,
    const int seq_size,
    const global int* seq,
    global int* seq_pos, // TODO are all states always the same?
    global uint* rng_state, // TODO are all states always the same?
    global real* points
) {
    const real2 param = point_from_id_dense(bounds);

    newton_state state;
    ns_init(&state, z0, c, param.x, param.y, seed, seq_size, seq);

    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    vstore2(state.z, seq_start_coord, points);
    if (coord.x == 0 && coord.y == 0) {
        vstore2(state.rng_state, 0, rng_state);
    }
    seq_pos[seq_start_coord] = state.seq_pos;
}

kernel void capture_points_next(
    const real4 bounds,
    const real2 c,
    const int skip,
    const int seq_size,
    const global int* seq,
    global int* seq_pos,
    global uint* rng_state,
    global real* points
) {
    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    const real2 param = point_from_id_dense(bounds);

    newton_state state;
    ns_init(&state, vload2(seq_start_coord, points), c, param.x, param.y, 0, seq_size, seq);

    state.rng_state = vload2(0, rng_state);
    state.seq_pos = seq_pos[seq_start_coord];

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    if (coord.x == 0 && coord.y == 0) {
        vstore2(state.rng_state, 0, rng_state);
    }
    vstore2(state.z, seq_start_coord, points);
    seq_pos[seq_start_coord] = state.seq_pos;
}

kernel void capture_points_finalize(
    const real4 bounds,
    const real2 c,
    const real tol,
    const int skip,
    const int iter,
    const int capture_points,
    const int seq_size,
    const global int* seq,
    global int* seq_pos,
    global uint* rng_state,
    global real* points,
    global real* points_captured,
    global int* periods
) {
    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;
    const size_t points_output_coord = seq_start_coord * iter;
    const real2 param = point_from_id_dense(bounds);

    newton_state state;
    ns_init(&state, vload2(seq_start_coord, points), c, param.x, param.y, 0, seq_size, seq);
    state.rng_state = vload2(0, rng_state);
    state.seq_pos = seq_pos[seq_start_coord];

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const real2 base = state.z;

    int p = iter;
    int period_ready = 0;

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
        if (any(isnan(state.z)) || any(fabs(state.z) > 1e6)) {
            p = 0;
            break;
        }
    }

    if (p != 0) {
        const real2 base = state.z;
        p = iter;
        for (int i = 0; i < iter; ++i) {
            ns_next(&state);

            if (capture_points) {
                vstore2(state.z, points_output_coord + i, points_captured);
            }

            if (!period_ready && (any(isnan(state.z)) || any(fabs(state.z) > 1e6))) {
                p = iter;
                period_ready = 1;
                if (!capture_points) {
                    break;
                }
            }

            if (!period_ready && (all(fabs(base - state.z) < tol))) {
                p = i + 1;
                period_ready = 1;
                if (!capture_points) {
                    break;
                }
            }
        }
    } else {
        p = 0;
    }

    periods[seq_start_coord] = p;
}


kernel void color_param_map(
    const int iter,
    const global int* periods,
    write_only image2d_t out
) {
    const int2 coord = COORD_2D_INV_Y;
    float3 color = color_for_count_old(periods[coord.y * get_global_size(0) + coord.x], iter);
    write_imagef(out, coord, (float4)(color, 1.0));
}
