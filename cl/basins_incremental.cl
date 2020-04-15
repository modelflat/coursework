#include "newton.clh"
#include "util.clh"

kernel void capture_points_init(
    const real4 bounds,
    const real2 c,
    const real h,
    const real alpha,
    const ulong seed,
    const int seq_size,
    const global int* seq,
    global int* seq_pos, // TODO are all states always the same?
    global uint* rng_state, // TODO are all states always the same?
    global real* points
) {
    newton_state state;
    ns_init(
        &state,
        point_from_id_dense(bounds), c, h, alpha,
        seed, seq_size, seq
    );

    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    vstore2(state.rng_state, seq_start_coord, rng_state);
    vstore2(state.z, seq_start_coord, points);
    seq_pos[seq_start_coord] = state.seq_pos;
}

kernel void capture_points_next(
    const int skip,
    const real2 c,
    const real h,
    const real alpha,
    const int seq_size,
    const global int* seq,
    global int* seq_pos,
    global uint* rng_state,
    global real* points
) {
    const int2 coord = COORD_2D_INV_Y;
    const size_t seq_start_coord = coord.y * get_global_size(0) + coord.x;

    newton_state state;
    ns_init(
        &state,
        vload2(seq_start_coord, points),
        c, h, alpha,
        0, seq_size, seq
    );

    if (any(isnan(state.z))) {
        // state has turned to nan, therefore no need to iterate here anymore
        state.z.xy = NAN;
        vstore2(state.z, seq_start_coord, points);
        return;
    }

    state.rng_state = vload2(seq_start_coord, rng_state);
    state.seq_pos = seq_pos[seq_start_coord];

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    vstore2(state.rng_state, seq_start_coord, rng_state);
    vstore2(state.z, seq_start_coord, points);
    seq_pos[seq_start_coord] = state.seq_pos;
}

kernel void capture_points_finalize(
    const int skip,
    const int iter,
    const real2 c,
    const real h,
    const real alpha,
    const real tol,
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

    newton_state state;
    ns_init(
        &state,
        vload2(seq_start_coord, points),
        c, h, alpha,
        0, seq_size, seq
    );
    state.rng_state = vload2(seq_start_coord, rng_state);
    state.seq_pos = seq_pos[seq_start_coord];

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const real2 base = state.z;

    int p = iter;
    int period_ready = 0;

    for (int i = 0; i < iter; ++i) {
        ns_next(&state);
        vstore2(state.z, points_output_coord + i, points_captured);

        if (period_ready == 0 && all(fabs(base - state.z) < tol)) {
            p = i + 1;
            period_ready = 1;
        }
    }

    periods[seq_start_coord] = p;
}
