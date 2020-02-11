#include "newton.clh"
#include "heapsort.clh"
#include "util.clh"

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
    // NOTE flipped y
    const int2 coord = COORD_2D_INV_Y;
    result += (coord.y * get_global_size(0) + coord.x) * iter;

    const real2 param = point_from_id_dense(bounds);

    newton_state state;
    ns_init(
        &state,
        z0, c, param.x,
        //1,
        param.y,
        seed, seq_size, seq
    );

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    for (int i = 0; i < iter; ++i) {
        ns_next(&state);
        result[i] = as_ulong(convert_int2_rtz(state.z / tol));
    }
}

// Draw parameter map using computed samples
kernel void draw_periods(
    const int scale_factor,
    const int num_points,
    global ulong* points,
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y to correspond compute_periods
    const int2 coord = COORD_2D_INV_Y / scale_factor;
    const int size_x = get_global_size(0) / scale_factor;

    points += (coord.y * size_x + coord.x) * num_points;

    int unique = count_unique(points, num_points, 1e-4);

    float3 color = color_for_count(unique, num_points);

    periods[coord.y * size_x + coord.x] = unique;

    // NOTE flipped y to correspond to image coordinates (top left (0,0))
    write_imagef(out, COORD_2D_INV_Y, (float4)(color, 1.0));
}
