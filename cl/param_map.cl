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
    const int num_points,
    global ulong* points,
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y to correspond compute_periods
    const int2 coord = COORD_2D_INV_Y;
    const int size_x = get_global_size(0);

    points += (coord.y * size_x + coord.x) * num_points;

    int unique = count_unique(points, num_points, 1e-4);

    float3 color = color_for_count(unique, num_points);

    periods[coord.y * size_x + coord.x] = unique;

    // NOTE flipped y to correspond to image coordinates (top left (0,0))
    write_imagef(out, COORD_2D_INV_Y, (float4)(color, 1.0));
}

// Compute parameter map using simple approach
kernel void fast_param_map(
    const real2 z0,
    const real2 c,

    const real4 bounds,

    const int skip,
    const int iter,
    const real tol,

    // root selection
    // seed
    const ulong seed,

    // root sequence size and contents
    const int seq_size,
    const global int* seq,

    // output
    global int* periods,
    write_only image2d_t out
) {
    // NOTE flipped y
    const int2 coord = COORD_2D_INV_Y;
    const real2 param = point_from_id_dense(bounds);

    newton_state state;
    ns_init(
        &state,
        z0, c, param.x, param.y,
        seed, seq_size, seq
    );

    int p = iter;
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
            if (any(isnan(state.z)) || any(fabs(state.z) > 1e6)) {
                p = iter;
                break;
            }
            if (all(fabs(base - state.z) < tol)) {
                p = i + 1;
                break;
            }
        }
    } else {
        p = 0;
    }

    periods[coord.y * get_global_size(0) + coord.x] = p;

    float3 color = color_for_count_old(p, iter);
    write_imagef(out, coord, (float4)(color, 1.0));
}

// Fetch color for a periods
kernel void get_color(const int total, global int* count, global float* res) {
    const int id = get_global_id(0);
    res += id * 3;

    float3 color = color_for_count_old(count[id], total);

    res[0] = color.x;
    res[1] = color.y;
    res[2] = color.z;
}
