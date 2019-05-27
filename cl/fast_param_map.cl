#include "newton.clh"
#include "heapsort.clh"
#include "util.clh"

inline float3 _color_for_count(int count, int total) {
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

kernel void fast_param_map(
    const int render_image,
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

    for (int i = 0; i < skip; ++i) {
        ns_next(&state);
    }

    const real2 base = state.z;

    int p = iter;
    for (int i = 0; i < iter; ++i) {
        ns_next(&state);
        if (all(fabs(base - state.z) < tol)) {
            p = i;
            break;
        }
    }

    periods[coord.y * get_global_size(0) + coord.x] = p + 1;

    if (render_image) {
        float3 color = _color_for_count(p + 1, iter);
        write_imagef(
            out, coord, (float4)(color, 1.0)
        );
    }
}