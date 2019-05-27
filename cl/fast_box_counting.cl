constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

inline bool is_not_empty(uint4 color) {
    return color.x == 0 && color.y == 0 && color.z == 0;
}

inline ulong intercalate(uint2 coord) {
    ulong v = 0;
    for (uint i = 0, mask = 1; i < 32; mask <<= 1, ++i) {
        v |= (((coord.x & mask) << (i + 1)) | (coord.y & mask) << i);
    }
    return v;
}

kernel void intercalate_coord(
    const ulong empty_box_inted_value,
    read_only image2d_t image,
    global ulong* intercalated_coords
) {
    const int2 coord = { get_global_id(0), get_global_id(1) };
    intercalated_coords += coord.y * get_global_size(0) + coord.x;

    const uint4 color = read_imageui(image, sampler, coord);
    if (is_not_empty(color)) {
        *intercalated_coords = intercalate(as_uint2(coord));
    } else {
        *intercalated_coords = empty_box_inted_value;
    }
}

kernel void count_non_empty(
    const ulong empty_box_inted_value,
    const int box_size,
    const uint strip_bits,
    const int intercalated_coords_len,
    const global ulong* intercalated_coords,
    global int* black_count,
    global int* gray_count
) {
    const uint2 coord = { get_global_id(0), get_global_id(1) };
    const int flat_coord = coord.y * get_global_size(0) + coord.x;

    ulong value = intercalated_coords[flat_coord];
    if (value != empty_box_inted_value) {
        value >>= strip_bits;
        if (flat_coord == 0 || (intercalated_coords[flat_coord - 1] >> strip_bits) != value) {
            // the first element of chunk
            if (flat_coord + (box_size - 1) < intercalated_coords_len) {
                if (value == (intercalated_coords[flat_coord + (box_size - 1)] >> strip_bits)) {
                    // black box
                    atomic_inc(black_count);
                } else {
                    // gray box
                    atomic_inc(gray_count);
                }
            } else {
                // gray box
                atomic_inc(gray_count);
            }
        }
    }
}