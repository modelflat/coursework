constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

inline int is_empty(float4 color) {
    return color.x > 0;
}

inline int scan_col(int x, image2d_t img) {
    int h = get_image_height(img);
    for (int2 coord = {x, 0}; coord.y < h; ++(coord.y)) {
        if (!is_empty(read_imagef(img, sampler, coord))) {
            return 1;
        }
    }
    return 0;
}

inline int scan_row(int y, image2d_t img) {
    int w = get_image_width(img);
    for (int2 coord = {0, y}; coord.x < w; ++coord.x) {
        if (!is_empty(read_imagef(img, sampler, coord))) {
            return 1;
        }
    }
    return 0;
}

// NB: This kernel is more suitable for use with CPU device
kernel void compute_bounding_box(
    read_write image2d_t image,
    global int* bounding_box
) {
    const int id = get_global_id(0);
    const int2 im = get_image_dim(image);

    switch (id % 4) {
        case 0:
            for (int x = 0; x < im.x; ++x) {
                if (scan_col(x, image)) {
                    bounding_box[0] = x;
                    break;
                }
            }
            break;
        case 1:
            for (int x = im.x - 1; x >= 0; --x) {
                if (scan_col(x, image)) {
                    bounding_box[1] = x;
                    break;
                }
            }
            break;
        case 2:
            for (int y = im.y - 1; y >= 0; --y) {
                if (scan_row(y, image)) {
                    bounding_box[2] = y;
                    break;
                }
            }
            break;
        case 3:
            for (int y = 0; y < im.y; ++y) {
                if (scan_row(y, image)) {
                    bounding_box[3] = y;
                    break;
                }
            }
            break;
        default: {}
    }
}
