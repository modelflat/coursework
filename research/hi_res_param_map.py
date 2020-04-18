from datetime import datetime

import numpy

from config import C
from core.param_map import ParameterMap
from core.utils import create_context_and_queue, CLImg

ctx, queue = create_context_and_queue()
par = ParameterMap(ctx)


bounds_top_left = (-6, 0, 0.5, 1.0)
# bounds_top_left = (-3.75, -3 + -0.75 / 4, 0.75 - 0.25 / 4 + 0.25 / 16, 0.75)
bounds_bottom_right = (0, 6, 0.0, 0.5)


def compute_hi_res_param_map(filename, bounds, root_sequence, mode="tile"):
    TS = datetime.now().strftime("%Y%m%d-%H%M%S")

    full_size = (12000, 16000)
    # full_size = (12000, 12000)
    skip = 1 << 12
    iter = 1 << 6

    if mode == "inc":
        image = CLImg(ctx, (full_size[0], full_size[1] // 2))
        skip_batch_size = 1 << 8

        periods_part_1, _ = par.compute_incremental(
            queue, image,
            skip=skip, skip_batch_size=skip_batch_size, iter=iter,
            z0=complex(0.5, 0.0), c=C,
            bounds=(bounds[0], bounds[1], bounds[2], bounds[2] + (bounds[3] - bounds[2]) / 2),
            root_seq=root_sequence,
            seed=42, tolerance_decimals=3, capture_points=False, draw_image=True
        )
        image_part_1 = image.host.copy()

        periods_part_2, _ = par.compute_incremental(
            queue, image,
            skip=skip, skip_batch_size=skip_batch_size, iter=iter,
            z0=complex(0.5, 0.0), c=C,
            bounds=(bounds[0], bounds[1], bounds[2] + (bounds[3] - bounds[2]) / 2, bounds[3]),
            root_seq=root_sequence,
            seed=42, tolerance_decimals=3, capture_points=False, draw_image=True
        )
        image_part_2 = image.host.copy()

        image = numpy.concatenate((image_part_2, image_part_1), axis=0)
        periods = numpy.concatenate((periods_part_2, periods_part_1), axis=0)
    else:
        img = CLImg(ctx, (256, 256))
        image, periods, _ = par.compute_tiled(
            queue, img, full_size,
            skip=skip, iter=iter,
            z0=complex(0.5, 0.0),
            c=C,
            bounds=bounds,
            root_seq=root_sequence,
            seed=42, tol=3
        )

    im = CLImg(ctx, (1, 1))
    im.shape = full_size
    im.img = (image, image)
    im.save(f"{filename}_{TS}.png")
    numpy.save(f"{filename}_periods_{TS}.npy", periods)


compute_hi_res_param_map("final_0_inc", bounds_top_left, (0,), mode="inc")
compute_hi_res_param_map("final_01_inc", bounds_top_left, (0, 1), mode="inc")
compute_hi_res_param_map("final_001_inc", bounds_top_left, (0, 0, 1), mode="inc")
compute_hi_res_param_map("final_0001_inc", bounds_top_left, (0, 0, 0, 1), mode="inc")
compute_hi_res_param_map("final_00001_inc", bounds_top_left, (0, 0, 0, 0, 1), mode="inc")
compute_hi_res_param_map("final_00000001_inc", bounds_top_left, (0, 0, 0, 0, 0, 0, 0, 1), mode="inc")

compute_hi_res_param_map("final_0_inc", bounds_bottom_right, (0,), mode="inc")
compute_hi_res_param_map("final_01_inc", bounds_bottom_right, (0, 1), mode="inc")
compute_hi_res_param_map("final_001_inc", bounds_bottom_right, (0, 0, 1), mode="inc")
compute_hi_res_param_map("final_0001_inc", bounds_bottom_right, (0, 0, 0, 1), mode="inc")
compute_hi_res_param_map("final_00001_inc", bounds_bottom_right, (0, 0, 0, 0, 1), mode="inc")
compute_hi_res_param_map("final_00000001_inc", bounds_bottom_right, (0, 0, 0, 0, 0, 0, 0, 1), mode="inc")
