import os

import numpy

from tqdm import tqdm

from app.config import C
from app.core.param_map import ParameterMap
from app.core.utils import create_context_and_queue, CLImg

ctx, queue = create_context_and_queue()

solver = ParameterMap(ctx)

SIZE = (12288, 16384)
SKIP = 100000
SKIP_BATCH = 1000
ITER = 1 << 6
Z0 = complex(0.5, 0.0)
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../COURSEWORK_DATA/PARAMETER_MAPS"))

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(OUTPUT_PATH)


bounds_top_left = (-6, 0, 0.5, 1.0)
bounds_bottom_right = (0, 6, 0.0, 0.5)

script = [
    ("top_left/map_0", bounds_top_left, (0,)),
    ("top_left/map_01", bounds_top_left, (0, 1)),
    ("top_left/map_001", bounds_top_left, (0, 0, 1)),
    ("top_left/map_0001", bounds_top_left, (0, 0, 0, 1)),
    ("top_left/map_00001", bounds_top_left, (0, 0, 0, 0, 1)),
    ("top_left/map_00000001", bounds_top_left, (0, 0, 0, 0, 0, 0, 0, 1)),
    ("bottom_right/map_0", bounds_bottom_right, (0,)),
    ("bottom_right/map_01", bounds_bottom_right, (0, 1)),
    ("bottom_right/map_001", bounds_bottom_right, (0, 0, 1)),
    ("bottom_right/map_0001", bounds_bottom_right, (0, 0, 0, 1)),
    ("bottom_right/map_00001", bounds_bottom_right, (0, 0, 0, 0, 1)),
    ("bottom_right/map_00000001", bounds_bottom_right, (0, 0, 0, 0, 0, 0, 0, 1)),
]


def compute_high_res_param_map(filename, bounds, root_sequence, parts=4):
    assert SIZE[1] % parts == 0

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(f"{filename}.png"):
        print(f"Skipping {filename} as it already exists ...")
        return

    image = CLImg(ctx, (SIZE[0], SIZE[1] // parts))

    image_parts = []
    periods_parts = []

    for part in range(parts):
        periods_part, _ = solver.compute_incremental(
            queue, image,
            skip=SKIP, skip_batch_size=SKIP_BATCH, iter=ITER, z0=Z0, c=C,
            bounds=(
                bounds[0],
                bounds[1],
                bounds[2] + part * (bounds[3] - bounds[2]) / parts,
                bounds[2] + (part + 1) * (bounds[3] - bounds[2]) / parts,
            ),
            root_seq=root_sequence, seed=42, tolerance_decimals=3, capture_points=False, draw_image=True
        )
        periods_parts.append(periods_part)
        image_parts.append(image.host.copy())

    image = numpy.concatenate(image_parts[::-1], axis=0)
    periods = numpy.concatenate(periods_parts[::-1], axis=0)

    im = CLImg(ctx, (1, 1))
    im.shape = SIZE
    im.img = (image, image)
    im.save(f"{filename}.png")
    numpy.save(f"{filename}.npy", periods)

    del image
    del image_parts
    del periods
    del periods_parts


for statement in tqdm(script, ncols=120):
    compute_high_res_param_map(*statement)
