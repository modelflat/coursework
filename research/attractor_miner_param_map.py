import numpy

from config import C
from core.param_map import ParameterMap
from core.utils import create_context_and_queue, CLImg


def main(ctx, queue):
    param = ParameterMap(ctx)
    image = CLImg(ctx, (1920 // 2, 1080))

    def compute_map(name):
        points, periods = param.compute_incremental(
            queue, image,
            skip=1 << 16,
            skip_batch_size=1 << 6,
            iter=1 << 6,
            z0=(1e-5, -1e-5),
            c=C,
            bounds=(-6, -3, 0.5, 1.0),
            root_seq=(0, 0, 1),
            seed=42,
            draw_image=True,
        )
        image.save(f"{name}.png")
        numpy.save(f"{name}.npy", points)

    compute_map("test-map")


if __name__ == '__main__':
    main(*create_context_and_queue())
