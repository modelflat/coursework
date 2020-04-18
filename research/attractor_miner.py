import json

from datetime import datetime

import numpy
from tqdm import tqdm

from config import C, phase_shape
from core.basins import BasinsOfAttraction
from core.utils import CLImg
from core.utils import create_context_and_queue



def main(ctx, queue):
    h_bounds = (-6, -3)
    h_gran = 1
    a_bounds = (0.5, 1.0)
    a_gran = 1

    skip = 1 << 24
    skip_batch = 1 << 14
    iter = 1 << 8
    root_seq = (0, 0, 1)

    basins = BasinsOfAttraction(ctx)
    img = CLImg(ctx, (256, 256))

    all_attractors_with_support = dict()

    def compute_attractors_at_params(h, a):

        points, periods = basins.deep_capture(
            queue, img.shape, skip, skip_batch, iter, h, a, C,
            phase_shape, root_seq, seed=42
        )

        numpy.save(f"points_2_24_400_{h},{a}.npy", points)
        numpy.save(f"periods_2_24_400_{h},{a}.npy", periods)

    compute_attractors_at_params(-3.9141, 0.7598)
    compute_attractors_at_params(-3.4570, 0.6265)
    compute_attractors_at_params(-5.5430, 0.6807)

    #
    #     print(points, periods)
    #
    #     attractors = basins.compute_periods_and_attractors(
    #         queue, img,
    #         skip=skip,
    #         iter=iter,
    #         h=h, alpha=a, c=C, bounds=phase_shape,
    #         root_seq=root_seq,
    #         seed=42,
    #         compute_image=False,
    #         only_good_attractors=True
    #     )
    #     return [(k.real, k.imag, sum(e[1] for e in v)) for k, v in attractors.items()]
    #
    # for h in tqdm(numpy.linspace(*h_bounds, num=h_gran), total=h_gran, ncols=120):
    #     all_attractors_with_support[h] = dict()
    #     for a in tqdm(numpy.linspace(*a_bounds, num=a_gran), total=a_gran, ncols=120):
    #         attrs = compute_attractors_at_params(h, a)
    #         all_attractors_with_support[h][a] = attrs
    # with open("attractor_mining.json.checkpoint", "w") as f:
    #     json.dump(all_attractors_with_support, f, indent=2, sort_keys=True)
    #
    # with open(f"attractor_mining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    #     json.dump(all_attractors_with_support, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    ctx, queue = create_context_and_queue()
    main(ctx, queue)
