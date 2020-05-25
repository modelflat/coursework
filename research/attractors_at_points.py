import json
import os

import numpy
from matplotlib import pyplot

from app.core.basins import BasinsOfAttraction
from app.core.utils import create_context_and_queue, CLImg

from app.config import Config

ctx, queue = create_context_and_queue()
basins = BasinsOfAttraction(ctx)

# SIZE = (1024, 1024)
SIZE = (128, 128)
BOUNDS = (-2, 2, -2, 2)
SKIP = 1 << 12
ITER = 1 << 6

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../COURSEWORK_DATA/ATTRACTORS"))

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(OUTPUT_PATH)


points_file = "points.json"


def find_in_point(filename, root_seq, h, alpha):
    # if os.path.exists(filename):
    #     return
    image = CLImg(ctx, SIZE)

    image.clear(queue)

    attractors = None
    def color_init(attrs, n_collisions):
        import json
        print(json.dumps(attrs, indent=2))
        # TODO improve API to avoid doing things like this
        nonlocal attractors
        attractors = attrs

    _ = basins.compute(
        queue, image, SKIP, ITER, h, alpha, Config.C, BOUNDS,
        root_seq=root_seq,
        tolerance_decimals=3,
        seed=42,
        color_init=color_init,
        color_fn=lambda a: (240, 1, 1) if a["period"] == 3 else (120, 1, 1),
    )

    image.save("test2.png")

    return

    fig, ax = pyplot.subplots(1, 1, dpi=200, figsize=(24, 18))

    result = attractors
    
    print([(y, len(x[0])) for y, x in result.items()])
    
    thr = 128
    total = numpy.prod(SIZE)

    if len(result) == 1:
        alpha = 1.0
    else:
        alpha = 0.7

    attractors_found = False
    attractors = dict()

    for period, (data, counts) in result.items():
        data = data[counts > thr]
        if data.shape[0] == 0:
            continue

        attractors_found = True
        counts = counts[counts > thr]

        order = numpy.argsort(counts)[::-1][:5]
        data = data[order]
        counts = counts[order]

        found = attractors.get(str(period), dict())
        for att, c in zip(data, counts):
            att = json.dumps(tuple(map(lambda x: tuple(x), att)))
            found[att] = found.get(att, 0) + int(c)
        attractors[str(period)] = found

        for i, attr in enumerate(data):
            area = 100 * counts[i] / total
            ax.scatter(*attr.T, label=f"Attractor #{i + 1} of period {period}; {area:.2f}% of total", alpha=alpha)

    if attractors_found:
        ax.grid(which="both")
        fig.tight_layout()
        fig.legend()
        fig.savefig(filename)
        pyplot.close(fig)

        if os.path.exists("attractors.json"):
            with open("attractors.json") as f:
                saved_attractors = json.load(f)
        else:
            saved_attractors = dict()
        
        for period, attrs in attractors.items():
            for attr, c in attrs.items():
                attrs[attr] = attrs.get(attr, 0) + c
            saved_attractors[period] = attrs

        with open("attractors.json", "w") as f:
            json.dump(saved_attractors, f, sort_keys=True, indent=2)


def main():
    root_seq = (0, 0, 1)

    # with open(points_file) as f:
    #     points = json.load(f)

    # for i, (h, alpha) in enumerate(points):
    #     find_in_point(f"attractors_{i}_{h:.3}_{alpha:.3}.png", root_seq, h, alpha)

    # h = -3.4609375
    # alpha = 0.79296875
    # h = -5.91015625
    # alpha = 0.7109375
    h = -5.140625
    alpha = 0.62890625
    find_in_point(f"attractors_{0}_{h:.3}_{alpha:.3}.png", root_seq, h, alpha)


if __name__ == '__main__':
    main()
