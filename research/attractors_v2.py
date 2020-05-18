from matplotlib import pyplot

from core.basins import BasinsOfAttraction
from core.utils import create_context_and_queue

from config import C

ctx, queue = create_context_and_queue()
basins = BasinsOfAttraction(ctx)

bounds = (-2, 2, -2, 2)
shape = (1024, 1024)

skip = 1 << 12
iter = 1 << 6

points_file = "attr_res/points.json"


def find_in_point(root_seq, h, alpha):
    result = basins.find_attractors(
        queue, shape, skip, iter, h, alpha, C, bounds, root_seq, tolerance_decimals=3, seed=42,
    )
    print([(y, len(x[0])) for y, x in result.items()])
    thr = 128
    for period, (data, counts) in result.items():
        data = data[counts > thr]
        if data.shape[0] == 0:
            continue
        counts = counts[counts > thr]

        fig, ax = pyplot.subplots(1, 1)

        print(f"{period} : {len(data)}")

        for i, attr in enumerate(data):
            print(list(map(lambda x: tuple(x), attr)), counts[i])
            ax.scatter(*attr.T, label=f"{i}")

        fig.tight_layout()
        fig.legend()
        fig.savefig(f"D:\\test-{period}.png")


def main():
    root_seq = (0, 0, 1)
    import json
    with open(points_file) as f:
        points = json.load(f)

    for h, alpha in points[:5]:
        find_in_point(root_seq, h, alpha)


if __name__ == '__main__':
    main()
