import numpy
from matplotlib import pyplot


def draw_points_on_map(map_file, points_file, bounds, output):
    import json
    with open(points_file) as f:
        points = json.load(f)

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 192000000

    image = Image.open(map_file)
    dpi = 64
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 140})

    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
    ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

    for i, point in enumerate(points):
        ax.scatter(*point,
                   marker="v",
                   s=1000,
                   color=(1.0, 1.0, 1.0),
                   label=f"{i}")
        ax.annotate(str(i), point)

    fig.tight_layout()
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    fig.tight_layout()
    fig.savefig(output)
    pyplot.close(fig)


draw_points_on_map(
    "../../COURSEWORK_DATA/OLD_MAPS/final_001_inc_20200518-162846.png",
    "../../COURSEWORK_DATA/ATTRACTORS/points.json",
    (-6, 0, 0.5, 1.0),
    "../../COURSEWORK_DATA/map_001.png"
)


draw_points_on_map(
    "../../COURSEWORK_DATA/OLD_MAPS/final_0001_inc_20200518-163151.png",
    "../../COURSEWORK_DATA/ATTRACTORS/points.json",
    (-6, 0, 0.5, 1.0),
    "../../COURSEWORK_DATA/map_0001.png"
)

draw_points_on_map(
    "../../COURSEWORK_DATA/OLD_MAPS/final_00001_inc_20200518-163457.png",
    "../../COURSEWORK_DATA/ATTRACTORS/points.json",
    (-6, 0, 0.5, 1.0),
    "../../COURSEWORK_DATA/map_00001.png"
)
