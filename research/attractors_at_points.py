import json
import os

from collections import Counter, defaultdict

import numpy
from matplotlib import pyplot
from tqdm import tqdm

from app.core.basins import BasinsOfAttraction, analyze_attractors
from app.core.utils import create_context_and_queue, CLImg

from app.config import Config

ctx, queue = create_context_and_queue()
basins = BasinsOfAttraction(ctx)

SIZE = (640, 480)
BOUNDS = (-2, 2, -2, 2)
SKIP = 1 << 16
ITER = 1 << 6

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../COURSEWORK_DATA/ATTRACTORS/0002"))

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(OUTPUT_PATH)


points_file = "points.json"

NORM = 100


def to_file(image, bounds, filename, colors_and_labels, dpi=64, legend_bbox_anchor=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image, origin="upper", extent=bounds, aspect="auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(numpy.linspace(*bounds[0:2], 10))
    ax.set_yticks(numpy.linspace(*bounds[2:4], 10))

    for (col, label) in colors_and_labels:
        ax.scatter(
            bounds[0] - 1, bounds[2] - 1, marker="o", 
            color=tuple(numpy.clip(col, 0.0, 1.0)), label=label
        )
    if legend_bbox_anchor is not None:
        ax.legend(bbox_to_anchor=legend_bbox_anchor)
        fig.tight_layout()
    else:
        ax.legend(loc="upper right")

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def merge_attractor_indexes(index1, index2, filter_by_norm):
    for period, period_data in index2.items():
        if period not in index1:
            index1[period] = dict()
        for attractor, attractor_data in period_data.items():
            attr = attractor_data.pop("attractor")
            if not filter_by_norm or numpy.linalg.norm(attr.flatten()) < NORM:
                if attractor not in index1[period]:
                    index1[period][attractor] = dict()
                index1[period][attractor]["count"] = index1[period][attractor].get("count", 0) + attractor_data["count"] 
    return index1


def find_in_point(filename, root_seq, h, alpha, zoom=None):
    if os.path.exists(filename):
        pass

    do_update_attractors = True

    image = CLImg(ctx, SIZE)

    image.clear(queue)

    if os.path.exists("all_attractors.json"):
        with open("all_attractors.json") as f:
            all_attractors = json.load(f)
    else:
        all_attractors = dict()

    known_colors = dict()
    for period_data in all_attractors.values():
        for attractor, attractor_data in period_data.items():
            if "color" in attractor_data:
                known_colors[attractor] = attractor_data["color"]
    
    assigned_colors = defaultdict(lambda: [])
    assigned_colors_tracker = set()

    def color_fn(attractor_string, attractor):
        period = attractor["period"]
        attr = attractor["attractor"]

        norm = numpy.linalg.norm(attr.flatten())

        if period == 1:
            col = (0, 1.0, 1.0)
        elif period == 3:
            p1, p2, p3 = attr.view(dtype=numpy.complex128).flatten()
            if abs(p1) < 1.0 and abs(p3) < 1.0 and abs(p2) > 20.0:
                n = 0.6 * min(norm / 80, 1.0)
                col = (240, 1.0, 1.0 - n)
            else:
                col = (280, 1.0, 1.0)
        elif norm > NORM:
            col = (300 * period / 64, 1.0, 0.3)
        else:
            col = (240 * period / 64, 1.0, 1.0)

        i = 0
        while col in assigned_colors_tracker:
            col = (col[0], col[1] - 0.25, col[2])
            i += 1
            if i >= 3:
                break
        
        assigned_colors_tracker.add(col)

        assigned_colors[period].append((col, attractor))

        return col

    attractors, attractors_raw, _ = analyze_attractors(
        basins,
        queue, image, SKIP, ITER, h, alpha, Config.C, BOUNDS,
        root_seq=root_seq,
        tolerance_decimals=2,
        seed=42,
        threshold=100,
        color_fn=color_fn
    )

    merge_attractor_indexes(all_attractors, attractors, filter_by_norm=True)

    if do_update_attractors:
        with open("all_attractors.json", "w") as f:
            json.dump(all_attractors, f, indent=2, sort_keys=True)

    with open(f"{filename.rsplit('.', 1)[0]}.json", "w") as f:
        json.dump(attractors, f, indent=2, sort_keys=True)

    if attractors_raw:
        fig, ax = pyplot.subplots(1, 1, dpi=200, figsize=(20, 15))

        total = numpy.prod(SIZE)

        if len(attractors_raw) == 1:
            alpha = 1.0
        else:
            alpha = 0.7

        attractors_found = False
        attractors = dict()

        total = sum((attr["occurences"] for attr in attractors_raw))

        period_counts = Counter()

        for attractor in sorted(attractors_raw, key=lambda x: (x["period"], -x["occurences"])):
            period = attractor["period"]
            area = 100 * attractor["occurences"] / total
            attr = attractor["attractor"]
            period_counts[period] += 1
            if period_counts[period] < 4:
                x, y = attr.T
                # order = numpy.argsort(x)
                # x = x[order]
                # y = y[order]
                ax.plot(
                    x, y, "-o",
                    label=f"Attractor #{period_counts[period]} ({period}) ({area:.2f}%)", 
                    alpha=alpha,
                )
        
        ax.grid(which="both")
        fig.tight_layout()
        fig.legend()
        fig.savefig(filename)
        pyplot.close(fig)
        
        if zoom is not None:
            period_counts = Counter()
            fig, ax = pyplot.subplots(1, 1, dpi=200, figsize=(20, 15))
            
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
    
            for attractor in sorted(attractors_raw, key=lambda x: (x["period"], -x["occurences"])):
                period = attractor["period"]
                area = 100 * attractor["occurences"] / total
                attr = attractor["attractor"]
                period_counts[period] += 1
                if period_counts[period] < 4:
                    x, y = attr.T
                    # order = numpy.argsort(x)
                    # x = x[order]
                    # y = y[order]
                    ax.plot(
                        x, y, "-o",
                        label=f"Attractor #{period_counts[period]} ({period}) ({area:.2f}%)", 
                        alpha=alpha,
                    )

            ax.grid(which="both")
            fig.tight_layout()
            fig.legend()
            fn = filename.index("_")
            fn = filename.index("_", fn + 1)
            fig.savefig(filename[:fn] + "_zoomed" + filename[fn:])
            pyplot.close(fig)

    from matplotlib.colors import hsv_to_rgb
    colors_and_labels = []
    for period, data in assigned_colors.items():
        for i, (col, attractor) in enumerate(sorted(data, key=lambda x: x[1]["occurences"], reverse=True)[:3]):
            area = 100 * attractor["occurences"] / total

            col = (col[0] / 360, col[1], col[2])
            col = hsv_to_rgb(numpy.array(col))
            
            colors_and_labels.append(
                (col, f"#{i + 1} ({period}) ({area:.2f}%)")
            )

    fn = filename.index("_")
    fn = filename.index("_", fn + 1)
    to_file(
        image.as_img(), BOUNDS, filename[:fn] + "_basins" + filename[fn:], colors_and_labels,
        dpi=64, legend_bbox_anchor=(1.3, 1.012)
    )


def main():
    root_seq = (0, 0, 0, 2)

    with open(points_file) as f:
        points = json.load(f)

    zoomed_in_fragments = {
        2: [-3, 3, -3, 3],
        13: [-1, 1, -1, 1],
        14: [-1, 1, -1, 1],
        15: [-1, 1, -1, 1],
        16: [-1, 1, -1, 1],
        17: [-1, 1, -1, 1],
    }

    for i, (h, alpha) in enumerate(tqdm(points)):
        find_in_point(f"attractors_{i}_{h:.3}_{alpha:.3}.png", root_seq, h, alpha, zoom=zoomed_in_fragments.get(i))



if __name__ == '__main__':
    main()
