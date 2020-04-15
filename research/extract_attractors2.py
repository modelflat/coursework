import numpy
from tqdm import tqdm

def step1():

    raw = numpy.load("test-map.npy") \
        .reshape((1920 // 2 * 1080, 64, 2)) \
        .round(3)


    decent_periodic_sequences = []


    for line in tqdm(raw, ncols=120):
        if numpy.any(numpy.isnan(line)):
            continue
        un, cn = numpy.unique(line, return_counts=True, axis=0)
        if len(cn) > len(line) // 4:
            continue
        if numpy.any(numpy.abs(line.flat) > 1e2):
            continue

        # print(un, cn, f"period {len(cn)}")
        decent_periodic_sequences.append(line)

    d = numpy.array(decent_periodic_sequences, dtype=float)
    numpy.save("periodic_seqs.npy", d)


def step2():
    periodic = numpy.load("periodic_seqs.npy")

    print(periodic)
    print(len(periodic))

    un, cn = numpy.unique(periodic, return_counts=True, axis=0)

    print(un)
    print(cn)

    print(len(un))



step2()
