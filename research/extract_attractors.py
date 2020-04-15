import os

import numpy


if True:
    res = numpy.load("temp_res.npy")
    counts = numpy.load("temp_counts.npy")
else:
    res = numpy.load("test-map.npy")\
        .reshape((1920 // 2 * 1080 * 64, 2))\
        .round(4)

    res = res[(numpy.abs(res[:, 0]) < 20) & (numpy.abs(res[:, 1]) < 20)]
    res, counts = numpy.unique(res, axis=0, return_counts=True)
    numpy.save("temp_res.npy", res)
    numpy.save("temp_counts.npy", counts)
    print("saved")


res = res[counts > 100]
counts = counts[counts > 100]

order = numpy.argsort(counts)[::-1]

print(counts[order])
print(res[order])


x, y = res.T

print(len(x))


from matplotlib import pyplot

# heatmap, xedges, yedges = numpy.histogram2d(x, y, bins=10)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
# pyplot.imshow(heatmap.T, extent=extent, origin='lower')

pyplot.scatter(x, y, marker='.')
pyplot.grid()
pyplot.show()




