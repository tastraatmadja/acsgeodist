import gc

import numpy as np
from scipy import signal

def gaussianKernel2d(x, xi, y, yi, hx, hy):
    dx = (x - xi) / hx
    dy = (y - yi) / hy
    u2 = dx ** 2 + dy ** 2

    del dx
    del dy
    gc.collect()

    return np.exp(-0.5 * u2)

def getKDEImage(x, y, xBins, yBins, hx, hy):
    idx0 = np.digitize(x, xBins)
    idx1 = np.digitize(y, yBins)

    xMids = 0.5 * (xBins[idx0 - 1] + xBins[idx0])
    yMids = 0.5 * (yBins[idx1 - 1] + yBins[idx1])

    weights = gaussianKernel2d(x, xMids, y, yMids, hx, hy)

    nx, ny = xBins.size - 1, yBins.size - 1

    kdeImage = np.zeros((ny, nx))

    for i, j, value in zip(idx1 - 1, idx0 - 1, weights):
        ## print(i, j, value)
        kdeImage[i, j] += value

    normalizedKDEImage = kdeImage / kdeImage.sum()

    del idx0
    del idx1
    del xMids
    del yMids
    del weights
    del kdeImage
    gc.set_threshold(2, 1, 1)
    gc.collect()

    return normalizedKDEImage


def phaseCorrelate2d(x1, y1, x2, y2, dx, dy, hx, hy):
    ## Calculate coverage of the
    xMin, xMax = np.floor(np.nanmin(np.hstack([x1, x2]))), np.ceil(np.nanmax(np.hstack([x1, x2])))
    yMin, yMax = np.floor(np.nanmin(np.hstack([y1, y2]))), np.ceil(np.nanmax(np.hstack([y1, y2])))

    nPointsX = int((xMax - xMin) // dx) + 1
    nPointsY = int((yMax - yMin) // dy) + 1

    nBinsX = nPointsX + 1
    nBinsY = nPointsY + 1

    ## print(xMin, xMax, yMin, yMax, nPointsX, nPointsY, nBinsX, nBinsY)

    xBins = np.linspace(xMin - 0.5 * dx, xMax + 0.5 * dx, nBinsX, endpoint=True)
    yBins = np.linspace(yMin - 0.5 * dy, yMax + 0.5 * dy, nBinsY, endpoint=True)

    kdeIm1 = getKDEImage(x1, y1, xBins, yBins, hx, hy)
    kdeIm2 = getKDEImage(x2, y2, xBins, yBins, hx, hy)

    corrIm = signal.correlate(kdeIm1, kdeIm2, mode='full', method='auto')

    del kdeIm1
    del kdeIm2
    gc.set_threshold(2, 1, 1)
    gc.collect()

    idx = np.unravel_index(np.argmax(corrIm), corrIm.shape)

    shiftX = (dx * (idx[1] - int(corrIm.shape[1] // 2)))
    shiftY = (dy * (idx[0] - int(corrIm.shape[0] // 2)))

    return shiftX, shiftY, idx, corrIm

