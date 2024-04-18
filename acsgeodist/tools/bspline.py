import numpy as np
from scipy import signal

def getForwardModelBSpline(x, kOrder, knots):
    ## Number of coefficients based on the knot points and the order the B-spline
    nParsT = knots.size + kOrder  ## Number of parameters include constant parameter (zero point)

    h = knots[1] - knots[0]

    xMin = knots[0]

    ## Forward model
    X = np.zeros((x.size, nParsT))

    X[:, 0] = 1.0
    for i in range(1, nParsT):
        knotID = i - kOrder - 1
        X[:, i] = signal.bspline((x - xMin) / h - 0.5 * (kOrder + 1) - knotID, kOrder)

    return X