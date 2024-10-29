import numpy as np
from scipy import interpolate

def getForwardModelBSpline(x, kOrder, knots):
    ## Number of coefficients based on the knot points and the order the B-spline
    nParsT = knots.size + kOrder  ## Number of parameters INCLUDES constant parameter (zero point)

    h = knots[1] - knots[0]

    nSplines = knots.size + kOrder - 1
    tKnotSpl = np.linspace(knots[0] - kOrder * h, knots[-1] + kOrder * h, knots.size + 2 * kOrder, endpoint=True)

    ## Forward model
    X = np.zeros((x.size, nParsT))

    X[:, 0] = 1.0

    for i in range(1, nParsT):
        start = i - 1
        end = start + kOrder + 2

        bspline = interpolate.BSpline.basis_element(tKnotSpl[start:end], extrapolate=False)

        X[:, i] = bspline(x)

    X[np.isnan(X)] = 0.0

    return X