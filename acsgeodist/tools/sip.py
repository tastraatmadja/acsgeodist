'''
SIP (Simple Imaging Polynomial) related tools
'''

from acsgeodist import acsconstants
import numpy as np

def getUpperTriangularMatrixNumberOfElements(n):
    return int(n * (n+1) / 2)

'''
Return the upper triangular index of a square matrix with any size 
Example below for n = 5, the p index for any given i,j index will
be as the following:
 * i\j|  0 |  1 |  2 |  3 |  4 |
 * ---+----+----+----+----+----|
 *  0 |  0 |  2 |  5 |  9 | 14 |
 * ---+----+----+----+----+-----
 *  1 |  1 |  4 |  8 | 13 |
 * ---+----+----+----+-----
 *  2 |  3 |  7 | 12 |
 * ---+----+----+-----
 *  3 |  6 | 11 |
 * ---+----+-----
 *  4 | 10 |
 * ---------
'''
def getUpperTriangularIndex(i, j):
    return int(getUpperTriangularMatrixNumberOfElements(i+j) + j)

'''
Given an upper triangular index p, return the Cantor pair of integers
i and j
Source: https://en.wikipedia.org/wiki/Pairing_function
'''
def getCantorPair(p):
    w = np.floor((np.sqrt(8 * p + 1) - 1) / 2)
    t = (w * w + w) / 2
    j = int(p - t)
    return int(w - j), j

def buildModel(X, Y, pOrder, scalerX=1.0, scalerY=1.0, bothAxes=False):
    n = pOrder + 1

    vanderX = np.vander(X / scalerX, n, increasing=True)
    vanderY = np.vander(Y / scalerY, n, increasing=True)

    vanderScalerX = np.vander(np.array([scalerX]), n, increasing=True)
    vanderScalerY = np.vander(np.array([scalerY]), n, increasing=True)

    P = getUpperTriangularMatrixNumberOfElements(n)  ## Number of parameters PER AXIS!

    naxis = 1
    if bothAxes:
        naxis = 2

    XModel = np.zeros((naxis * X.shape[0], naxis * P))
    scaler = np.zeros((1, naxis * P))

    for ii in range(n):
        for jj in range(n - ii):
            pVal = vanderX[:, ii] * vanderY[:, jj]
            sVal = vanderScalerX[:, ii] * vanderScalerY[:, jj]
            ppp  = getUpperTriangularIndex(ii, jj)

            for axis in range(naxis):
                XModel[axis::naxis, ppp * naxis + axis] = pVal
                scaler[:, ppp * naxis + axis]           = sVal

    return XModel, scaler.flatten()

'''
Return the coefficient 'name' in Vera's formatting, i.e. i = 2 and j = 1
returns XXY, i = 3 and j = 2 returns XXXYY
'''
def getCoeffName(i, j):
    if (i == 0) and (j == 0):
        return 'CONSTANT'
    else:
        name = ''
        for pp in range(i):
            name += 'X'
        for pp in range(j):
            name += 'Y'
        return name


def getCoefficients(df, pOrder, iloc=0):
    n = pOrder+ 1
    P = getUpperTriangularMatrixNumberOfElements(n)

    results = np.zeros((acsconstants.NAXIS, P))

    for ii in range(n):
        for jj in range(n - ii):
            ppp = getUpperTriangularIndex(ii, jj)

            ## sVal = vanderScalerX[:, ii] * vanderScalerY[:, jj]

            for axis in range(acsconstants.NAXIS):
                coeffName = '{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], ppp + 1)
                results[axis, ppp] = df.iloc[iloc][coeffName]

    return results

def getBothAxesIndices(indices_in, naxis=2):
    indices_out = np.zeros(indices_in.size * naxis, dtype=int)

    for axis in range(naxis):
        indices_out[axis::naxis] = naxis * indices_in + axis

    return indices_out

def getCXCYCoeffsFromABCoeffs(A, B):
    ## Extract the linear matrix that perform the affine transform from XY detector frame into V2-V3
    ## focal plane frame.
    lin_mat = np.array([[A[1], A[2]], [B[1], B[2]]])

    ## Determinant of the linear matrix
    det_lin_mat = lin_mat[0, 0] * lin_mat[1, 1] - lin_mat[1, 0] * lin_mat[0, 1]

    ## Inverse of the linear matrix
    inv_lin_mat = np.array(
        [[lin_mat[1, 1], -lin_mat[0, 1]], [-lin_mat[1, 0], lin_mat[0, 0]]]) / det_lin_mat

    CX = np.zeros((A.size, 1))
    CY = np.zeros_like(CX)

    '''
    for p in range(1, 3):
        CX[p] = inv_lin_mat[0, p - 1]
        CY[p] = inv_lin_mat[1, p - 1]
    ''';

    for p in range(1, CX.shape[0]):
        CX[p], CY[p] = (inv_lin_mat @ np.array([[A[p]], [B[p]]])).flatten()

    return CX, CY, inv_lin_mat