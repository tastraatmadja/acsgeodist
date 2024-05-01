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
 * i\j|  0    1    2    3    4 |
 * ---|-------------------------
 *  0 |  0 |  2 |  5 |  9 | 14 |
 *    |-------------------------
 *  1 |  1 |  4 |  8 | 13 |
 *    |--------------------
 *  2 |  3 |  7 | 12 |
 *    |---------------
 *  3 |  6 | 11 |
 *    |----------
 *  4 | 10 |
 * ---------
'''
def getUpperTriangularIndex(i, j):
    return int(getUpperTriangularMatrixNumberOfElements((i+j)) + i)

def buildModel(X, Y, pOrder, scalerX=1.0, scalerY=1.0):
    n = pOrder + 1

    vanderX = np.vander(X / scalerX, n, increasing=True)
    vanderY = np.vander(Y / scalerY, n, increasing=True)

    vanderScalerX = np.vander(np.array([scalerX]), n, increasing=True)
    vanderScalerY = np.vander(np.array([scalerY]), n, increasing=True)

    P = getUpperTriangularMatrixNumberOfElements(n)  ## Number of parameters PER AXIS!

    XModel = np.zeros((X.shape[0], P))
    scaler = np.zeros((1, P))

    for ii in range(n):
        for jj in range(n - ii):
            pVal = vanderX[:, ii] * vanderY[:, jj]
            sVal = vanderScalerX[:, ii] * vanderScalerY[:, jj]
            ppp = getUpperTriangularIndex(jj, ii)

            XModel[:, ppp] = pVal
            scaler[:, ppp] = sVal

    return XModel, scaler.flatten()

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


def getCoefficients(df, pOrder):
    n = pOrder+ 1
    P = getUpperTriangularMatrixNumberOfElements(n)

    results = np.zeros((acsconstants.NAXIS, P))

    for ii in range(n):
        for jj in range(n - ii):
            ppp = getUpperTriangularIndex(jj, ii)

            ## sVal = vanderScalerX[:, ii] * vanderScalerY[:, jj]

            for axis in range(acsconstants.NAXIS):
                coeffName = '{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], ppp + 1)
                results[axis, ppp] = df.iloc[0][coeffName]

    return results