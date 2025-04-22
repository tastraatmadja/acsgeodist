import os

from acsgeodist.tools import sip
from acsgeodist import acsconstants
import pandas as pd
def readFitResults(fitResultsFilename, pOrder):
    columns = ['rootname', 'chip', 'time', 'pa_v3', 'orientat', 'vaFactor', 'tExp', 'posTarg1', 'posTarg2',
               'nIterTotal', 'nStars', 'rmsXi', 'rmsEta', 'crval1', 'crval2', 'cd11', 'cd12', 'cd21', 'cd22']

    n = pOrder + 1

    nParsAxis = sip.getUpperTriangularMatrixNumberOfElements(n)

    for ppp in range(nParsAxis):
        for axis in range(acsconstants.NAXIS):
            coeffName = '{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], ppp + 1)

            columns.append(coeffName)

    if hasattr(fitResultsFilename, '__iter__'):
        df_fitResults = []
        for filename in fitResultsFilename:
            df_fitResults.append(pd.read_csv(filename, sep='\s+', header=None, comment='#', names=columns))
        return pd.concat(df_fitResults)
    else:
        return pd.read_csv(fitResultsFilename, sep='\s+', header=None, comment='#', names=columns)

def readHST1PassFile(hst1passFilename):
    split = os.path.splitext(hst1passFilename)

    columns = list(split[-1].replace(".", ""))

    return pd.read_csv(hst1passFilename, sep='\\s+', header=None, comment='#', names=columns)
