import numpy as np
from scipy.spatial import distance
from scipy.stats import norm

LN_2PI = np.log(2 * np.pi)

SCALER_RSE  = 1.0 / (norm.ppf(0.90) - norm.ppf(0.10))
SCALER_RSTD = 1.0 / (norm.ppf(0.95) - norm.ppf(0.05))

'''
Weighting function wdecay
Returns the downweighting factor for normalized deviation z,
using a modified exponential decay function.
It effectively gives no weight to data points more than 10-20 sigmas
from the fitted value (whereas whuber gives significant weight at
these distances).
* @author L.Lindegren, Lund Observatory (2004 May 19)
* @param z double : normalized residual, expected to be approximately N(0,1)
* @return double : factor by which the statistical weight should be reduced
'''
def wdecay(z):
    if isinstance(z, (list, np.ndarray)):
        weights = np.zeros(z.size)
        for i in range(weights.size):
            weights[i] = wdecay(z[i])
        return weights.flatten()
    else:
        zabs = np.absolute(z)

        if (zabs <= 2.0):
            return 1.0
        elif (zabs <= 3.0):
            t = zabs - 2.0
            return 1.0 - (1.77373519609519 - 1.14161463726663*t)*t*t;
        elif (zabs <= 10.0):
            return np.exp(-zabs/3.0)
        else:
            return 0.0


def estimateMeanAndCovarianceMatrix(x):
    ## Use median instead of mean because it is more robust to outlier
    median = np.nanpercentile(x, 50, axis=0)

    ## Robustly estimate the standard deviations using rescaled 90% credible interval
    ## of the sample
    stdDev = 0.5 * (np.nanpercentile(x, 95.0, axis=0) - np.nanpercentile(x, 5.0, axis=0)) / 1.645

    ## Use numpy.cov to estimate the covariance matrix
    cov = np.cov(x.T)

    ## Only take the correlation coefficients
    ## corr = cov[0,1] / np.sqrt(cov[0,0]) / np.sqrt(cov[1,1])

    covOut = np.zeros((median.size, median.size))

    np.fill_diagonal(covOut, stdDev ** 2)

    for i in range(median.size):
        for j in range(i + 1, median.size):
            corr = cov[i, j] / np.sqrt(cov[i, i]) / np.sqrt(cov[j, j])
            covOut[i, j] = corr * stdDev[i] * stdDev[j]
            covOut[j, i] = covOut[i, j]
    return median, covOut


def estimateMeanAndCovarianceMatrixRobust(x, w):
    ## Normalize the weights such that they sum to unity
    w = w / np.sum(w)

    mean = np.average(x, weights=w, axis=0)
    cov  = np.cov(x.T, ddof=0, aweights=w) / (1.0 - np.sum(w ** 2))
    ## cov  = np.cov(x.T, ddof=0, aweights=w)
    ## var  = np.average((x - mean)**2, weights=w, axis=0) * sumW / (sumW - 1)

    ## print(mean)
    ## print(var, np.sqrt(var))
    ## print(cov)
    return mean, cov


def getMahalanobisDistances(x, mean, invCov):
    distances = np.zeros((x.shape[0],))

    for i in range(x.shape[0]):
        distances[i] = distance.mahalanobis(x[i], mean, invCov)

    return distances

def getBIC(k, n, chiSq):
    return chiSq + k * np.log(n)

def getChiSq(res, weights, stdDev):
    return np.nansum(weights * res**2 / stdDev**2) / np.nansum(weights)

def getLnLL(res, weights):
    selection = weights > 0
    return 0.5 * (-LN_2PI * float(weights[selection].size) +
            np.nansum(np.log(weights[selection])) - np.nansum(weights * res**2))

def getLnPMD(k, res, lnPM, weights=None):
    if (weights is None):
        chiSq = np.nansum(res**2)
        nEff  = float(res.size)
    else:
        nEff  = float(weights[weights > 0].size)
        chiSq = np.nansum(weights * res**2)
    return -0.5*getBIC(k, nEff, chiSq) + lnPM

def addLnProb(lnP1, lnP2):
    if ((lnP1 > -np.inf) and (lnP2 > -np.inf)):
        if (lnP1 > lnP2):
            return lnP1 + np.log1p(np.exp(lnP2 - lnP1))
        else:
            return lnP2 + np.log1p(np.exp(lnP1 - lnP2))
    else:
        if (lnP1 > lnP2):
            return lnP1
        else:
            return lnP2

def getWeightedAverage(x, weights):
    selection = np.isfinite(x) & np.isfinite(weights)
    return np.average(x[selection], weights=weights[selection])

'''
RSE stands for Robust Scatter Estimate (Lindegren et al. 2012). RSE is the interdecile range (9th decile minus the first decile)
multiplied by 0.390152 to scale it to the standard deviation.
'''
def getRSE(x):
    return SCALER_RSE * (np.nanpercentile(x, 90.0) - np.nanpercentile(x, 10.0))

'''
Robust Std. Deviation is from Astraatmadja & Bailer Jones 2016. It is the 90% credible interval (i.e. the range between 
the 95th percentile and the 5th percentile) divided by 2s, where s = 1.645, to scale it to the standard deviation (s is
the ratio of the 90% to 68.3% credible in a standard gaussian).
'''
def getRobustStdDev(x):
    return SCALER_RSTD * (np.nanpercentile(x, 95.0) - np.nanpercentile(x, 5.0))

