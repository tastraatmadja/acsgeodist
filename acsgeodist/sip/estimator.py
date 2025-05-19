from acsgeodist import acsconstants
from acsgeodist.tools import astro, bspline, coords, litho, plotting, sip, stat
from acsgeodist.tools.time import convertTime
from astropy import table
from astropy import units as u
from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.io import ascii, fits
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astroquery.gaia import Gaia
from copy import deepcopy
import gc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from photutils.aperture import CircularAperture
from scipy import interpolate, linalg, sparse
import seaborn as sns
from sklearn import linear_model
import time

NAXIS = acsconstants.NAXIS

## Cross-matching parameters
Q_MIN   = 1.e-6
Q_MAX   = 1.0
MAX_SEP = 3.0 * u.pix
HEIGHT  = 0.5 * u.deg
WIDTH   = HEIGHT

## Number of linear parameters
N_PARS_LINE  = 2
N_PARS_CONST = 1

## For cross-matching, select only Gaia sources with good astrometry_old measurements
MIN_RUWE = 0.8
MAX_RUWE = 1.2

chips   = acsconstants.CHIP_NUMBER
headers = chips[::-1] # hdu[SCI, x]
X0      = 2048.00
Y0      = np.array([1024.0, 2048.0+1024.0])

XRef = X0
YRef = Y0[0]

scalerX = 2048.0
scalerY = 1024.0

N_ITER_OUTER = 10
N_ITER_INNER = 100
N_ITER_CD    = 5

retainedColor  = 'k'
nonFullColor   = '#fc8d59' ## Orange
discardedColor = 'r'

cMap = 'Greys'

markerSize = 3

dX, dMX = 1000, 200
dY, dMY = 500, 100

xLabel, yLabel = r'$X$ [pix]', r'$Y$ [pix]'

class SIPEstimator:
    def __init__(self, referenceCatalog, referenceWCS, tRef0, qMax=0.5, min_n_app=3, max_pix_tol=1.0,
                 min_n_refstar=100, individualZP=True, make_lithographic_and_filter_mask_corrections=True,
                 cross_match=True):
        self.individualZP  = individualZP
        self.refCat        = deepcopy(referenceCatalog)
        self.wcsRef        = deepcopy(referenceWCS)
        self.tRef0         = deepcopy(tRef0)
        self.qMax          = qMax
        self.min_n_app     = min_n_app
        self.max_pix_tol   = max_pix_tol
        self.min_n_refstar = min_n_refstar
        self.alpha0        = float(self.wcsRef.to_header()['CRVAL1']) * u.deg
        self.delta0        = float(self.wcsRef.to_header()['CRVAL2']) * u.deg

        self.make_lithographic_and_filter_mask_corrections = make_lithographic_and_filter_mask_corrections

        if self.make_lithographic_and_filter_mask_corrections:
            correctionTableDir = os.environ['ACSGEODIST_CONFIG']

            dtab_chip1_path = '{0:s}/wfc1.f606w.64x64.dtab'.format(correctionTableDir)
            dtab_chip2_path = '{0:s}/wfc2.f606w.64x64.dtab'.format(correctionTableDir)
            ftab_chip1_path = '{0:s}/wfc1.f606w.64x64.ffftab'.format(correctionTableDir)
            ftab_chip2_path = '{0:s}/wfc2.f606w.64x64.ffftab'.format(correctionTableDir)

            self.dtabs = [dtab_chip2_path, dtab_chip1_path]
            self.ftabs = [ftab_chip2_path, ftab_chip1_path]
        else:
            self.dtabs = None
            self.ftabs = None

        self.cross_match = cross_match

        if self.cross_match:
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            Gaia.ROW_LIMIT       = -1

            coord = SkyCoord(ra=self.alpha0, dec=self.delta0, frame='icrs')

            g = Gaia.query_object_async(coordinate=coord, width=WIDTH, height=HEIGHT)

            gaia_selection = (g['ruwe'] >= MIN_RUWE) & (g['ruwe'] <= MAX_RUWE)

            g = g[gaia_selection]

            self.gdr3_id = np.array(['GDR3_{0:d}'.format(sourceID) for sourceID in g['SOURCE_ID']], dtype=str)

            self.c_gdr3 = SkyCoord(ra=g['ra'].value * u.deg, dec=g['dec'].value * u.deg,
                              pm_ra_cosdec=g['pmra'].value * u.mas / u.yr, pm_dec=g['pmdec'].value * u.mas / u.yr,
                              obstime=Time(g['ref_epoch'].value, format='jyear', scale='tcb'))

        if (not self.individualZP):
            print("INDIVIDUAL CHIP ZERO POINT = FALSE. ZERO POINT FOR CHIP 2 IS MEASURED RELATIVE TO CHIP 1.")

    def processHST1PassFile(self, pOrder, hst1passFile, imageFilename, outDir='.', **kwargs):
        addendumFilename = hst1passFile.replace('.csv', '_addendum.csv')

        baseImageFilename = os.path.basename(hst1passFile).replace('_hst1pass_stand.csv', '')

        hduList = fits.open(imageFilename)

        tExp = float(hduList[0].header['EXPTIME'])
        tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
        t_acs = Time(tstring, scale='ut1', format='fits')

        pa_v3 = float(hduList[0].header['PA_V3'])

        posTarg1 = float(hduList[0].header['POSTARG1'])
        posTarg2 = float(hduList[0].header['POSTARG2'])

        dt = t_acs.decimalyear - self.tRef0.utc.value

        ## We use the observation time, in combination with the proper motions to move
        ## the coordinates into the time
        self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
        self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

        hst1pass = table.hstack([ascii.read(hst1passFile), ascii.read(addendumFilename)])

        matchRes = np.sqrt((hst1pass['xPred'] - hst1pass['xRef']) ** 2 + (hst1pass['yPred'] - hst1pass['yRef']) ** 2)

        okays = np.zeros(chips.size, dtype='bool')
        nGoodData = np.zeros(chips.size, dtype=int)
        for jj in range(acsconstants.N_CHIPS):
            chip = acsconstants.CHIP_NUMBER

            selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                        hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                    hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                    matchRes <= self.max_pix_tol)

            nGoodData[jj] = len(hst1pass[selection])

            if (nGoodData[jj] > self.min_n_refstar):
                okays[jj] = True

            print(acsconstants.CHIP_LABEL(acsconstants.WFC[jj], acsconstants.CHIP_POSITIONS[jj]), "N_STARS =",
                  nGoodData[jj], "OKAY:", okays[jj])

        del selection
        gc.collect()

        okayToProceed = np.prod(okays, dtype='bool')

        print("OKAY TO PROCEED:", okayToProceed)

        textResults = None

        if okayToProceed:
            ## Final table filename
            outTableFilename = '{0:s}/{1:s}_hst1pass_stand_pOrder{2:d}_resids.csv'.format(outDir,
                                                                                          baseImageFilename, pOrder)

            if (not os.path.exists(outTableFilename)):
                startTime0 = time.time()

                ## Plotting detected sources
                xSize1 = 12
                ySize1 = xSize1

                nRows = 2
                nCols = 1

                fig1, axes = plt.subplots(figsize=(xSize1, ySize1), nrows=nRows, ncols=nCols, rasterized=True)

                ## Save the old match residuals before it is wiped out with nans
                delX = hst1pass['xPred'] - hst1pass['xRef']
                delY = hst1pass['yPred'] - hst1pass['yRef']

                ## Change the columns with default values
                hst1pass['xPred']    = np.nan
                hst1pass['yPred']    = np.nan
                hst1pass['xRef']     = np.nan
                hst1pass['yRef']     = np.nan
                hst1pass['dx']       = np.nan
                hst1pass['dy']       = np.nan
                hst1pass['retained'] = False
                hst1pass['weights']  = 0.0  ## Final weights for all detected sources in the chip
                hst1pass['xi']       = np.nan
                hst1pass['eta']      = np.nan
                hst1pass['xiRef']    = np.nan
                hst1pass['etaRef']   = np.nan
                hst1pass['resXi']    = np.nan
                hst1pass['resEta']   = np.nan
                hst1pass['alpha']    = np.nan
                hst1pass['delta']    = np.nan
                hst1pass['sourceID'] = np.zeros(len(hst1pass), dtype='<U24')

                if not self.individualZP:
                    dxs, dys, rolls = [], [], []

                textResults = ""
                for jj in range(acsconstants.N_CHIPS):
                    startTime = time.time()

                    ver = jj + 1
                    jjj = acsconstants.N_CHIPS - ver  ## Index for plotting

                    chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[jj], acsconstants.CHIP_POSITIONS[jj])

                    hdu = hduList['SCI', ver]

                    k = int(hdu.header['CCDCHIP'])

                    ## Zero point of the y coordinates.
                    if (ver == 1):
                        yzp    = 0.0
                        naxis2 = int(hdu.header['NAXIS2'])
                    else:
                        yzp += float(naxis2)

                        naxis2 = int(hdu.header['NAXIS2'])

                    orientat = Angle(float(hdu.header['ORIENTAT']) * u.deg).wrap_at('360d').value
                    vaFactor = float(hdu.header['VAFACTOR'])

                    selection = (hst1pass['k'] == k) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                                hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                            hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                            matchRes <= self.max_pix_tol)

                    refStarIdx = hst1pass[selection]['refCatIndex'].value

                    nData = len(refStarIdx)

                    print("CHIP:", chipTitle, "P_ORDER:", pOrder, "N_STARS:", nData)

                    xi  = self.refCat[refStarIdx]['xt'] / vaFactor
                    eta = self.refCat[refStarIdx]['yt'] / vaFactor

                    XC = hst1pass['X'][selection] - X0
                    YC = hst1pass['Y'][selection] - Y0[jj]

                    if self.make_lithographic_and_filter_mask_corrections:
                        dcorr = np.array(litho.interp_dtab_ftab_data(self.dtabs[jj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T
                        fcorr = np.array(litho.interp_dtab_ftab_data(self.ftabs[jj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                        ## print(dcorr)
                        ## print(fcorr)

                        ## Apply the lithographic and filter mask correction
                        XC -= (dcorr[:, 2] - fcorr[:, 2])
                        YC -= (dcorr[:, 3] - fcorr[:, 3])

                        del dcorr
                        del fcorr

                    xSize = 8
                    ySize = xSize

                    X, scalerArray = sip.buildModel(XC, YC, pOrder, scalerX=scalerX, scalerY=scalerY)

                    plotFilename1 = "{0:s}/plot_{1:s}_chip{2:d}_pOrder{3:d}_residualDistribution.pdf".format(outDir,
                                                                                                             baseImageFilename,
                                                                                                             chip, pOrder)

                    pp1 = PdfPages(plotFilename1)

                    plotFilename2 = "{0:s}/plot_{1:s}_chip{2:d}_pOrder{3:d}_residualsXY.pdf".format(outDir,
                                                                                                    baseImageFilename, chip,
                                                                                                    pOrder)

                    pp2 = PdfPages(plotFilename2)

                    alpha0Im = float(hdu.header['CRVAL1'])
                    delta0Im = float(hdu.header['CRVAL2'])

                    xi0, eta0 = self.wcsRef.wcs_world2pix(np.array([alpha0Im]), np.array([delta0Im]), 1)

                    ## Initialize shift and rotation
                    sx, sy, roll = xi0[0], eta0[0], np.deg2rad(orientat)

                    ## Initialize the reference coordinates
                    xiRef  = deepcopy(xi)
                    etaRef = deepcopy(eta)

                    ## Initialize the raw coordinates
                    xyRaw = np.vstack([XC, YC]).T

                    ## Initialize the indices
                    indices = np.argwhere(selection)

                    ## Initialize the weights using the match residuals
                    ## weights = np.ones(XC.size)
                    residuals = np.array([delX[selection], delY[selection]]).T

                    ## Use the weights to estimate the mean and covariance matrix of the residual distribution
                    mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals, np.ones(residuals.shape[0]))

                    weights = stat.wdecay(stat.getMahalanobisDistances(residuals, mean, np.linalg.inv(cov)))

                    previousWeightSum = np.sum(weights)

                    ## IF we want to have individual zero points for each chip we initialize the container for shifts
                    ## and rolls here
                    if self.individualZP:
                        dxs, dys, rolls = [], [], []

                    nIterTotal = 0
                    for iteration in range(N_ITER_OUTER):
                        if (self.individualZP or (chip == 1)):
                            dxs.append(sx)
                            dys.append(sy)
                            rolls.append(roll)

                            xiRef, etaRef = coords.shift_rotate_coords(xiRef, etaRef, sx, sy, roll)
                        else:
                            for (sx, sy, roll) in zip(dxs, dys, rolls):
                                xiRef, etaRef = coords.shift_rotate_coords(xiRef, etaRef, sx, sy, roll)

                        for iteration2 in range(N_ITER_INNER):
                            nIterTotal += 1

                            nStars = xiRef.size

                            ## Initialize the linear regression
                            reg = linear_model.LinearRegression(fit_intercept=False, copy_X=False)

                            reg.fit(X, xiRef / scalerX, sample_weight=weights)

                            coeffsA = reg.coef_ * scalerX / scalerArray

                            reg.fit(X, etaRef / scalerY, sample_weight=weights)

                            coeffsB = reg.coef_ * scalerY / scalerArray

                            coeffs = np.zeros((coeffsA.size + coeffsB.size))

                            coeffs[0::2] = coeffsA
                            coeffs[1::2] = coeffsB

                            xiPred  = np.matmul(X * scalerArray, coeffs[0::2])
                            etaPred = np.matmul(X * scalerArray, coeffs[1::2])

                            ## Residuals already in pixel and in image axis
                            residualsXi  = xiRef - xiPred
                            residualsEta = etaRef - etaPred

                            rmsXi  = np.sqrt(np.average(residualsXi ** 2, weights=weights))
                            rmsEta = np.sqrt(np.average(residualsEta ** 2, weights=weights))

                            residuals = np.vstack([residualsXi, residualsEta]).T

                            ## Use the weights to estimate the mean and covariance matrix of the residual
                            ## distribution
                            mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals, weights)

                            ## Calculate the Mahalanobis Distance, i.e. standardized distance
                            ## from the center of the gaussian distribution
                            z = stat.getMahalanobisDistances(residuals, mean, np.linalg.inv(cov))

                            ## We now use the z statistics to re-calculate the weights
                            weights = stat.wdecay(z)

                            ## What we now call 'retained' are those stars with full weight
                            retained0 = weights >= 1.0

                            ## We have non-full weight stars but non-zero weights
                            nonFull = (~retained0) & (weights > 0)

                            ## Finally those stars with zero weights
                            rejected = weights <= 0

                            weightSum = np.sum(weights)

                            weightSumDiff = np.abs(weightSum - previousWeightSum) / weightSum

                            ## Assign the current weight summation for the next iteration
                            previousWeightSum = weightSum

                            print(baseImageFilename, k, pOrder, iteration + 1, iteration2 + 1,
                                  '{0:.3e} {1:.3e} {2:.3e}'.format(dxs[-1], dys[-1], rolls[-1]),
                                  "N_STARS: {0:d}/{1:d}".format(xiRef[~rejected].size, (xi.size)),
                                  "RMS: {0:.6f} {1:.6f}".format(rmsXi, rmsEta), "W_SUM: {0:0.6f}".format(weightSum))

                            fig = plt.figure(figsize=(xSize, ySize), rasterized=True)

                            ax = fig.add_subplot(111)

                            ax.plot(residuals[rejected][:, 0], residuals[rejected][:, 1], '.', markersize=markerSize,
                                    label=r'$w = 0$', color=discardedColor)
                            ax.plot(residuals[nonFull][:, 0], residuals[nonFull][:, 1], '.', markersize=markerSize,
                                    label=r'$0 < w < 1$', color=nonFullColor)
                            ax.plot(residuals[retained0][:, 0], residuals[retained0][:, 1], '.', markersize=markerSize,
                                    label=r'$w = 1$', color=retainedColor)

                            ax.axhline()
                            ax.axvline()

                            xMin1, xMax1 = ax.get_xlim()
                            yMin1, yMax1 = ax.get_ylim()

                            maxRange = np.nanmax(np.abs(np.array([xMin1, xMax1, yMin1, yMax1])))

                            ax.set_xlim(-maxRange, +maxRange)
                            ax.set_ylim(-maxRange, +maxRange)

                            ax.set_aspect('equal')

                            ax.set_xlabel(r'$\Delta X$ [pix]')
                            ax.set_ylabel(r'$\Delta Y$ [pix]')

                            ax.xaxis.set_major_locator(AutoLocator())
                            ax.xaxis.set_minor_locator(AutoMinorLocator())

                            ax.yaxis.set_major_locator(AutoLocator())
                            ax.yaxis.set_minor_locator(AutoMinorLocator())

                            ax.legend(frameon=True)

                            ax.set_title(
                                '{0:s}, {1:s}, $p$ = {2:d}, iter1 {3:d}, iter2 {4:d}'.format(baseImageFilename, chipTitle,
                                                                                             pOrder, iteration + 1,
                                                                                             iteration2 + 1))

                            pp1.savefig(fig)

                            plt.close(fig=fig)

                            ## Plotting residuals
                            xSize2 = 12
                            ySize2 = 0.5 * xSize2

                            nRows2 = 2
                            nCols2 = 2

                            fig2, axes2 = plt.subplots(figsize=(xSize2, ySize2), nrows=nRows2, ncols=nCols2,
                                                       rasterized=True)

                            xLabels = [r'$X_{\rm raw}$ [pix]', r'$Y_{\rm raw}$ [pix]']
                            yLabels = [r'$\Delta X$ [pix]', r'$\Delta Y$ [pix]']

                            XY0 = np.array([X0, Y0[0]])

                            xMin = np.array([-2048, -1024])
                            xMax = np.array([+2048, +1024])

                            yMin = +np.inf
                            yMax = -np.inf

                            for axis1 in range(NAXIS):
                                for axis2 in range(NAXIS):
                                    coordinatesDiscarded = xyRaw[rejected][:, axis2]
                                    residualsDiscarded = residuals[rejected][:, axis1]

                                    coordinatesNonFull = xyRaw[nonFull][:, axis2]
                                    residualsNonFull = residuals[nonFull][:, axis1]

                                    coordinatesRetained = xyRaw[retained0][:, axis2]
                                    residualsRetained = residuals[retained0][:, axis1]

                                    mean = np.nanmean(residuals[retained0][:, axis1])
                                    stdDev = np.nanstd(residuals[retained0][:, axis1])

                                    axes2[axis1, axis2].plot(coordinatesDiscarded, residualsDiscarded, '.',
                                                             markersize=markerSize, zorder=1, label=r'$w = 0$',
                                                             color=discardedColor)
                                    axes2[axis1, axis2].plot(coordinatesNonFull, residualsNonFull, '.',
                                                             markersize=markerSize, zorder=1, label=r'$0 < w < 1$',
                                                             color=nonFullColor)
                                    axes2[axis1, axis2].plot(coordinatesRetained, residualsRetained, '.',
                                                             markersize=markerSize, zorder=1, label=r'$w = 1$',
                                                             color=retainedColor)

                                    axes2[axis1, axis2].axhline(0,
                                                                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                                                                linestyle='--', zorder=2)

                                    axes2[axis1, axis2].axhline(mean, color='r', linestyle='-', zorder=3)

                                    ## xMin, xMax = axes2[axis1,axis2].get_xlim()

                                    resMin = mean - stdDev
                                    resMax = mean + stdDev

                                    yMin = np.nanmin(np.array([yMin, mean - 5.0 * stdDev, np.nanmin(residuals)]))
                                    yMax = np.nanmax(np.array([yMax, mean + 5.0 * stdDev, np.nanmax(residuals)]))

                                    axes2[axis1, axis2].fill_between([xMin[axis2], xMax[axis2]], [resMin, resMin],
                                                                     y2=[resMax, resMax], alpha=0.2, zorder=3)

                                    axes2[axis1, axis2].set_xlabel(xLabels[axis2])
                                    axes2[axis1, axis2].set_ylabel(yLabels[axis1])

                                    axes2[axis1, axis2].xaxis.set_major_locator(MultipleLocator(dX))
                                    axes2[axis1, axis2].xaxis.set_minor_locator(MultipleLocator(dMX))

                                    axes2[axis1, axis2].yaxis.set_major_locator(AutoLocator())
                                    axes2[axis1, axis2].yaxis.set_minor_locator(AutoMinorLocator())

                                    axes2[axis1, axis2].set_xlim(xMin[axis2], xMax[axis2])

                            ## print("Y_MIN:", yMin, "Y_MAX:", yMax)

                            yMaxMin = np.nanmax(np.abs(np.array([yMin, yMax])))

                            for axis1 in range(NAXIS):
                                for axis2 in range(NAXIS):
                                    axes2[axis1, axis2].set_ylim([-yMaxMin, +yMaxMin])

                            ## axes2[0,0].legend()

                            axCommons = plotting.drawCommonLabel('', '', fig2, xPad=0, yPad=0)

                            axCommons.set_title(
                                '{0:s}, {1:s}, $p$ = {2:d}, iter1 {3:d}, iter2 {4:d}'.format(baseImageFilename, chipTitle,
                                                                                             pOrder, iteration + 1,
                                                                                             iteration2 + 1))

                            plt.subplots_adjust(wspace=0.25, hspace=0.3)

                            pp2.savefig(fig2, bbox_inches='tight', dpi=300)

                            plt.close(fig=fig2)

                            del reg
                            gc.collect()

                            ## At the last iteration, re-calculate the shift and rolls
                            ## if ((iteration2+1) == (N_ITER_INNER)):
                            if ((weightSumDiff < 1.e-9) or (iteration2 + 1) == (N_ITER_INNER)):
                                ## Shift and rotate the reference coordinates using the new
                                ## zero-th order coefficients and rotation angle
                                sx, sy = coeffs[0], coeffs[1]
                                epsilon = np.arctan(coeffs[4] / coeffs[5])

                                roll = -epsilon

                                break
                            else:
                                X = X[~rejected]

                                weights = weights[~rejected]

                                xiRef = xiRef[~rejected]
                                etaRef = etaRef[~rejected]

                                xyRaw = xyRaw[~rejected]

                                indices = indices[~rejected]

                        if (not (self.individualZP or (chip == 1))):
                            break

                    ## print("SHIFTS_X:", dxs)
                    ## print("SHIFTS_Y:", dys)
                    ## print("ROLLS:", rolls)

                    ## Now that we have the coefficients, we repeat the model building for ALL objects in the chip
                    selection = (hst1pass['k'] == k)

                    hasRefStar = (hst1pass['refCatIndex'] >= 0)

                    refStarIdx = hst1pass[selection]['refCatIndex'].value

                    xiRef  = self.refCat[refStarIdx]['xt'] / vaFactor
                    etaRef = self.refCat[refStarIdx]['yt'] / vaFactor

                    XC = hst1pass['X'][selection] - X0
                    YC = hst1pass['Y'][selection] - Y0[jj]

                    if self.make_lithographic_and_filter_mask_corrections:
                        dcorr = np.array(litho.interp_dtab_ftab_data(self.dtabs[jj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T
                        fcorr = np.array(litho.interp_dtab_ftab_data(self.ftabs[jj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                        ## Apply the lithographic mask correction
                        XC -= (dcorr[:, 2] - fcorr[:, 2])
                        YC -= (dcorr[:, 3] - fcorr[:, 3])

                    X, scalerArray = sip.buildModel(XC, YC, pOrder, scalerX=scalerX, scalerY=scalerY)

                    for iiii in range(len(dxs)):
                        sx   = dxs[iiii]
                        sy   = dys[iiii]
                        roll = rolls[iiii]

                        xiRef, etaRef = coords.shift_rotate_coords(xiRef, etaRef, sx, sy, roll)

                    xiPred  = np.matmul(X * scalerArray, coeffs[0::2])
                    etaPred = np.matmul(X * scalerArray, coeffs[1::2])

                    hst1pass['xPred'][selection] = xiPred
                    hst1pass['yPred'][selection] = etaPred
                    hst1pass['xRef'][selection] = xiRef
                    hst1pass['yRef'][selection] = etaRef
                    hst1pass['dx'][selection] = xiRef - xiPred
                    hst1pass['dy'][selection] = etaRef - etaPred
                    hst1pass['retained'][indices] = True
                    hst1pass['weights'][indices.flatten()] = weights.flatten()

                    ## Assign NaNs to values of non-reference stars
                    hst1pass['xRef'][selection & ~hasRefStar] = np.nan
                    hst1pass['yRef'][selection & ~hasRefStar] = np.nan
                    hst1pass['dx'][selection & ~hasRefStar] = np.nan
                    hst1pass['dy'][selection & ~hasRefStar] = np.nan

                    pp1.close()

                    print("Residual 2d distribution plots saved to {0:s}".format(plotFilename1))

                    pp2.close()

                    print("Residual XY-distribution plots saved to {0:s}".format(plotFilename2))

                    ## Finally, we calculate the CD Matrix used to transform the intermediate world coordinate
                    ## We take the reference coordinate to be the CRVAL1,2 in the header and a create a SkyCoord object
                    c0Im = SkyCoord(ra=alpha0Im * u.deg, dec=delta0Im * u.deg, frame='icrs')

                    ## Now we take the normal triad pqr_0 of the reference coordinate
                    pqr0Im = coords.getNormalTriad(c0Im)

                    selection  = (hst1pass['k'] == k) & (hst1pass['refCatIndex'] >= 0) & hst1pass['retained']
                    refStarIdx = hst1pass['refCatIndex'][selection].value

                    XCorr = hst1pass['xPred'][selection].value
                    YCorr = hst1pass['yPred'][selection].value

                    for iteration3 in range(N_ITER_CD):
                        ## Calculate the normal coordinates relative to the pqr triad centered on the current CRVAL1,2
                        self.refCat = self._getNormalCoordinates(self.refCat, 'xt', 'yt', self.wcsRef, pqr0Im)

                        ## The reference coordinates used for regression of the CD matrix is relative to the current
                        ## CRVAL1, CRVAL2 coordinates
                        xiRef  = (self.refCat['xi'][refStarIdx].value  * u.arcsec).to(u.deg) / vaFactor
                        etaRef = (self.refCat['eta'][refStarIdx].value * u.arcsec).to(u.deg) / vaFactor

                        ## Store the reference coordinates back in pixel scale
                        hst1pass['xiRef'][selection] = xiRef.to_value(u.arcsec) / acsconstants.ACS_PLATESCALE.to_value(
                            u.arcsec / u.pix)
                        hst1pass['etaRef'][selection] = etaRef.to_value(u.arcsec) / acsconstants.ACS_PLATESCALE.to_value(
                            u.arcsec / u.pix)

                        ## Calculate the CD Matrix that transform the corrected coordinates
                        ## XCorr, YCorr into normal coordinates Xi, Eta
                        CDMatrix = self._getCDMatrix(XCorr, YCorr, xiRef.value, etaRef.value,
                                               weights=hst1pass['weights'][selection].value)

                        ## Replace the CRVAL1, CRVAL2 coordinates using the constants of the CD Matrix
                        c0Im = coords.getCelestialCoordinatesFromNormalCoordinates(CDMatrix[0, 0] * u.deg,
                                                                                   CDMatrix[1, 0] * u.deg, c0Im,
                                                                                   frame='icrs')

                        ## Replace the triad pqr_0 using the new value of CRVAL1, CRVAL2
                        pqr0Im = coords.getNormalTriad(c0Im)

                        ## Apply the CD Matrix to all sources in the chip, in order to obtain the new normal coordinates
                        selectionChip = hst1pass['k'] == k

                        H, _ = sip.buildModel(hst1pass['xPred'][selectionChip].value,
                                              hst1pass['yPred'][selectionChip].value,
                                              1)

                        ## Calculate the normal coordinates Xi, Eta and assign them to the table
                        hst1pass['xi'][selectionChip] = (
                                    ((H @ CDMatrix[0]) * u.deg) / acsconstants.ACS_PLATESCALE).to_value(u.pix)
                        hst1pass['eta'][selectionChip] = (
                                    ((H @ CDMatrix[1]) * u.deg) / acsconstants.ACS_PLATESCALE).to_value(u.pix)

                        ## Calculate the residuals
                        hst1pass['resXi'][selection] = (
                                    hst1pass['xi'][selection].value - hst1pass['xiRef'][selection].value)
                        hst1pass['resEta'][selection] = (
                                    hst1pass['eta'][selection].value - hst1pass['etaRef'][selection].value)

                        print("CD_ITERATION:", (iteration3 + 1))
                        print(CDMatrix)
                        print(hst1pass['resXi', 'resEta'][selection].to_pandas().describe())

                    alpha0Im, delta0Im = c0Im.ra.value, c0Im.dec.value

                    xi0, eta0 = self.wcsRef.wcs_world2pix(np.array([alpha0Im]), np.array([delta0Im]), 1)

                    xi0  = float(self.wcsRef.to_header()['CRPIX1']) - xi0[0]
                    eta0 = eta0[0] - float(self.wcsRef.to_header()['CRPIX2'])

                    selection = (hst1pass['k'] == k)

                    hst1pass['xi'][selection]  = hst1pass['xi'][selection]  + xi0
                    hst1pass['eta'][selection] = hst1pass['eta'][selection] + eta0

                    hst1pass['xiRef'][selection]  = hst1pass['xiRef'][selection] + xi0
                    hst1pass['etaRef'][selection] = hst1pass['etaRef'][selection] + eta0

                    ## Calculate the residuals
                    hst1pass['resXi'][selection] = (
                            hst1pass['xi'][selection].value - hst1pass['xiRef'][selection].value)
                    hst1pass['resEta'][selection] = (
                            hst1pass['eta'][selection].value - hst1pass['etaRef'][selection].value)

                    rmsXi  = np.nan
                    rmsEta = np.nan

                    residual_selection = (np.isfinite(hst1pass['resXi'][selection].value) &
                                          np.isfinite(hst1pass['resEta'][selection].value) &
                                          np.isfinite(hst1pass['weights'][selection].value))

                    if (np.sum(hst1pass['weights'][selection].value) > 0) and (
                            residual_selection[residual_selection].size > 0):
                        rmsXi  = np.sqrt(stat.getWeightedAverage(hst1pass['resXi'][selection].value ** 2,
                                                                hst1pass['weights'][selection].value))
                        rmsEta = np.sqrt(stat.getWeightedAverage(hst1pass['resEta'][selection].value ** 2,
                                                                 hst1pass['weights'][selection].value))

                    textResults += "{0:s} {1:d} {2:.8f} {3:.6f} {4:.13f} {5:.12e} {6:0.2f} {7:f} {8:f}".format(
                        baseImageFilename, k, t_acs.decimalyear, pa_v3, orientat, vaFactor, tExp, posTarg1, posTarg2)
                    textResults += " {0:d} {1:d}".format(nIterTotal, nStars)
                    textResults += " {0:0.6f} {1:0.6f}".format(rmsXi, rmsEta)
                    textResults += " {0:0.12f} {1:0.12f}".format(alpha0Im, delta0Im)
                    textResults += " {0:0.12e} {1:0.12e} {2:0.12e} {3:0.12e}".format(CDMatrix[0, 1], CDMatrix[0, 2],
                                                                                     CDMatrix[1, 1], CDMatrix[1, 2])
                    for coeff in coeffs:
                        textResults += " {0:0.12e}".format(coeff)
                    textResults += "\n"

                    ## Repeat the selection process
                    selection = (hst1pass['k'] == k) & (hst1pass['refCatID'] >= 0) & (hst1pass['q'] > 0) & (
                                hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                            hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                            matchRes <= self.max_pix_tol)

                    retained = np.zeros(len(hst1pass), dtype=bool)

                    retained[indices] = True

                    image = hdu.data

                    vmin, vmax = ZScaleInterval(contrast=0.10, max_iterations=5).get_limits(image)

                    axes[jjj].imshow(image, cmap=cMap, aspect='equal', vmin=vmin, vmax=vmax, origin='lower', rasterized=True)

                    xyPosRetained = np.vstack([hst1pass[retained]['X'].value, hst1pass[retained]['Y'].value - yzp]).T

                    aperturesRetained = CircularAperture(xyPosRetained, r=20.0)

                    aperturesRetained.plot(color='#1b9e77', lw=1, alpha=1, ax=axes[jjj],
                                           rasterized=True);  ## GREENS are accepted sources

                    xyPosRejected = np.vstack([hst1pass[(~retained) & selection]['X'].value,
                                               hst1pass[(~retained) & selection]['Y'].value - yzp]).T

                    aperturesRejected = CircularAperture(xyPosRejected, r=15.0)

                    aperturesRejected.plot(color='#d95f02', lw=1, alpha=1, ax=axes[jjj],
                                           rasterized=True);  ## BROWNS are accepted sources

                    axes[jjj].set_title('{0:s} --- {1:s}'.format(baseImageFilename, chipTitle))

                    axes[jjj].xaxis.set_major_locator(MultipleLocator(dX))
                    axes[jjj].xaxis.set_minor_locator(MultipleLocator(dMX))

                    axes[jjj].yaxis.set_major_locator(MultipleLocator(dY))
                    axes[jjj].yaxis.set_minor_locator(MultipleLocator(dMY))

                    elapsedTime = time.time() - startTime
                    print("FITTING DONE FOR {0:s}.".format(chipTitle), "Elapsed time:", convertTime(elapsedTime))

                plt.subplots_adjust(wspace=0.0, hspace=0.15)

                axCommons = plotting.drawCommonLabel(xLabel, yLabel, fig1, xPad=20, yPad=15)

                plotFilename3 = "{0:s}/plot_{1:s}_pOrder{2:d}_retainedSources.pdf".format(outDir, baseImageFilename,
                                                                                          pOrder)
                fig1.savefig(plotFilename3, bbox_inches='tight', dpi=300)

                print("Image saved to {0:s}".format(plotFilename3))

                plt.close(fig=fig1)

                ## Assign name for each sources in each chip. We first grab the xi, eta from the catalogue.
                xi  = (hst1pass['xi']  * u.pix) * acsconstants.ACS_PLATESCALE
                eta = (hst1pass['eta'] * u.pix) * acsconstants.ACS_PLATESCALE

                ## Use the zero-point of the reference catalogue and declare a SkyCoord object from zero-point.
                c0 = SkyCoord(ra=self.alpha0, dec=self.delta0, frame='icrs')

                ## Find only sources with defined xi and eta. Don't worry if they're crap sources, we'll deal with them
                ## later in the next phase
                argsel = np.argwhere(~np.isnan(xi) & ~np.isnan(eta)).flatten()

                ## Calculate the equatorial coordinates and assign them to the table
                c = coords.getCelestialCoordinatesFromNormalCoordinates(xi[argsel], eta[argsel], c0, frame='icrs')

                hst1pass['alpha'][argsel] = c.ra.value
                hst1pass['delta'][argsel] = c.dec.value

                ## Based on the equatorial coordinates assign a source ID for each source
                hst1pass['sourceID'][argsel] = astro.generateSourceID(c)

                ## Now we query the Gaia catalogue, if cross_match is set to True. For this we will have different
                ## criteria than before. We now only cross-match sources with 0 < q <= Q_MAX, for these are more likely
                ## to be bona-fide point-sources (i.e. stars).
                if self.cross_match:
                    argsel = np.argwhere(~np.isnan(xi) & ~np.isnan(eta) & (hst1pass['q'] > Q_MIN) & (hst1pass['q'] <= Q_MAX)).flatten()

                    c = SkyCoord(ra=hst1pass['alpha'][argsel] * u.deg, dec=hst1pass['delta'][argsel] * u.deg, frame='icrs')

                    self.c_gdr3 = self.c_gdr3.apply_space_motion(t_acs)

                    idx, sep, _ = c.match_to_catalog_sky(self.c_gdr3)

                    sep_pix = sep.to(u.mas) / acsconstants.ACS_PLATESCALE

                    selection_gdr3 = sep_pix < MAX_SEP

                    ## We now assign a different source ID for sources with known GDR3 stars counterpart
                    hst1pass['sourceID'][argsel[selection_gdr3]] = self.gdr3_id[idx[selection_gdr3]]

                ## Write the final table
                hst1pass.write(outTableFilename, overwrite=True)

                print("Final table written to", outTableFilename)

                ## Plot the coordinates and their residuals on a common reference frame
                xSize3 = 12
                ySize3 = 1.0075 * xSize3

                xMin3, xMax3 = -5500, +5500
                yMin3, yMax3 = xMin3, xMax3

                dX3, dMX3 = 2000, 500
                dY3, dMY3 = dX3, dMX3

                resMin = -0.29
                resMax = +0.29
                dRes   = 0.2
                dMRes  = 0.05

                markerSize3 = 8

                fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(xSize3, ySize3), sharex='col', sharey='row',
                                       width_ratios=[1.0, 0.25, 0.25], height_ratios=[0.25, 0.25, 1.0])

                ax3[0,0].set_title(hduList[0].header['ROOTNAME']+' --- '+hduList[0].header['DATE-OBS']+' UT'+hduList[0].header['TIME-OBS'])

                ax3[0, 1].set_visible(False)
                ax3[0, 2].set_visible(False)
                ax3[1, 2].set_visible(False)

                print("FINAL RESIDUALS (COMBINED):")
                df_resids = hst1pass.to_pandas()

                selection = (df_resids['refCatIndex'] >= 0) & df_resids['retained']

                print(df_resids.loc[selection, ['resXi', 'resEta']].describe())

                sns.scatterplot(data=df_resids[selection], x='resXi', y='resEta', hue='weights', legend=False, ax=ax3[1, 1],
                                s=markerSize3, rasterized=True)

                sns.scatterplot(data=df_resids[selection], x='xi', y='resXi', hue='weights', legend=False, ax=ax3[0, 0],
                                s=markerSize3, rasterized=True)
                sns.scatterplot(data=df_resids[selection], x='xi', y='resEta', hue='weights', legend=False, ax=ax3[1, 0],
                                s=markerSize3, rasterized=True)

                sns.scatterplot(data=df_resids[selection], x='xi', y='eta', hue='weights', legend=True, ax=ax3[2, 0],
                                s=markerSize3, rasterized=True)

                sns.scatterplot(data=df_resids[selection], x='resXi', y='eta', hue='weights', legend=False, ax=ax3[2, 1],
                                s=markerSize3, rasterized=True)
                sns.scatterplot(data=df_resids[selection], x='resEta', y='eta', hue='weights', legend=False, ax=ax3[2, 2],
                                s=markerSize3, rasterized=True)

                resLabels = ['res_xi [pix]', 'res_eta [pix]']

                for axis in range(NAXIS):
                    ax3[2, axis + 1].set_xlabel(resLabels[axis])
                    ax3[axis, 0].set_ylabel(resLabels[axis])

                    ax3[2, axis + 1].xaxis.set_major_locator(MultipleLocator(dRes))
                    ax3[2, axis + 1].xaxis.set_minor_locator(MultipleLocator(dMRes))

                    ax3[2, axis + 1].yaxis.set_major_locator(MultipleLocator(dY3))
                    ax3[2, axis + 1].yaxis.set_minor_locator(MultipleLocator(dMY3))

                    ax3[axis, 0].xaxis.set_major_locator(MultipleLocator(dX3))
                    ax3[axis, 0].xaxis.set_minor_locator(MultipleLocator(dMX3))

                    ax3[axis, 0].yaxis.set_major_locator(MultipleLocator(dRes))
                    ax3[axis, 0].yaxis.set_minor_locator(MultipleLocator(dMRes))

                    ax3[2, axis + 1].set_xlim(resMin, resMax)
                    ax3[axis, 0].set_ylim(resMin, resMax)

                    ax3[axis, 0].axhline(y=0, linewidth=1)
                    ax3[2, axis+1].axvline(x=0, linewidth=1)

                ax3[1, 1].xaxis.set_major_locator(MultipleLocator(dRes))
                ax3[1, 1].xaxis.set_minor_locator(MultipleLocator(dMRes))
                ax3[1, 1].yaxis.set_major_locator(MultipleLocator(dRes))
                ax3[1, 1].yaxis.set_minor_locator(MultipleLocator(dMRes))

                ax3[1, 1].axvline(x=0)
                ax3[1, 1].axhline(y=0)

                ax3[2, 0].axhline(linestyle='--', color='r', y=0, linewidth=1)
                ax3[2, 0].axvline(linestyle='--', color='r', x=0, linewidth=1)

                ax3[2,0].set_xlabel(r'xi [pix]')
                ax3[2,0].set_ylabel(r'eta [pix]')

                for iii in range(3):
                    ax3[iii,0].xaxis.set_major_locator(MultipleLocator(dX3))
                    ax3[iii,0].xaxis.set_minor_locator(MultipleLocator(dMX3))

                    ax3[2,iii].yaxis.set_major_locator(MultipleLocator(dY3))
                    ax3[2,iii].yaxis.set_minor_locator(MultipleLocator(dMY3))

                ax3[2,0].set_xlim(xMin3, xMax3)
                ax3[2,0].set_ylim(yMin3, yMax3)

                ax3[2,0].set_aspect('equal')

                ax3[2,0].invert_xaxis()

                plt.subplots_adjust(wspace=0.0, hspace=0.0)

                plotFilename3 = "{0:s}/plot_{1:s}_pOrder{2:d}_retainedSources_commonCoordinates.pdf".format(outDir, baseImageFilename, pOrder)

                fig3.savefig(plotFilename3, dpi=300, bbox_inches='tight')

                plt.close(fig=fig3)

                elapsedTime0 = time.time() - startTime0
                print("P_ORDER = {0:d} DONE!".format(pOrder), "Elapsed time:", convertTime(elapsedTime0))
            else:
                print("P_ORDER = {0:d}:".format(pOrder), "Skipping analysis because final table exists already!")
        else:
            print(
                "NOT ENOUGH GOOD-QUALITY REFERENCE STARS IN THE PLATE CATALOGUE. N_REF_STARS = {0} (MININUM {1:d} ON ALL CHIPS)".format(
                    nGoodData, self.min_n_refstar))

        hduList.close()

        gc.set_threshold(2, 1, 1)
        ## print('Thresholds:', gc.get_threshold())
        ## print('Counts:', gc.get_count())

        del hduList
        del hst1pass
        gc.collect()
        ## print('Counts:', gc.get_count())

        return textResults

    def _getCDMatrix(self, xiInt, etaInt, xiRef, etaRef, weights=None, pOrder=1, scalerX=1.0, scalerY=1.0):
        H, _ = sip.buildModel(xiInt, etaInt, pOrder, scalerX=scalerX, scalerY=scalerY)

        reg = linear_model.LinearRegression(fit_intercept=False, copy_X=True)

        reg.fit(H, xiRef, sample_weight=weights)

        C1 = reg.coef_

        reg.fit(H, etaRef, sample_weight=weights)

        C2 = reg.coef_

        return np.vstack([C1, C2])

    def _getNormalCoordinates(self, tab, x, y, wcs, pqr0, selection=None):
        if selection is None:
            selection = np.full(len(tab), True, dtype=bool)

        tab['alpha'] = np.nan
        tab['delta'] = np.nan
        tab['xi'] = np.nan
        tab['eta'] = np.nan

        tab['alpha'][selection], tab['delta'][selection] = wcs.wcs_pix2world(tab[x][selection].value,
                                                                             tab[y][selection].value, 1)

        xi, eta = coords.getNormalCoordinates(
            SkyCoord(ra=tab['alpha'][selection].value * u.deg, dec=tab['delta'][selection].value * u.deg, frame='icrs'),
            pqr0)

        tab['xi'][selection] = xi.to_value(u.arcsec)
        tab['eta'][selection] = eta.to_value(u.arcsec)

        return tab

class TimeDependentBSplineEstimator(SIPEstimator):
    def __init__(self, tMin, tMax, referenceCatalog, referenceWCS, tRef0, pOrderIndiv, pOrder, kOrder, nKnots,
                 qMax=0.5, min_n_app=3, max_pix_tol=1.0, min_n_refstar=100, min_t_exp=99.0, max_pos_targs=0.0,
                 individualZP=True, make_lithographic_and_filter_mask_corrections=True, cross_match=True):
        super().__init__(referenceCatalog, referenceWCS, tRef0, qMax=qMax, min_n_app=min_n_app, max_pix_tol=max_pix_tol,
                       min_n_refstar=min_n_refstar,
                       make_lithographic_and_filter_mask_corrections=make_lithographic_and_filter_mask_corrections,
                       cross_match=cross_match, individualZP=individualZP)
        self.pOrderIndiv = pOrderIndiv ## Maximum polynomial order that are inferred individually for each image
        self.pOrder      = pOrder      ## Total polynomial orders, including those with time-dependent model
        self.kOrder      = kOrder      ## B-spline order
        self.nKnots      = nKnots      ## Number of knots
        self.tMin        = tMin
        self.tMax        = tMax

        ## These are numbers of parameters PER AXIS! Note the suffix A and B to indicate the axes
        n = self.pOrderIndiv + 1

        self.nParsSIP = sip.getUpperTriangularMatrixNumberOfElements(self.pOrder+1)

        self.indivParsIndices_A = []
        self.indivParsIndices_B = []

        for ii in range(n):
            for jj in range(n - ii):
                ppp = sip.getUpperTriangularIndex(jj, ii)
                self.indivParsIndices_A.append(ppp)
                self.indivParsIndices_B.append(ppp)

        if (self.pOrderIndiv < 1):
            self.indivParsIndices_A.append(2)

        self.indivParsIndices_A = np.array(sorted(self.indivParsIndices_A))
        self.indivParsIndices_B = np.array(sorted(self.indivParsIndices_B))

        self.nParsIndiv_A = self.indivParsIndices_A.size
        self.nParsIndiv_B = self.indivParsIndices_B.size

        self.splineParsIndices_A = []
        self.splineParsIndices_B = []

        n = self.pOrder + 1
        for ii in range(n):
            for jj in range(n - ii):
                ppp = sip.getUpperTriangularIndex(jj, ii)

                if ppp not in self.indivParsIndices_A:
                    self.splineParsIndices_A.append(ppp)
                if ppp not in self.indivParsIndices_B:
                    self.splineParsIndices_B.append(ppp)

        self.splineParsIndices_A = np.array(sorted(self.splineParsIndices_A))
        self.splineParsIndices_B = np.array(sorted(self.splineParsIndices_B))

        self.nParsSpline_A = self.splineParsIndices_A.size
        self.nParsSpline_B = self.splineParsIndices_B.size

        ## self.nParsSpline_A = sip.getUpperTriangularMatrixNumberOfElements(self.pOrder + 1) - self.nParsIndiv_A
        ## self.nParsSpline_B = sip.getUpperTriangularMatrixNumberOfElements(self.pOrder + 1) - self.nParsIndiv_B

        self.nParsK = self.nKnots + self.kOrder  ## Number of B-spline parameters include constant parameter (zero point)

        print("INDIVIDUAL PARAMETER INDICES A:", self.indivParsIndices_A, self.nParsIndiv_A)
        print("INDIVIDUAL PARAMETER INDICES B:", self.indivParsIndices_B, self.nParsIndiv_B)
        print("SPLINE PARAMETER INDICES A:", self.splineParsIndices_A, self.nParsSpline_A)
        print("SPLINE PARAMETER INDICES B:", self.splineParsIndices_B, self.nParsSpline_B)
        print("K_ORDER AND NUMBER OF KNOTS:", self.kOrder, self.nKnots)
        print("NUMBER OF B-SPLINE PARAMETERS (PER COEFFICIENT):", self.nParsK)

        self.tKnot  = np.linspace(self.tMin, self.tMax, self.nKnots, endpoint=True)
        self.dtKnot = self.tKnot[1] - self.tKnot[0]

        self.min_t_exp     = min_t_exp
        self.max_pos_targs = max_pos_targs

    def estimateTimeDependentBSplineCoefficients(self, hst1passFiles, imageFilenames, outDir='.', makePlots=True,
                                                 saveIntermediateResults=True, nCPUs=None, **kwargs):

        startTimeAll = time.time()

        self.nOkay    = 0
        self.nDataAll = np.zeros(2, dtype=int)

        self.okayIDs = []

        self.plateIDAll = []
        self.indicesAll = []

        self.tAll = []

        self.rootnamesAll = []

        self.xiAll = []
        self.etaAll = []

        self.XpAll_A  = []
        self.XkpAll_A = []

        self.XpAll_B  = []
        self.XkpAll_B = []

        self.xyRawAll = []

        self.dxAll = []
        self.dyAll = []
        self.rollAll = []

        self.nDataImages = []

        self.rootnames = []
        self.XtAll = []
        self.tObs  = []

        XAll_A = []
        XAll_B = []

        self.matchResAll = []

        startTime = time.time()

        indivDataFilenames   = []
        modelCoeffsFilenames = []
        outTableFilenames    = []

        okays = []

        for chip in chips:
            indivDataFilename = '{0:s}/individualCoefficients_chip{1:d}_tMin{2:0.4f}_tMax{3:0.4f}_FINAL.csv'.format(
                outDir, chip, self.tMin, self.tMax)

            modelCoeffsFilename = '{0:s}/splineCoefficients_chip{1:d}_tMin{2:0.4f}_tMax{3:0.4f}_FINAL.csv'.format(
                outDir, chip, self.tMin, self.tMax)

            outTableFilename = "{0:s}/resids_chip{1:d}_pOrder{2:d}_kOrder{3:d}_nKnots{4:d}_FINAL.csv".format(outDir,
                                                                                                             chip,
                                                                                                             self.pOrder,
                                                                                                             self.kOrder,
                                                                                                             self.nKnots)

            indivDataFilenames.append(indivDataFilename)
            modelCoeffsFilenames.append(modelCoeffsFilename)
            outTableFilenames.append(outTableFilename)

            if (os.path.exists(indivDataFilename) and os.path.exists(modelCoeffsFilename) and
                    os.path.exists(outTableFilename)):
                okays.append(True)
            else:
                okays.append(False)

            self.plateIDAll.append([])
            self.indicesAll.append([])
            self.tAll.append([])
            self.rootnamesAll.append([])

            self.xiAll.append([])
            self.etaAll.append([])

            self.XpAll_A.append([])
            self.XkpAll_A.append([])
            self.XpAll_B.append([])
            self.XkpAll_B.append([])

            self.xyRawAll.append([])

            self.dxAll.append([])
            self.dyAll.append([])
            self.rollAll.append([])

            self.nDataImages.append([])

            self.matchResAll.append([])

            XAll_A.append([])
            XAll_B.append([])

        proceed = ~np.prod(np.array(okays, dtype=bool), dtype=bool)

        if proceed:
            print("READING FILES...")
            self.scalerArray           = np.ones(self.nParsSIP)
            self.selectedHST1PassFiles = []
            self.selectedImageFiles    = []
            if (nCPUs is not None):
                argumentList = [(i, hst1passFile, imageFilename) for i, (hst1passFile, imageFilename) in
                                enumerate(zip(hst1passFiles, imageFilenames))]

                nJobs = len(argumentList)

                print("NUMBER OF CPUS:", nCPUs)
                print("NUMBER OF JOBS:", nJobs)

                pool = mp.Pool(min(nCPUs, nJobs))

                _ = pool.starmap_async(self._processFile, argumentList)

                pool.close()

                pool.join()
            else:
                for i, (hst1passFile, imageFilename) in enumerate(zip(hst1passFiles, imageFilenames)):
                    self._processFile(i, hst1passFile, imageFilename)

            print()
            print("NUMBER OF SELECTED FILES:", self.nOkay)

            self.okayIDs = np.array(self.okayIDs)

            self.nDataImages = np.array(self.nDataImages)

            self.rootnames = np.array(self.rootnames)
            self.XtAll     = np.vstack(self.XtAll)
            self.tObs      = np.array(self.tObs)

            gc.set_threshold(2, 1, 1)

            for chip in chips:
                jjj = chip - 1

                self.xiAll[jjj]  = np.hstack(self.xiAll[jjj])
                self.etaAll[jjj] = np.hstack(self.etaAll[jjj])

                self.plateIDAll[jjj]   = np.hstack(self.plateIDAll[jjj])
                self.indicesAll[jjj]   = np.hstack(self.indicesAll[jjj])
                self.tAll[jjj]         = np.hstack(self.tAll[jjj])
                self.rootnamesAll[jjj] = np.hstack(self.rootnamesAll[jjj])

                self.dxAll[jjj]   = np.hstack(self.dxAll[jjj])
                self.dyAll[jjj]   = np.hstack(self.dyAll[jjj])
                self.rollAll[jjj] = np.hstack(self.rollAll[jjj])

                if (len(self.XpAll_A[jjj]) > 0):
                    XAll_A[jjj] = sparse.hstack([sparse.block_diag(self.XpAll_A[jjj], format='csr'), sparse.vstack(self.XkpAll_A[jjj])])
                else:
                    XAll_A[jjj] = sparse.vstack(self.XkpAll_A[jjj])

                if (len(self.XpAll_B[jjj]) > 0):
                    XAll_B[jjj] = sparse.hstack([sparse.block_diag(self.XpAll_B[jjj], format='csr'), sparse.vstack(self.XkpAll_B[jjj])])
                else:
                    XAll_B[jjj] = sparse.vstack(self.XkpAll_B[jjj])

                self.xyRawAll[jjj] = np.vstack(self.xyRawAll[jjj])

                self.matchResAll[jjj] = np.vstack(self.matchResAll[jjj])

                self.XpAll_A[jjj]  = None
                self.XpAll_B[jjj]  = None
                self.XkpAll_A[jjj] = None
                self.XkpAll_B[jjj] = None

            del self.XpAll_A
            del self.XkpAll_A
            del self.XpAll_B
            del self.XkpAll_B
            gc.collect()

            elapsedTime = time.time() - startTime
            print("READING DATA AND BUILDING MODEL DONE! Elapsed time:", convertTime(elapsedTime))

            if makePlots:
                xSize1 = 12
                ySize1 = 0.25 * xSize1

                xMin, xMax = np.inf, -np.inf
                yMin, yMax = 0.0, 1.1

                dY, dMY = 0.5, 0.1

                xLabel2 = r'Time [yr]'

                ## This is the grid of plotting points
                nPointsGrid = 1001

                tPlot = np.linspace(self.tMin - self.kOrder * self.dtKnot, self.tMax + self.kOrder * self.dtKnot,
                                    nPointsGrid, endpoint=True)

                nSplines = self.tKnot.size + self.kOrder - 1
                tKnotSpl = np.linspace(self.tKnot[0] - self.kOrder * self.dtKnot, self.tKnot[-1] + self.kOrder * self.dtKnot,
                                       self.nKnots + 2 * self.kOrder, endpoint=True)

                xMin = min(xMin, self.tMin - self.kOrder * self.dtKnot - 0.25 * self.dtKnot)
                xMax = max(xMax, self.tMax + self.kOrder * self.dtKnot + 0.25 * self.dtKnot)

                nonZeroSplines = []
                BSplines = []

                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                knotColors = ['#b2df8a', '#a6cee3']

                fig1 = plt.figure(figsize=(xSize1, ySize1))

                ax1 = fig1.add_subplot(111)

                ## for knot in range(-kOrder, tT.size - 2 * kOrder - 1):
                for ii in range(nSplines):
                    b = interpolate.BSpline.basis_element(tKnotSpl[ii:ii + self.kOrder + 2], extrapolate=False)

                    BSpline = b(tPlot)

                    nonZero = BSpline > 0

                    color = colors[ii % len(colors)]

                    ax1.plot(tPlot[nonZero], BSpline[nonZero], '-', color=color)

                    nonZeroSplines.append(nonZero)
                    BSplines.append(BSpline)

                for ii in range(1, self.XtAll.shape[1]):
                    nonZero = self.XtAll[:, ii] > 0

                    color = colors[(ii - 1) % len(colors)]

                    ax1.plot(self.tObs[nonZero], self.XtAll[nonZero, ii], '.', color=color, rasterized=True)

                for ii in range(self.tKnot.size):
                    ax1.axvline(self.tKnot[ii], linewidth=1, linestyle='-', color=knotColors[0])

                ax1.set_xlim(xMin, xMax)
                ax1.xaxis.set_major_locator(AutoLocator())
                ax1.xaxis.set_minor_locator(AutoMinorLocator())

                ax1.set_ylim(yMin, yMax)
                ax1.yaxis.set_major_locator(MultipleLocator(dY))
                ax1.yaxis.set_minor_locator(MultipleLocator(dMY))

                ax1.set_xlabel(xLabel2);
                ax1.set_ylabel(r'$B_{i,k}(t)$');

                plotFilename1 = "{0:s}/plot_BSplines_vs_time_kOrder{1:d}.pdf".format(outDir, self.kOrder)

                fig1.savefig(plotFilename1, bbox_inches='tight')

                print("B-Spline plot saved to {0:s}".format(plotFilename1))

                plt.close(fig=fig1)

            self.nImages = self.nOkay

            print("N_PARS_K = {0:d} (K_ORDER = {1:d}, N_KNOTS = {2:d})".format(self.nParsK, self.kOrder, self.nKnots))

            P_A = self.nImages * self.nParsIndiv_A + self.nParsK * self.nParsSpline_A
            P_B = self.nImages * self.nParsIndiv_B + self.nParsK * self.nParsSpline_B

            print("P_A = {0:d}".format(P_A))
            print("P_B = {0:d}".format(P_B))
            print("N =", self.nDataAll)

            scalerArrayAll_A = np.zeros(P_A)
            scalerArrayAll_B = np.zeros(P_B)

            if (self.nParsIndiv_A > 0):
                scalerArrayAll_A[:self.nImages * self.nParsIndiv_A]  = np.tile(self.scalerArray[self.indivParsIndices_A], self.nImages)
            scalerArrayAll_A[self.nImages  * self.nParsIndiv_A:] = np.repeat(self.scalerArray[self.splineParsIndices_A], self.nParsK)

            if (self.nParsIndiv_B > 0):
                scalerArrayAll_B[:self.nImages * self.nParsIndiv_B]  = np.tile(self.scalerArray[self.indivParsIndices_B], self.nImages)
            scalerArrayAll_B[self.nImages  * self.nParsIndiv_B:] = np.repeat(self.scalerArray[self.splineParsIndices_B], self.nParsK)

            N_ITER_OUTER = 10
            N_ITER_INNER = 100

            markerSize = 0.1

            ## Plotting detected sources
            xSize1 = 12
            ySize1 = xSize1

            nRows = 2
            nCols = 1

            cMap = 'Greys'

            dX, dMX = 1000, 200
            dY, dMY = 500, 100

            xLabel, yLabel = r'$X$ [pix]', r'$Y$ [pix]'

            ## Plotting residuals
            xSize2 = 12
            ySize2 = 0.5 * xSize2

            nRows2 = 2
            nCols2 = 2

            retainedColor = 'k'
            nonFullColor = '#fc8d59'  ## Orange
            discardedColor = 'r'

            for jjj, chip in enumerate(chips):
                chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[jjj], acsconstants.CHIP_POSITIONS[jjj])

                indivDataFilename   = indivDataFilenames[jjj]
                modelCoeffsFilename = modelCoeffsFilenames[jjj]
                outTableFilename    = outTableFilenames[jjj]

                if (self.individualZP or (chip == 2)):
                    dxs   = [self.dxAll[jjj]]
                    dys   = [self.dyAll[jjj]]
                    rolls = [self.rollAll[jjj]]

                plateID   = self.plateIDAll[jjj]
                indices   = self.indicesAll[jjj]
                tObs      = self.tAll[jjj]
                rootnames = self.rootnamesAll[jjj]

                X_A = deepcopy(XAll_A[jjj])
                X_B = deepcopy(XAll_B[jjj])

                ## Initialize the reference coordinates
                xiRef  = deepcopy(self.xiAll[jjj])
                etaRef = deepcopy(self.etaAll[jjj])

                ## Initialize the raw coordinates
                xyRaw = self.xyRawAll[jjj]

                ## Initialize the weights using the match residuals
                ## weights = np.ones(X.shape[0])
                residuals = self.matchResAll[jjj]

                ## Use the weights to estimate the mean and covariance matrix of the residual
                ## distribution. Calculate the mean and covariance matrix for individual exposures,
                ## as they are time-dependent
                weights = np.zeros(residuals.shape[0])
                for i in range(self.nOkay):
                    j = self.okayIDs[i]

                    selection = plateID == j

                    nSelection = selection[selection].size

                    mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals[selection], np.ones(nSelection))

                    weights[selection] = stat.wdecay(stat.getMahalanobisDistances(residuals[selection], mean, np.linalg.inv(cov)))

                previousWeightSum = np.sum(weights)

                nIterTotal = 0

                plotFilename1 = "{0:s}/plot_time-dependent_model_chip{1:d}_pOrder{2:d}_kOrder{3:d}_nKnots{4:d}_residualDistribution.pdf".format(
                    outDir, chip, self.pOrder, self.kOrder, self.nKnots)

                plotFilename2 = "{0:s}/plot_time-dependent_model_chip{1:d}_pOrder{2:d}_kOrder{3:d}_nKnots{4:d}_residualsXY.pdf".format(
                    outDir, chip, self.pOrder, self.kOrder, self.nKnots)

                if makePlots:
                    pp1 = PdfPages(plotFilename1)

                    pp2 = PdfPages(plotFilename2)

                for iteration in range(N_ITER_OUTER):
                    print("OUTER_ITERATION {0:d}, AVERAGE SHIFTS AND ROLL:".format(iteration + 1),
                          np.nanmean(dxs[iteration]), np.nanmean(dys[iteration]), np.nanmean(rolls[iteration]))
                    if (self.individualZP or (chip == 2)):
                       xiRef, etaRef = self._shiftRotateReferenceCoordinates(xiRef, etaRef, plateID, dxs[iteration],
                                                                             dys[iteration], rolls[iteration])
                    else:
                        for (dx, dy, roll) in zip(dxs, dys, rolls):
                            xiRef, etaRef = self._shiftRotateReferenceCoordinates(xiRef, etaRef, plateID, dx, dy, roll)

                    for iteration2 in range(N_ITER_INNER):
                        startTime = time.time()

                        nIterTotal += 1

                        nStars = xiRef.size

                        W = sparse.spdiags([weights], 0)

                        A_A = X_A.T @ W @ X_A
                        A_B = X_B.T @ W @ X_B

                        b_xi  = (X_A.T @ W @ (xiRef / scalerX)).reshape((-1, 1))
                        b_eta = (X_B.T @ W @ (etaRef / scalerY)).reshape((-1, 1))

                        coeffsA, res, rnk, s = linalg.lstsq(A_A.todense(), b_xi, overwrite_a=True, overwrite_b=True)

                        coeffsA = coeffsA.flatten() * scalerX / scalerArrayAll_A

                        coeffsB, res, rnk, s = linalg.lstsq(A_B.todense(), b_eta, overwrite_a=True, overwrite_b=True)

                        coeffsB = coeffsB.flatten() * scalerY / scalerArrayAll_B

                        if ((((iteration2 + 1) % 10) == 0) or (iteration2 == 0)) and saveIntermediateResults:
                            coeffsAFilename = "{0:s}/coeffsA_chip{1:d}_pOrder{2:d}_kOrder{3:d}_nKnots{4:d}_iter1_{5:03d}_iter2_{6:03d}.npy".format(
                                outDir, chip, self.pOrder, self.kOrder, self.nKnots, iteration + 1, iteration2 + 1)

                            np.save(coeffsAFilename, coeffsA)

                            coeffsBFilename = "{0:s}/coeffsB_chip{1:d}_pOrder{2:d}_kOrder{3:d}_nKnots{4:d}_iter1_{5:03d}_iter2_{6:03d}.npy".format(
                                outDir, chip, self.pOrder, self.kOrder, self.nKnots, iteration + 1, iteration2 + 1)

                            np.save(coeffsBFilename, coeffsB)

                        ## Residuals already in pixel and in image axis
                        residualsXi  = xiRef  - ((X_A.multiply(scalerArrayAll_A)) @ coeffsA)
                        residualsEta = etaRef - ((X_B.multiply(scalerArrayAll_B)) @ coeffsB)

                        rmsXi  = np.sqrt(np.average(residualsXi ** 2,  weights=weights))
                        rmsEta = np.sqrt(np.average(residualsEta ** 2, weights=weights))

                        residuals = np.vstack([residualsXi, residualsEta]).T

                        ## Use the weights to estimate the mean and covariance matrix of the residual
                        ## distribution
                        for i in range(self.nOkay):
                            j = self.okayIDs[i]

                            selection = plateID == j

                            mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals[selection], weights[selection])

                            weights[selection] = stat.wdecay(stat.getMahalanobisDistances(residuals[selection],
                                                                                          mean, np.linalg.inv(cov)))

                        ## What we now call 'retained' are those stars with full weight
                        retained0 = weights >= 1.0

                        ## We have non-full weight stars but non-zero weights
                        nonFull = (~retained0) & (weights > 0)

                        ## Finally those stars with zero weights
                        rejected = weights <= 0

                        weightSum = np.sum(weights)

                        weightSumDiff = np.abs(weightSum - previousWeightSum) / weightSum

                        ## Assign the current weight summation for the next iteration
                        previousWeightSum = weightSum

                        elapsedTime = time.time() - startTime

                        if ((((iteration2 + 1) % 10) == 0) or (iteration2 == 0)):
                            print(chip, iteration + 1, iteration2 + 1,
                                  "N_STARS: {0:d}/{1:d}".format(xiRef[~rejected].size, (self.xiAll[jjj].size)),
                                  "RMS: {0:.6f} {1:.6f}".format(rmsXi, rmsEta), "W_SUM: {0:0.6f}".format(weightSum),
                                  "Elapsed time: {0}".format(convertTime(elapsedTime)))

                            if makePlots:
                                xSize = 8
                                ySize = xSize

                                fig = plt.figure(figsize=(xSize, ySize), rasterized=True)

                                ax = fig.add_subplot(111)

                                ax.plot(residuals[rejected][:, 0], residuals[rejected][:, 1], '.', markersize=markerSize,
                                        label=r'$w = 0$', color=discardedColor, rasterized=True)
                                ax.plot(residuals[nonFull][:, 0], residuals[nonFull][:, 1], '.', markersize=markerSize,
                                        label=r'$0 < w < 1$', color=nonFullColor, rasterized=True)
                                ax.plot(residuals[retained0][:, 0], residuals[retained0][:, 1], '.', markersize=markerSize,
                                        label=r'$w = 1$', color=retainedColor, rasterized=True)

                                ax.axhline()
                                ax.axvline()

                                xMin1, xMax1 = ax.get_xlim()
                                yMin1, yMax1 = ax.get_ylim()

                                maxRange = np.nanmax(np.abs(np.array([xMin1, xMax1, yMin1, yMax1])))

                                ax.set_xlim(-maxRange, +maxRange)
                                ax.set_ylim(-maxRange, +maxRange)

                                ax.set_aspect('equal')

                                ax.set_xlabel(r'$\Delta X$ [pix]')
                                ax.set_ylabel(r'$\Delta Y$ [pix]')

                                ax.xaxis.set_major_locator(AutoLocator())
                                ax.xaxis.set_minor_locator(AutoMinorLocator())

                                ax.yaxis.set_major_locator(AutoLocator())
                                ax.yaxis.set_minor_locator(AutoMinorLocator())

                                ax.legend(frameon=True)

                                ax.set_title(
                                    '{0:s}, $p$ = {1:d}, $k$ = {2:d}, $n_k$ = {3:d}, iter1 {4:d}, iter2 {5:d}'.format(
                                        chipTitle, self.pOrder, self.kOrder, self.nKnots, iteration + 1, iteration2 + 1))

                                pp1.savefig(fig)

                                plt.close(fig=fig)

                                fig2, axes2 = plt.subplots(figsize=(xSize2, ySize2), nrows=nRows2, ncols=nCols2,
                                                           rasterized=True)

                                xLabels = [r'$X_{\rm raw}$ [pix]', r'$Y_{\rm raw}$ [pix]']
                                yLabels = [r'$\Delta X$ [pix]', r'$\Delta Y$ [pix]']

                                XY0 = np.array([X0, Y0[0]])

                                xMin = np.array([0, 0])
                                xMax = np.array([4096, 2048])

                                yMin = +np.inf
                                yMax = -np.inf

                                for axis1 in range(NAXIS):
                                    for axis2 in range(NAXIS):
                                        coordinatesDiscarded = xyRaw[rejected][:, axis2]
                                        residualsDiscarded = residuals[rejected][:, axis1]

                                        coordinatesNonFull = xyRaw[nonFull][:, axis2]
                                        residualsNonFull = residuals[nonFull][:, axis1]

                                        coordinatesRetained = xyRaw[retained0][:, axis2]
                                        residualsRetained = residuals[retained0][:, axis1]

                                        mean = np.nanmean(residuals[retained0][:, axis1])
                                        stdDev = np.nanstd(residuals[retained0][:, axis1])

                                        axes2[axis1, axis2].plot(coordinatesDiscarded, residualsDiscarded, '.',
                                                                 markersize=markerSize, zorder=1, label=r'$w = 0$',
                                                                 color=discardedColor, rasterized=True)
                                        axes2[axis1, axis2].plot(coordinatesNonFull, residualsNonFull, '.',
                                                                 markersize=markerSize, zorder=1, label=r'$0 < w < 1$',
                                                                 color=nonFullColor, rasterized=True)
                                        axes2[axis1, axis2].plot(coordinatesRetained, residualsRetained, '.',
                                                                 markersize=markerSize, zorder=1, label=r'$w = 1$',
                                                                 color=retainedColor, rasterized=True)

                                        axes2[axis1, axis2].axhline(0,
                                                                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                                                                    linestyle='--', zorder=2)

                                        axes2[axis1, axis2].axhline(mean, color='r', linestyle='-', zorder=3)

                                        ## xMin, xMax = axes2[axis1,axis2].get_xlim()

                                        resMin = mean - stdDev
                                        resMax = mean + stdDev

                                        yMin = np.nanmin(np.array([yMin, mean - 5.0 * stdDev, np.nanmin(residuals)]))
                                        yMax = np.nanmax(np.array([yMax, mean + 5.0 * stdDev, np.nanmax(residuals)]))

                                        axes2[axis1, axis2].fill_between([xMin[axis2], xMax[axis2]], [resMin, resMin],
                                                                         y2=[resMax, resMax], alpha=0.2, zorder=3)

                                        axes2[axis1, axis2].set_xlabel(xLabels[axis2])
                                        axes2[axis1, axis2].set_ylabel(yLabels[axis1])

                                        axes2[axis1, axis2].xaxis.set_major_locator(MultipleLocator(dX))
                                        axes2[axis1, axis2].xaxis.set_minor_locator(MultipleLocator(dMX))

                                        axes2[axis1, axis2].yaxis.set_major_locator(AutoLocator())
                                        axes2[axis1, axis2].yaxis.set_minor_locator(AutoMinorLocator())

                                        axes2[axis1, axis2].set_xlim(xMin[axis2], xMax[axis2])

                                ## print("Y_MIN:", yMin, "Y_MAX:", yMax)

                                yMaxMin = np.nanmax(np.abs(np.array([yMin, yMax])))

                                for axis1 in range(NAXIS):
                                    for axis2 in range(NAXIS):
                                        axes2[axis1, axis2].set_ylim([-yMaxMin, +yMaxMin])

                                ## axes2[0,0].legend()

                                axCommons = plotting.drawCommonLabel('', '', fig2, xPad=0, yPad=0)

                                axCommons.set_title(
                                    '{0:s}, $p$ = {1:d}, $k$ = {2:d}, $n_k$ = {3:d}, iter1 {4:d}, iter2 {5:d}'.format(
                                        chipTitle, self.pOrder, self.kOrder, self.nKnots, iteration + 1,
                                        iteration2 + 1))

                                plt.subplots_adjust(wspace=0.25, hspace=0.3)

                                pp2.savefig(fig2, bbox_inches='tight', dpi=300)

                                plt.close(fig=fig2)

                        gc.set_threshold(2, 1, 1)

                        del residualsXi
                        del residualsEta
                        del residuals
                        gc.collect()

                        ## At the last iteration, re-calculate the shift and rolls
                        ## if ((iteration2+1) == (N_ITER_INNER)):
                        if ((weightSumDiff < 1.e-12) or (iteration2 + 1) == (N_ITER_INNER)):
                            ## Find the shift and rotation of the reference coordinates
                            ## using the new zero-th order coefficients and rotation angle
                            end_A = self.nImages * self.nParsIndiv_A
                            end_B = self.nImages * self.nParsIndiv_B

                            dxs.append(coeffsA[0:end_A:self.nParsIndiv_A])
                            dys.append(coeffsB[0:end_B:self.nParsIndiv_B])

                            if (self.pOrderIndiv == 0):
                                coeffsA3 = coeffsA[1:end_A:self.nParsIndiv_A]

                                start = self.nImages * self.nParsIndiv_B + self.nParsK
                                end   = start + self.nParsK

                                coeffsB3 = self.XtAll @ coeffsB[start:end]
                            else:
                                thisP = 2

                                coeffsA3 = coeffsA[thisP:end_A:self.nParsIndiv_A]
                                coeffsB3 = coeffsB[thisP:end_B:self.nParsIndiv_B]

                            rolls.append(-np.arctan(coeffsA3 / coeffsB3))

                            break
                        else:
                            X_A = X_A[~rejected]
                            X_B = X_B[~rejected]

                            weights = weights[~rejected]

                            xiRef = xiRef[~rejected]
                            etaRef = etaRef[~rejected]

                            xyRaw = xyRaw[~rejected]

                            plateID   = plateID[~rejected]
                            indices   = indices[~rejected]
                            tObs      = tObs[~rejected]
                            rootnames = rootnames[~rejected]

                    if (not (self.individualZP or (chip == 2))):
                        break

                if makePlots:
                    pp1.close()

                    print("Residual 2d distribution plots saved to {0:s}".format(plotFilename1))

                    pp2.close()

                    print("Residual XY-distribution plots saved to {0:s}".format(plotFilename2))

                self._parseIndividualAndSplineCoefficientsDataFrame(chip, coeffsA, coeffsB,
                                                                   modelCoeffsFilename=modelCoeffsFilename,
                                                                   indivDataFilename=indivDataFilename)

                xiPred  = (X_A.multiply(scalerArrayAll_A)) @ coeffsA
                etaPred = (X_B.multiply(scalerArrayAll_B)) @ coeffsB

                residualsXi  = xiRef  - xiPred
                residualsEta = etaRef - etaPred

                outTable = QTable(
                    [xyRaw[:, 0], xyRaw[:, 1], xiPred, etaPred, xiRef, etaRef, residualsXi, residualsEta, weights, plateID,
                     indices, tObs, rootnames, np.full_like(rootnames, chip)], names=(
                    'X', 'Y', 'xPred', 'yPred', 'xRef', 'yRef', 'dx', 'dy', 'weights', 'plateID', 'indices', 'tObs', 'rootname', 'chip'))

                outTable.write(outTableFilename, overwrite=True)

                print("Residual table written to {0:s}".format(outTableFilename))

        print("APPLYING TIME-DEPENDENT COEFFICIENTS TO SELECTED HST1PASS FILES...")
        fitResultsFilename = '{0:s}/fitResults_pOrder{1:d}_kOrder{2:d}.txt'.format(outDir, self.pOrder, self.kOrder)

        if (not os.path.exists(fitResultsFilename)):
            ## Read the output table from the time-dependent coefficient fitting
            resids = table.vstack([ascii.read(outTableFilenames[0]),
                                   ascii.read(outTableFilenames[1])])

            ## Read individual SIP coefficients and spline coefficients
            df_indiv_data = pd.concat([pd.read_csv(indivDataFilename) for indivDataFilename in indivDataFilenames],
                                      ignore_index=True)
            df_spline_coeffs = pd.concat(
                [pd.read_csv(modelCoeffsFilename) for modelCoeffsFilename in modelCoeffsFilenames], ignore_index=True)

            fitResultsText = []

            for i, (hst1passFile, imageFilename) in enumerate(zip(hst1passFiles, imageFilenames)):
                addendumFilename = hst1passFile.replace('.csv', '_addendum.csv')

                baseImageFilename = os.path.basename(hst1passFile).replace('_hst1pass_stand.csv', '')

                if (os.path.exists(imageFilename)) and (os.path.exists(addendumFilename)):
                    hduList = fits.open(imageFilename)

                    tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
                    t_acs   = Time(tstring, scale='utc', format='fits')

                    if (self.tMin <= t_acs.decimalyear <= self.tMax):
                        print(i, hst1passFile)

                        rootname = hduList[0].header['ROOTNAME']

                        dt = t_acs.decimalyear - self.tRef0.utc.value

                        tExp = float(hduList[0].header['EXPTIME'])

                        pa_v3 = float(hduList[0].header['PA_V3'])

                        posTarg1 = float(hduList[0].header['POSTARG1'])
                        posTarg2 = float(hduList[0].header['POSTARG2'])

                        Xt = bspline.getForwardModelBSpline(t_acs.decimalyear, self.kOrder, self.tKnot)

                        ## We use the observation time, in combination with the proper motions to move
                        ## the coordinates into the time
                        self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
                        self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

                        hst1pass = table.hstack([ascii.read(hst1passFile), ascii.read(addendumFilename)])

                        ## Add new columns and assign default values
                        hst1pass['dx']       = np.nan
                        hst1pass['dy']       = np.nan
                        hst1pass['retained'] = False
                        hst1pass['weights']  = 0.0  ## Final weights for all detected sources in the chip
                        hst1pass['xi']       = np.nan
                        hst1pass['eta']      = np.nan
                        hst1pass['xiRef']    = np.nan
                        hst1pass['etaRef']   = np.nan
                        hst1pass['resXi']    = np.nan
                        hst1pass['resEta']   = np.nan
                        hst1pass['alpha']    = np.nan
                        hst1pass['delta']    = np.nan
                        hst1pass['sourceID'] = np.zeros(len(hst1pass), dtype='<U24')

                        resids_selection = resids['plateID'] == i
                        resids_indices   = resids['indices'][resids_selection].value

                        if (resids_indices.size > 0):
                            ## This existing columns can be replaced with nans because we're going to use the
                            ## time-dependent model to calculate them, and we'll use the previously calculated
                            ## values
                            hst1pass['xPred'] = np.nan
                            hst1pass['yPred'] = np.nan
                            hst1pass['xRef']  = np.nan
                            hst1pass['yRef']  = np.nan

                            hst1pass['xRef'][resids_indices]     = resids['xRef'][resids_selection]
                            hst1pass['yRef'][resids_indices]     = resids['yRef'][resids_selection]
                            hst1pass['retained'][resids_indices] = True
                            hst1pass['weights'][resids_indices]  = resids['weights'][resids_selection]
                        else:
                            selection = hst1pass['refCatIndex'] >= 0

                            if (selection[selection].size > 0):
                                hst1pass['retained'][selection] = True

                                hst1pass['dx'][selection] = hst1pass['xRef'][selection] - hst1pass['xPred'][selection]
                                hst1pass['dy'][selection] = hst1pass['yRef'][selection] - hst1pass['yPred'][selection]

                                ## calculate the initial values of the weights
                                residuals = np.vstack([hst1pass['dx'][selection].value,
                                                       hst1pass['dy'][selection].value]).T

                                ## Use the weights to estimate the mean and covariance matrix of the residual distribution
                                mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals,
                                                                                       np.ones(residuals.shape[0]))

                                ## Assign weights based on the standardized distance of the residuals from the mean
                                hst1pass['weights'][selection] = stat.wdecay(
                                    stat.getMahalanobisDistances(residuals, mean, np.linalg.inv(cov)))

                        textResults = ""
                        for jjj, (chip, ver) in enumerate(zip(chips, headers)):
                            startTime = time.time()

                            jj = acsconstants.N_CHIPS - chip

                            chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[jjj], acsconstants.CHIP_POSITIONS[jjj])

                            hdu = hduList['SCI', ver]

                            ## Zero point of the y coordinates.
                            if (ver == 1):
                                yzp    = 0.0
                                naxis2 = int(hdu.header['NAXIS2'])
                            else:
                                yzp += float(naxis2)

                                naxis2 = int(hdu.header['NAXIS2'])

                            orientat = Angle(float(hdu.header['ORIENTAT']) * u.deg).wrap_at('360d').value
                            vaFactor = float(hdu.header['VAFACTOR'])

                            ## Now that we have the coefficients, we repeat the model building for ALL objects in the chip
                            selection = (hst1pass['k'] == chip)

                            hasRefStar = (hst1pass['refCatIndex'] >= 0)

                            refStarIdx = hst1pass[selection]['refCatIndex'].value

                            xiRef  = self.refCat[refStarIdx]['xt'] / vaFactor
                            etaRef = self.refCat[refStarIdx]['yt'] / vaFactor

                            nStars = xiRef.size

                            XC = hst1pass['X'][selection] - X0
                            YC = hst1pass['Y'][selection] - Y0[chip - 1]

                            if self.make_lithographic_and_filter_mask_corrections:
                                dcorr = np.array(
                                    litho.interp_dtab_ftab_data(self.dtabs[jjj], hst1pass['X'][selection].value,
                                                                hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T
                                fcorr = np.array(
                                    litho.interp_dtab_ftab_data(self.ftabs[jjj], hst1pass['X'][selection].value,
                                                                hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                                ## Apply the lithographic mask correction
                                XC -= (dcorr[:, 2] - fcorr[:, 2])
                                YC -= (dcorr[:, 3] - fcorr[:, 3])

                            X, scalerArray = sip.buildModel(XC, YC, self.pOrder, scalerX=scalerX, scalerY=scalerY)

                            thisCoeffsA = np.zeros((self.nParsSIP, 1), dtype=float)
                            thisCoeffsB = np.zeros_like(thisCoeffsA)

                            for p in self.indivParsIndices_A:
                                thisCoeffsA[p, 0] = self._getIndividualCoeffs(rootname, p, 0, chip, df_indiv_data)

                            for p in self.indivParsIndices_B:
                                thisCoeffsB[p, 0] = self._getIndividualCoeffs(rootname, p, 1, chip, df_indiv_data)

                            for p in self.splineParsIndices_A:
                                thisCoeffsA[p, 0] = Xt @ self._getSplineCoeffs(p, 0, chip, df_spline_coeffs)

                            for p in self.splineParsIndices_B:
                                thisCoeffsB[p, 0] = Xt @ self._getSplineCoeffs(p, 1, chip, df_spline_coeffs)

                            xPred = np.matmul(X * scalerArray, thisCoeffsA).flatten()
                            yPred = np.matmul(X * scalerArray, thisCoeffsB).flatten()

                            hst1pass['xPred'][selection] = xPred
                            hst1pass['yPred'][selection] = yPred
                            hst1pass['dx'][selection]    = hst1pass['xRef'][selection] - xPred
                            hst1pass['dy'][selection]    = hst1pass['yRef'][selection] - yPred

                            alpha0Im = float(hdu.header['CRVAL1'])
                            delta0Im = float(hdu.header['CRVAL2'])

                            ## We calculate the CD Matrix used to transform the intermediate world coordinate
                            ## We take the reference coordinate to be the CRVAL1,2 in the header and a create a SkyCoord object
                            c0Im = SkyCoord(ra=alpha0Im * u.deg, dec=delta0Im * u.deg, frame='icrs')

                            ## Now we take the normal triad pqr_0 of the reference coordinate
                            pqr0Im = coords.getNormalTriad(c0Im)

                            selection  = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & hst1pass['retained']
                            refStarIdx = hst1pass['refCatIndex'][selection].value

                            hst1pass['xiRef'][selection]  = self.refCat[refStarIdx]['xt'] / vaFactor
                            hst1pass['etaRef'][selection] = self.refCat[refStarIdx]['yt'] / vaFactor

                            CDMatrix = np.full((2, 3), np.nan)

                            if (selection[selection].size > 10):
                                for iteration3 in range(N_ITER_CD):
                                    CDMatrix = self._getCDMatrix(hst1pass['xPred'][selection].value,
                                                                 hst1pass['yPred'][selection].value,
                                                                 hst1pass['xiRef'][selection].value,
                                                                 hst1pass['etaRef'][selection].value,
                                                                 weights=hst1pass['weights'][selection].value)

                                    selectionChip = hst1pass['k'] == chip

                                    H, _ = sip.buildModel(hst1pass['xPred'][selectionChip].value,
                                                          hst1pass['yPred'][selectionChip].value,
                                                          1)

                                    ## Calculate the normal coordinates Xi, Eta and assign them to the table
                                    hst1pass['xi'][selectionChip]  = H @ CDMatrix[0]
                                    hst1pass['eta'][selectionChip] = H @ CDMatrix[1]

                                    '''
                                    ## Calculate the normal coordinates relative to the pqr triad centered on the current CRVAL1,2
                                    self.refCat = self._getNormalCoordinates(self.refCat, 'xt', 'yt', self.wcsRef, pqr0Im)

                                    ## The reference coordinates used for regression of the CD matrix is relative to the current
                                    ## CRVAL1, CRVAL2 coordinates
                                    xiRef  = (self.refCat['xi'][refStarIdx].value  * u.arcsec).to(u.deg) / vaFactor
                                    etaRef = (self.refCat['eta'][refStarIdx].value * u.arcsec).to(u.deg) / vaFactor

                                    ## Store the reference coordinates back in pixel scale
                                    hst1pass['xiRef'][selection] = xiRef.to_value(
                                        u.arcsec) / acsconstants.ACS_PLATESCALE.to_value(
                                        u.arcsec / u.pix)
                                    hst1pass['etaRef'][selection] = etaRef.to_value(
                                        u.arcsec) / acsconstants.ACS_PLATESCALE.to_value(
                                        u.arcsec / u.pix)

                                    ## Calculate the CD Matrix that transform the corrected coordinates
                                    ## XCorr, YCorr into normal coordinates Xi, Eta
                                    CDMatrix = self._getCDMatrix(XCorr, YCorr, xiRef.value, etaRef.value,
                                                                 weights=hst1pass['weights'][selection].value)

                                    ## Replace the CRVAL1, CRVAL2 coordinates using the constants of the CD Matrix
                                    c0Im = coords.getCelestialCoordinatesFromNormalCoordinates(CDMatrix[0, 0] * u.deg,
                                                                                               CDMatrix[1, 0] * u.deg, c0Im,
                                                                                               frame='icrs')

                                    ## Replace the triad pqr_0 using the new value of CRVAL1, CRVAL2
                                    pqr0Im = coords.getNormalTriad(c0Im)

                                    ## Apply the CD Matrix to all sources in the chip, in order to obtain the new normal coordinates
                                    selectionChip = hst1pass['k'] == chip

                                    H, _ = sip.buildModel(hst1pass['xPred'][selectionChip].value,
                                                          hst1pass['yPred'][selectionChip].value,
                                                          1)

                                    ## Calculate the normal coordinates Xi, Eta and assign them to the table
                                    hst1pass['xi'][selectionChip] = (
                                            ((H @ CDMatrix[0]) * u.deg) / acsconstants.ACS_PLATESCALE).to_value(u.pix)
                                    hst1pass['eta'][selectionChip] = (
                                            ((H @ CDMatrix[1]) * u.deg) / acsconstants.ACS_PLATESCALE).to_value(u.pix)
                                    ''';

                                    ## Calculate the residuals
                                    hst1pass['resXi'][selection] = (
                                            hst1pass['xi'][selection].value - hst1pass['xiRef'][selection].value)
                                    hst1pass['resEta'][selection] = (
                                            hst1pass['eta'][selection].value - hst1pass['etaRef'][selection].value)

                                    ## print("CD_ITERATION:", (iteration3 + 1))
                                    ## print(CDMatrix)
                                    ## print(hst1pass['resXi', 'resEta'][selection].to_pandas().describe())

                                    ## Re-calculate the weights for the next iteration, in case it's not a plate used for
                                    ## time-dependendent model calculation
                                    if (resids_indices.size <= 0):
                                        residuals = np.vstack([hst1pass[selection]['resXi'].value,
                                                               hst1pass[selection]['resEta'].value]).T

                                        ## Use the weights to estimate the mean and covariance matrix of the residual distribution
                                        mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals,
                                                                                               hst1pass[selection]['weights'].value)

                                        ## Assign weights based on the standardized distance of the residuals from the mean
                                        hst1pass[selection]['weights'] = stat.wdecay(stat.getMahalanobisDistances(residuals,
                                                                                                                  mean,
                                                                                                                  np.linalg.inv(
                                                                                                                      cov)))

                                alpha0Im, delta0Im = c0Im.ra.value, c0Im.dec.value

                            '''
                            xi0, eta0 = self.wcsRef.wcs_world2pix(np.array([alpha0Im]), np.array([delta0Im]), 1)

                            xi0  = float(self.wcsRef.to_header()['CRPIX1']) - xi0[0]
                            eta0 = eta0[0] - float(self.wcsRef.to_header()['CRPIX2'])

                            selection = (hst1pass['k'] == chip)

                            hst1pass['xi'][selection]  = hst1pass['xi'][selection]  + xi0
                            hst1pass['eta'][selection] = hst1pass['eta'][selection] + eta0

                            hst1pass['xiRef'][selection]  = hst1pass['xiRef'][selection]  + xi0
                            hst1pass['etaRef'][selection] = hst1pass['etaRef'][selection] + eta0

                            ## Calculate the residuals
                            hst1pass['resXi'][selection] = (
                                    hst1pass['xi'][selection].value - hst1pass['xiRef'][selection].value)
                            hst1pass['resEta'][selection] = (
                                    hst1pass['eta'][selection].value - hst1pass['etaRef'][selection].value)
                            ''';

                            rmsXi  = np.nan
                            rmsEta = np.nan

                            residual_selection = (np.isfinite(hst1pass['resXi'][selection].value) &
                                                  np.isfinite(hst1pass['resEta'][selection].value) &
                                                  np.isfinite(hst1pass['weights'][selection].value))

                            if (np.sum(hst1pass['weights'][selection].value) > 0) and (residual_selection[residual_selection].size > 0):
                                rmsXi  = np.sqrt(stat.getWeightedAverage(hst1pass['resXi'][selection].value ** 2,
                                                                         hst1pass['weights'][selection].value))
                                rmsEta = np.sqrt(stat.getWeightedAverage(hst1pass['resEta'][selection].value ** 2,
                                                                         hst1pass['weights'][selection].value))

                            textResults += "{0:s} {1:d} {2:.8f} {3:.6f} {4:.13f} {5:.12e} {6:0.2f} {7:f} {8:f}".format(
                                baseImageFilename, chip, t_acs.decimalyear, pa_v3, orientat, vaFactor, tExp, posTarg1,
                                posTarg2)
                            textResults += " {0:d} {1:d}".format(0, nStars)
                            textResults += " {0:0.6f} {1:0.6f}".format(rmsXi, rmsEta)
                            textResults += " {0:0.12f} {1:0.12f}".format(alpha0Im, delta0Im)
                            textResults += " {0:0.12e} {1:0.12e} {2:0.12e} {3:0.12e}".format(CDMatrix[0, 1], CDMatrix[0, 2],
                                                                                             CDMatrix[1, 1], CDMatrix[1, 2])

                            for coeffA, coeffB in zip(thisCoeffsA.flatten(), thisCoeffsB.flatten()):
                                textResults += " {0:0.12e}".format(coeffA)
                                textResults += " {0:0.12e}".format(coeffB)
                            textResults += "\n"

                            ## print("RESIDUALS FOR {0:s}".format(chipTitle))
                            ## print(hst1pass.to_pandas().loc[selection, ['resXi', 'resEta']].describe())

                            elapsedTime = time.time() - startTime
                            print("FITTING DONE FOR {0:s},".format(chipTitle), "elapsed time:", convertTime(elapsedTime), end='. ')

                        fitResultsText.append(textResults)

                        ## Assign name for each sources in each chip. We first grab the xi, eta from the catalogue.
                        '''
                        xi  = (hst1pass['xi'] * u.pix)  * acsconstants.ACS_PLATESCALE
                        eta = (hst1pass['eta'] * u.pix) * acsconstants.ACS_PLATESCALE

                        ## Use the zero-point of the reference catalogue and declare a SkyCoord object from zero-point.
                        c0 = SkyCoord(ra=self.alpha0, dec=self.delta0, frame='icrs')

                        ## Find only sources with defined xi and eta. Don't worry if they're crap sources, we'll deal with them
                        ## later in the next phase
                        argsel = np.argwhere(~np.isnan(xi) & ~np.isnan(eta)).flatten()

                        ## Calculate the equatorial coordinates and assign them to the table
                        c = coords.getCelestialCoordinatesFromNormalCoordinates(xi[argsel], eta[argsel], c0, frame='icrs')
                        ''';
                        argsel = np.argwhere(~np.isnan(hst1pass['xi']) & ~np.isnan(hst1pass['eta'])).flatten()

                        hst1pass['alpha'][argsel], hst1pass['delta'][argsel] = self.wcsRef.wcs_pix2world(
                            hst1pass['xi'][argsel], hst1pass['eta'][argsel], 1)

                        hst1pass['sourceID'][argsel] = astro.generateSourceID(
                            SkyCoord(ra=hst1pass['alpha'][argsel] * u.deg, dec=hst1pass['delta'][argsel] * u.deg,
                                     frame='icrs'))

                        ## hst1pass['alpha'][argsel] = c.ra.value
                        ## hst1pass['delta'][argsel] = c.dec.value

                        ## Based on the equatorial coordinates assign a source ID for each source
                        ## hst1pass['sourceID'][argsel] = astro.generateSourceID(c)

                        ## Now we query the Gaia catalogue, if cross_match is set to True. For this we will have different
                        ## criteria than before. We now only cross-match sources with 0 < q <= Q_MAX, for these are more likely
                        ## to be bona-fide point-sources (i.e. stars).
                        if self.cross_match:
                            xi  = (hst1pass['xi'] * u.pix) * acsconstants.ACS_PLATESCALE
                            eta = (hst1pass['eta'] * u.pix) * acsconstants.ACS_PLATESCALE

                            argsel = np.argwhere(~np.isnan(xi) & ~np.isnan(eta) & (hst1pass['q'] > Q_MIN) & (
                                        hst1pass['q'] <= Q_MAX)).flatten()

                            c = SkyCoord(ra=hst1pass['alpha'][argsel] * u.deg, dec=hst1pass['delta'][argsel] * u.deg,
                                         frame='icrs')

                            self.c_gdr3 = self.c_gdr3.apply_space_motion(t_acs)

                            idx, sep, _ = c.match_to_catalog_sky(self.c_gdr3)

                            sep_pix = sep.to(u.mas) / acsconstants.ACS_PLATESCALE

                            selection_gdr3 = sep_pix < MAX_SEP

                            ## We now assign a different source ID for sources with known GDR3 stars counterpart
                            hst1pass['sourceID'][argsel[selection_gdr3]] = self.gdr3_id[idx[selection_gdr3]]

                        ## Write the final table
                        outTableFilename = '{0:s}/{1:s}_hst1pass_stand_pOrder{2:d}_kOrder{3:d}_resids.csv'.format(
                            outDir, baseImageFilename, self.pOrder, self.kOrder)

                        hst1pass.write(outTableFilename, overwrite=True)

                        print("Final table written to", outTableFilename)

                        ## Plot the coordinates and their residuals on a common reference frame
                        if makePlots:
                            xSize3 = 12
                            ySize3 = 1.0075 * xSize3

                            xMin3, xMax3 = -5500, +5500
                            yMin3, yMax3 = xMin3, xMax3

                            dX3, dMX3 = 2000, 500
                            dY3, dMY3 = dX3, dMX3

                            resMin = -0.29
                            resMax = +0.29
                            dRes = 0.2
                            dMRes = 0.05

                            markerSize3 = 8

                            fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(xSize3, ySize3), sharex='col', sharey='row',
                                                     width_ratios=[1.0, 0.25, 0.25], height_ratios=[0.25, 0.25, 1.0])

                            ax3[0, 0].set_title(
                                hduList[0].header['ROOTNAME'] + ' --- ' + hduList[0].header['DATE-OBS'] + ' UT' +
                                hduList[0].header['TIME-OBS'])

                            ax3[0, 1].set_visible(False)
                            ax3[0, 2].set_visible(False)
                            ax3[1, 2].set_visible(False)

                            ## print("FINAL RESIDUALS (COMBINED):")
                            df_resids = hst1pass.to_pandas()

                            xi0  = float(self.wcsRef.to_header()['CRPIX1'])
                            eta0 = float(self.wcsRef.to_header()['CRPIX2'])

                            df_resids['xi']  = xi0 - df_resids['xi']
                            df_resids['eta'] = df_resids['eta'] - eta0

                            selection = (df_resids['refCatIndex'] >= 0) & df_resids['retained']

                            ## print(df_resids.loc[selection, ['resXi', 'resEta']].describe())

                            sns.scatterplot(data=df_resids[selection], x='resXi', y='resEta', hue='weights', legend=False,
                                            ax=ax3[1, 1],
                                            s=markerSize3, rasterized=True)

                            sns.scatterplot(data=df_resids[selection], x='xi', y='resXi', hue='weights', legend=False,
                                            ax=ax3[0, 0],
                                            s=markerSize3, rasterized=True)
                            sns.scatterplot(data=df_resids[selection], x='xi', y='resEta', hue='weights', legend=False,
                                            ax=ax3[1, 0],
                                            s=markerSize3, rasterized=True)

                            sns.scatterplot(data=df_resids[selection], x='xi', y='eta', hue='weights', legend=True,
                                            ax=ax3[2, 0],
                                            s=markerSize3, rasterized=True)

                            sns.scatterplot(data=df_resids[selection], x='resXi', y='eta', hue='weights', legend=False,
                                            ax=ax3[2, 1],
                                            s=markerSize3, rasterized=True)
                            sns.scatterplot(data=df_resids[selection], x='resEta', y='eta', hue='weights', legend=False,
                                            ax=ax3[2, 2],
                                            s=markerSize3, rasterized=True)

                            resLabels = ['res_xi [pix]', 'res_eta [pix]']

                            for axis in range(NAXIS):
                                ax3[2, axis + 1].set_xlabel(resLabels[axis])
                                ax3[axis, 0].set_ylabel(resLabels[axis])

                                ax3[2, axis + 1].xaxis.set_major_locator(MultipleLocator(dRes))
                                ax3[2, axis + 1].xaxis.set_minor_locator(MultipleLocator(dMRes))

                                ax3[2, axis + 1].yaxis.set_major_locator(MultipleLocator(dY3))
                                ax3[2, axis + 1].yaxis.set_minor_locator(MultipleLocator(dMY3))

                                ax3[axis, 0].xaxis.set_major_locator(MultipleLocator(dX3))
                                ax3[axis, 0].xaxis.set_minor_locator(MultipleLocator(dMX3))

                                ax3[axis, 0].yaxis.set_major_locator(MultipleLocator(dRes))
                                ax3[axis, 0].yaxis.set_minor_locator(MultipleLocator(dMRes))

                                ax3[2, axis + 1].set_xlim(resMin, resMax)
                                ax3[axis, 0].set_ylim(resMin, resMax)

                                ax3[axis, 0].axhline(y=0, linewidth=1)
                                ax3[2, axis + 1].axvline(x=0, linewidth=1)

                            ax3[1, 1].xaxis.set_major_locator(MultipleLocator(dRes))
                            ax3[1, 1].xaxis.set_minor_locator(MultipleLocator(dMRes))
                            ax3[1, 1].yaxis.set_major_locator(MultipleLocator(dRes))
                            ax3[1, 1].yaxis.set_minor_locator(MultipleLocator(dMRes))

                            ax3[1, 1].axvline(x=0)
                            ax3[1, 1].axhline(y=0)

                            ax3[2, 0].axhline(linestyle='--', color='r', y=0, linewidth=1)
                            ax3[2, 0].axvline(linestyle='--', color='r', x=0, linewidth=1)

                            ax3[2, 0].set_xlabel(r'xi [pix]')
                            ax3[2, 0].set_ylabel(r'eta [pix]')

                            for iii in range(3):
                                ax3[iii, 0].xaxis.set_major_locator(MultipleLocator(dX3))
                                ax3[iii, 0].xaxis.set_minor_locator(MultipleLocator(dMX3))

                                ax3[2, iii].yaxis.set_major_locator(MultipleLocator(dY3))
                                ax3[2, iii].yaxis.set_minor_locator(MultipleLocator(dMY3))

                            ax3[2, 0].set_xlim(xMin3, xMax3)
                            ax3[2, 0].set_ylim(yMin3, yMax3)

                            ax3[2, 0].set_aspect('equal')

                            ax3[2, 0].invert_xaxis()

                            plt.subplots_adjust(wspace=0.0, hspace=0.0)

                            plotFilename3 = "{0:s}/plot_{1:s}_pOrder{2:d}_kOrder{3:d}_retainedSources_commonCoordinates.pdf".format(outDir,
                                                                                                                        baseImageFilename,
                                                                                                                        self.pOrder,
                                                                                                                        self.kOrder)

                            fig3.savefig(plotFilename3, dpi=300, bbox_inches='tight')

                            plt.close(fig=fig3)

                            print()

            f = open(fitResultsFilename, 'w')
            for textResult in fitResultsText:
                if textResult is not None:
                    f.write(textResult)
            f.close()
            print("Fit results written to", fitResultsFilename)

        elapsedTime = time.time() - startTimeAll
        print("ALL DONE! Elapsed time:", convertTime(elapsedTime))

    def _processFile(self, i, hst1passFile, imageFilename):
        addendumFilename = hst1passFile.replace('.csv', '_addendum.csv')

        baseImageFilename = os.path.basename(hst1passFile).replace('_hst1pass_stand.csv', '')

        rootname = baseImageFilename.split('_')[0]

        if (os.path.exists(imageFilename)) and (os.path.exists(addendumFilename)):
            hduList = fits.open(imageFilename)

            tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
            t_acs = Time(tstring, scale='utc', format='fits')

            tExp = float(hduList[0].header['EXPTIME'])

            posTarg1 = float(hduList[0].header['POSTARG1'])
            posTarg2 = float(hduList[0].header['POSTARG2'])

            posTargResultant = np.sqrt(posTarg1 ** 2 + posTarg2 ** 2)

            if ((t_acs.decimalyear >= self.tMin) and (t_acs.decimalyear <= self.tMax) and (tExp > self.min_t_exp)
                    and (posTargResultant <= self.max_pos_targs)):
                pa_v3 = float(hduList[0].header['PA_V3'])

                dt = t_acs.decimalyear - self.tRef0.utc.value

                ## We use the observation time, in combination with the proper motions to move
                ## the coordinates into the time
                self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
                self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

                hst1pass = table.hstack([ascii.read(hst1passFile), ascii.read(addendumFilename)])

                delX = hst1pass['xPred'] - hst1pass['xRef']
                delY = hst1pass['yPred'] - hst1pass['yRef']

                matchRes = np.sqrt(delX ** 2 + delY ** 2)

                okays     = np.zeros(chips.size, dtype='bool')
                nGoodData = np.zeros(chips.size, dtype=int)
                for chip in chips:
                    jjj = chip - 1

                    selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                            hst1pass['q'] <= Q_MAX) & (~np.isnan(hst1pass['nAppearances'])) & (
                                        hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                        matchRes <= self.max_pix_tol)

                    nGoodData[jjj] = len(hst1pass[selection])

                    if (nGoodData[jjj] > self.min_n_refstar):
                        okays[jjj] = True

                    '''
                    print(acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1], acsconstants.CHIP_POSITIONS[chip - 1]),
                          "N_STARS =", nGoodData[jjj], "OKAY:", okays[jjj])
                    ''';

                del selection
                gc.collect()

                okayToProceed = np.prod(okays, dtype='bool')

                ## print("OKAY TO PROCEED:", okayToProceed)

                Xt = bspline.getForwardModelBSpline(t_acs.decimalyear, self.kOrder, self.tKnot)

                if okayToProceed:
                    print(rootname, end=' ')
                    self.selectedHST1PassFiles.append(hst1passFile)
                    self.selectedImageFiles.append(imageFilename)

                    self.nOkay += 1

                    self.okayIDs.append(i)

                    self.rootnames.append(rootname)
                    self.XtAll.append(Xt)
                    self.tObs.append(t_acs.decimalyear)

                    for chip, ver in zip(chips, headers):
                        jj  = chips.size - chip
                        jjj = chip - 1

                        chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[jjj], acsconstants.CHIP_POSITIONS[jjj])

                        hdu = hduList['SCI', ver]

                        ## Zero point of the y coordinates.
                        naxis2 = int(hdu.header['NAXIS2'])
                        if (ver == 1):
                            yzp = 0.0
                        else:
                            yzp += float(naxis2)

                            naxis2 = int(hdu.header['NAXIS2'])

                        orientat = Angle(float(hdu.header['ORIENTAT']) * u.deg).wrap_at('360d').value
                        vaFactor = float(hdu.header['VAFACTOR'])

                        selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (
                                hst1pass['q'] > 0) & (hst1pass['q'] <= Q_MAX) & (
                                        ~np.isnan(hst1pass['nAppearances'])) & (
                                            hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                            matchRes <= self.max_pix_tol)

                        refStarIdx = hst1pass[selection]['refCatIndex'].value

                        nData = len(refStarIdx)

                        ## print("CHIP:", chipTitle, "N_STARS:", nData)

                        self.nDataAll[jjj] += nData

                        xi = self.refCat[refStarIdx]['xt'] / vaFactor
                        eta = self.refCat[refStarIdx]['yt'] / vaFactor

                        XC = hst1pass['X'][selection] - X0
                        YC = hst1pass['Y'][selection] - Y0[chip - 1]

                        if self.make_lithographic_and_filter_mask_corrections:
                            dcorr = np.array(litho.interp_dtab_ftab_data(
                                self.dtabs[jjj],
                                hst1pass['X'][selection].value,
                                hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                            fcorr = np.array(litho.interp_dtab_ftab_data(
                                self.ftabs[jjj],
                                hst1pass['X'][selection].value,
                                hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                            ## Apply the lithographic and filter mask correction
                            XC -= (dcorr[:, 2] - fcorr[:, 2])
                            YC -= (dcorr[:, 3] - fcorr[:, 3])

                            del dcorr
                            del fcorr

                        Xp, self.scalerArray = sip.buildModel(XC, YC, self.pOrder, scalerX=scalerX, scalerY=scalerY)

                        Xkp_A = np.zeros((Xp.shape[0], self.nParsK * self.nParsSpline_A))
                        Xkp_B = np.zeros((Xp.shape[0], self.nParsK * self.nParsSpline_B))

                        for ii, p in enumerate(self.splineParsIndices_A):
                            for k in range(self.nParsK):
                                Xkp_A[:, ii * self.nParsK + k] = Xt[0, k] * Xp[:, p]

                        for ii, p in enumerate(self.splineParsIndices_B):
                            for k in range(self.nParsK):
                                Xkp_B[:, ii * self.nParsK + k] = Xt[0, k] * Xp[:, p]

                        self.XpAll_A[jjj].append(Xp[:, self.indivParsIndices_A])
                        if (self.indivParsIndices_B.size > 0):
                            self.XpAll_B[jjj].append(Xp[:, self.indivParsIndices_B])

                        self.XkpAll_A[jjj].append(sparse.csr_matrix(Xkp_A, dtype='d'))
                        self.XkpAll_B[jjj].append(sparse.csr_matrix(Xkp_B, dtype='d'))

                        centerStar = np.argmin(np.sqrt(XC ** 2 + YC ** 2))

                        self.xiAll[jjj].append(xi)
                        self.etaAll[jjj].append(eta)

                        self.xyRawAll[jjj].append(
                            np.vstack([hst1pass['X'][selection], hst1pass['Y'][selection] - yzp]).T)

                        self.plateIDAll[jjj].append(np.full(nData, i, dtype=int))
                        self.indicesAll[jjj].append(np.argwhere(selection).flatten())

                        self.tAll[jjj].append(np.full(nData, t_acs.decimalyear))

                        self.rootnamesAll[jjj].append(np.full(nData, rootname, dtype=object))

                        self.dxAll[jjj].append(xi[centerStar])
                        self.dyAll[jjj].append(eta[centerStar])
                        self.rollAll[jjj].append(np.deg2rad(orientat))

                        self.matchResAll[jjj].append(np.array([delX[selection], delY[selection]]).T)

                        self.nDataImages[jjj].append(nData)

                hduList.close()

                gc.set_threshold(2, 1, 1)
                ## print('Thresholds:', gc.get_threshold())
                ## print('Counts:', gc.get_count())

                del hduList
                del hst1pass
                gc.collect()
                ## print('Counts:', gc.get_count())
            else:
                del hduList
                gc.collect()

    def _shiftRotateReferenceCoordinates(self, xiRef, etaRef, plateID, dx, dy, rotation):
        for i in range(self.nOkay):
            j = self.okayIDs[i]

            sx, sy, roll = dx[i], dy[i], rotation[i]

            selection = plateID == j

            xiRef[selection], etaRef[selection] = coords.shift_rotate_coords(xiRef[selection],
                                                                             etaRef[selection],
                                                                             sx, sy, roll)
        return xiRef, etaRef
    def _getColumnNamesForIndividualCoefficients(self):
        columns_indiv = ['rootname', 'tObs', 'chip']

        for p in range(self.nParsSIP):
            if (p in self.indivParsIndices_A):
                columns_indiv.append(r'{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[0], p + 1))
            if (p in self.indivParsIndices_B):
                columns_indiv.append(r'{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[1], p + 1))
        return columns_indiv

    def _getColumnNamesForSplineCoefficients(self):
        columns_model = ['p', 'coeff_label', 'chip']

        for iii in range(self.nParsK):
            columns_model.append('k_{0:d}'.format(iii))

        return columns_model

    def _parseIndividualAndSplineCoefficientsDataFrame(self, chip, coeffsA, coeffsB,
                                                       modelCoeffsFilename='./splineCoeffs.csv',
                                                       indivDataFilename='./individualCoeffs.csv'):
        data_indiv = [self.rootnames, self.tObs, np.full_like(self.tObs, chip, dtype=int)]
        data_model = []

        columns_indiv = self._getColumnNamesForIndividualCoefficients()
        columns_model = self._getColumnNamesForSplineCoefficients()

        for p in range(self.nParsSIP):
            for axis in range(acsconstants.NAXIS):
                coeffsIndiv = None
                coeffsModel = None

                if ((p in self.indivParsIndices_A) or (p in self.indivParsIndices_B)):
                    if (axis == 0) and (p in self.indivParsIndices_A):
                        iii  = np.argwhere(self.indivParsIndices_A == p).flatten()[0]
                        end  = self.nImages * self.nParsIndiv_A
                        jump = self.nParsIndiv_A

                        coeffsIndiv = coeffsA[iii:end:jump]

                    elif (axis == 1) and (p in self.indivParsIndices_B):
                        iii  = np.argwhere(self.indivParsIndices_B == p).flatten()[0]
                        end  = self.nImages * self.nParsIndiv_B
                        jump = self.nParsIndiv_B

                        coeffsIndiv = coeffsB[iii:end:jump]

                if ((p in self.splineParsIndices_A) or (p in self.splineParsIndices_B)):
                    if (axis == 0) and (p in self.splineParsIndices_A):
                        iii   = np.argwhere(self.splineParsIndices_A == p).flatten()[0]
                        start = self.nImages * self.nParsIndiv_A + iii * self.nParsK
                        end   = start + self.nParsK

                        coeffsModel = coeffsA[start:end]

                    elif (axis == 1) and (p in self.splineParsIndices_B):
                        iii   = np.argwhere(self.splineParsIndices_B == p).flatten()[0]
                        start = self.nImages * self.nParsIndiv_B + iii * self.nParsK
                        end   = start + self.nParsK

                        coeffsModel = coeffsB[start:end]

                if (coeffsIndiv is not None):
                    data_indiv.append(coeffsIndiv)

                if (coeffsModel is not None):
                    dataLine = [p, r'{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], p + 1), chip]

                    for coefficient in coeffsModel:
                        dataLine.append(coefficient)

                    data_model.append(dataLine)

        df_spline_coeffs = pd.DataFrame(data_model, columns=columns_model)

        df_spline_coeffs.to_csv(modelCoeffsFilename, index=False)

        df_indiv_data = pd.DataFrame(np.vstack(data_indiv).T.tolist(), columns=columns_indiv).sort_values(
            ['rootname', 'chip'], ascending=[True, True]).reset_index()

        df_indiv_data.to_csv(indivDataFilename, index=False)

    def _getIndividualCoeffs(self, rootname, p, axis, chip, df):
        sel = df['rootname'].str.contains(rootname) & (df['chip'] == chip)
        if (df[sel].size > 0):
            columnName = '{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], p + 1)
            if columnName in df.columns:
                return df[sel]['{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], p + 1)].values[0]
            else:
                return 0.0
        else:
            return 0.0

    def _getSplineCoeffs(self, p, axis, chip, df):
        sel = (df['coeff_label'] == '{0:s}_{1:d}'.format(acsconstants.COEFF_LABELS[axis], p + 1)) & (df['chip'] == chip)
        return df[sel][['k_{0:d}'.format(k) for k in range(self.nParsK)]].values.reshape((-1,1))