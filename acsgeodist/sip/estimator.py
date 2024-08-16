from acsgeodist import acsconstants
from acsgeodist.tools import astro, bspline, coords, litho, plotting, sip, stat
from acsgeodist.tools.time import convertTime
from astropy import table
from astropy import units as u
from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.io import ascii, fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astroquery.gaia import Gaia
from copy import deepcopy
import gc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator
import numpy as np
import os
from photutils.aperture import CircularAperture
from scipy import sparse
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

## For cross-matching, select only Gaia sources with good astrometry measurements
MIN_RUWE = 0.8
MAX_RUWE = 1.2

chips = np.array([1, 2], dtype=int) ## hdu [SCI, x]
X0    = 2048.00
Y0    = np.array([1024.0, 2048.0+1024.0])

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

class SIPEstimator():
    def __init__(self, referenceCatalog, referenceWCS, tRef0, qMax=0.5, min_n_app=3, max_pix_tol=1.0,
                 min_n_refstar=100, make_lithographic_and_filter_mask_corrections=True, cross_match=True):
        self.refCat        = deepcopy(referenceCatalog)
        self.wcsRef        = deepcopy(referenceWCS)
        self.tRef0         = tRef0
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

        dt = t_acs.decimalyear - self.tRef0

        ## We use the observation time, in combination with the proper motions to move
        ## the coordinates into the time
        self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
        self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

        hst1pass = table.hstack([ascii.read(hst1passFile), ascii.read(addendumFilename)])

        matchRes = np.sqrt((hst1pass['xPred'] - hst1pass['xRef']) ** 2 + (hst1pass['yPred'] - hst1pass['yRef']) ** 2)

        okays = np.zeros(chips.size, dtype='bool')
        nDataBad = np.zeros(chips.size, dtype=int)
        for chip in chips:
            jjj = chip - 1

            selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                        hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                    hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                    matchRes <= self.max_pix_tol)

            nDataBad[jjj] = len(hst1pass[selection])

            if (nDataBad[jjj] > self.min_n_refstar):
                okays[jjj] = True

            print(acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1], acsconstants.CHIP_POSITIONS[chip - 1]), "N_STARS =", nDataBad[jjj], "OKAY:", okays[jjj])

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

                textResults = ""
                for chip in chips:
                    startTime = time.time()

                    jj = 2 - chip
                    jjj = chip - 1

                    chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1], acsconstants.CHIP_POSITIONS[chip - 1])

                    hdu = hduList['SCI', chip]

                    ## Zero point of the y coordinates.
                    if (chip == 1):
                        yzp    = 0.0
                        naxis2 = int(hdu.header['NAXIS2'])
                    else:
                        yzp += float(naxis2)

                        naxis2 = int(hdu.header['NAXIS2'])

                    orientat = Angle(float(hdu.header['ORIENTAT']) * u.deg).wrap_at('360d').value
                    vaFactor = float(hdu.header['VAFACTOR'])

                    selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                                hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                            hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                            matchRes <= self.max_pix_tol)

                    refStarIdx = hst1pass[selection]['refCatIndex'].value

                    nData = len(refStarIdx)

                    print("CHIP:", chipTitle, "P_ORDER:", pOrder, "N_STARS:", nData)

                    xi  = self.refCat[refStarIdx]['xt'] / vaFactor
                    eta = self.refCat[refStarIdx]['yt'] / vaFactor

                    XC = hst1pass['X'][selection] - X0
                    YC = hst1pass['Y'][selection] - Y0[chip - 1]

                    if self.make_lithographic_and_filter_mask_corrections:
                        dcorr = np.array(litho.interp_dtab_ftab_data(self.dtabs[jjj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T
                        fcorr = np.array(litho.interp_dtab_ftab_data(self.ftabs[jjj], hst1pass['X'][selection].value,
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

                    ## Because we want to have individual zero points for each chip we
                    ## initialize the container for shifts and rolls here
                    dxs, dys, rolls = [], [], []

                    nIterTotal = 0
                    for iteration in range(N_ITER_OUTER):
                        dxs.append(sx)
                        dys.append(sy)
                        rolls.append(roll)

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

                            xiPred = np.matmul(X * scalerArray, coeffs[0::2])
                            etaPred = np.matmul(X * scalerArray, coeffs[1::2])

                            ## Residuals already in pixel and in image axis
                            residualsXi  = xiRef - xiPred
                            residualsEta = etaRef - etaPred

                            rmsXi = np.sqrt(np.average(residualsXi ** 2, weights=weights))
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

                            print(baseImageFilename, chip, pOrder, iteration + 1, iteration2 + 1,
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

                    ## print("SHIFTS_X:", dxs)
                    ## print("SHIFTS_Y:", dys)
                    ## print("ROLLS:", rolls)

                    ## Now that we have the coefficients, we repeat the model building for ALL objects in the chip
                    selection = (hst1pass['k'] == chip)

                    hasRefStar = (hst1pass['refCatIndex'] >= 0)

                    refStarIdx = hst1pass[selection]['refCatIndex'].value

                    xiRef  = self.refCat[refStarIdx]['xt'] / vaFactor
                    etaRef = self.refCat[refStarIdx]['yt'] / vaFactor

                    XC = hst1pass['X'][selection] - X0
                    YC = hst1pass['Y'][selection] - Y0[chip - 1]

                    if self.make_lithographic_and_filter_mask_corrections:
                        dcorr = np.array(litho.interp_dtab_ftab_data(self.dtabs[jjj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T
                        fcorr = np.array(litho.interp_dtab_ftab_data(self.ftabs[jjj], hst1pass['X'][selection].value,
                                                               hst1pass['Y'][selection].value - yzp, XRef * 2, YRef * 2)).T

                        ## Apply the lithographic mask correction
                        XC -= (dcorr[:, 2] - fcorr[:, 2])
                        YC -= (dcorr[:, 3] - fcorr[:, 3])

                    X, scalerArray = sip.buildModel(XC, YC, pOrder, scalerX=scalerX, scalerY=scalerY)

                    for iiii in range(len(dxs)):
                        sx = dxs[iiii]
                        sy = dys[iiii]
                        roll = rolls[iiii]

                        xiRef, etaRef = coords.shift_rotate_coords(xiRef, etaRef, sx, sy, roll)

                    xiPred = np.matmul(X * scalerArray, coeffs[0::2])
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

                    selection  = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & hst1pass['retained']
                    refStarIdx = hst1pass['refCatIndex'][selection].value

                    XCorr = hst1pass['xPred'][selection].value
                    YCorr = hst1pass['yPred'][selection].value

                    for iteration3 in range(N_ITER_CD):
                        ## Calculate the normal coordinates relative to the pqr triad centered on the current CRVAL1,2
                        self.refCat = self._getNormalCoordinates(self.refCat, 'xt', 'yt', self.wcsRef, pqr0Im)

                        ## The reference coordinates used for regression of the CD matrix is relative to the current
                        ## CRVAL1, CRVAL2 coordinates
                        xiRef  = (self.refCat['xi'][refStarIdx].value * u.arcsec).to(u.deg) / vaFactor
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
                        selectionChip = hst1pass['k'] == chip

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

                    selection = (hst1pass['k'] == chip)

                    hst1pass['xi'][selection]  = hst1pass['xi'][selection]  + xi0
                    hst1pass['eta'][selection] = hst1pass['eta'][selection] + eta0

                    hst1pass['xiRef'][selection]  = hst1pass['xiRef'][selection] + xi0
                    hst1pass['etaRef'][selection] = hst1pass['etaRef'][selection] + eta0

                    ## Calculate the residuals
                    hst1pass['resXi'][selection] = (
                            hst1pass['xi'][selection].value - hst1pass['xiRef'][selection].value)
                    hst1pass['resEta'][selection] = (
                            hst1pass['eta'][selection].value - hst1pass['etaRef'][selection].value)

                    textResults += "{0:s} {1:d} {2:.8f} {3:.6f} {4:.13f} {5:.12e} {6:0.2f} {7:f} {8:f}".format(
                        baseImageFilename, chip, t_acs.decimalyear, pa_v3, orientat, vaFactor, tExp, posTarg1, posTarg2)
                    textResults += " {0:d} {1:d}".format(nIterTotal, nStars)
                    textResults += " {0:0.6f} {1:0.6f}".format(rmsXi, rmsEta)
                    textResults += " {0:0.12f} {1:0.12f}".format(alpha0Im, delta0Im)
                    textResults += " {0:0.12e} {1:0.12e} {2:0.12e} {3:0.12e}".format(CDMatrix[0, 1], CDMatrix[0, 2],
                                                                                     CDMatrix[1, 1], CDMatrix[0, 2])
                    for coeff in coeffs:
                        textResults += " {0:0.12e}".format(coeff)
                    textResults += "\n"

                    ## Repeat the selection process
                    selection = (hst1pass['k'] == chip) & (hst1pass['refCatID'] >= 0) & (hst1pass['q'] > 0) & (
                                hst1pass['q'] <= self.qMax) & (~np.isnan(hst1pass['nAppearances'])) & (
                                            hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                            matchRes <= self.max_pix_tol)

                    retained = np.zeros(len(hst1pass), dtype=bool)

                    retained[indices] = True

                    image = hdu.data

                    vmin, vmax = ZScaleInterval(contrast=0.10, max_iterations=5).get_limits(image)

                    axes[jj].imshow(image, cmap=cMap, aspect='equal', vmin=vmin, vmax=vmax, origin='lower', rasterized=True)

                    xyPosRetained = np.vstack([hst1pass[retained]['X'].value, hst1pass[retained]['Y'].value - yzp]).T

                    aperturesRetained = CircularAperture(xyPosRetained, r=20.0)

                    aperturesRetained.plot(color='#1b9e77', lw=1, alpha=1, ax=axes[jj],
                                           rasterized=True);  ## GREENS are accepted sources

                    xyPosRejected = np.vstack([hst1pass[(~retained) & selection]['X'].value,
                                               hst1pass[(~retained) & selection]['Y'].value - yzp]).T

                    aperturesRejected = CircularAperture(xyPosRejected, r=15.0)

                    aperturesRejected.plot(color='#d95f02', lw=1, alpha=1, ax=axes[jj],
                                           rasterized=True);  ## BROWNS are accepted sources

                    axes[jj].set_title('{0:s} --- {1:s}'.format(baseImageFilename, chipTitle))

                    axes[jj].xaxis.set_major_locator(MultipleLocator(dX))
                    axes[jj].xaxis.set_minor_locator(MultipleLocator(dMX))

                    axes[jj].yaxis.set_major_locator(MultipleLocator(dY))
                    axes[jj].yaxis.set_minor_locator(MultipleLocator(dMY))

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
                    nDataBad, self.min_n_refstar))

        hduList.close()

        gc.set_threshold(2, 1, 1)
        print('Thresholds:', gc.get_threshold())
        print('Counts:', gc.get_count())

        del hduList
        del hst1pass
        gc.collect()
        print('Counts:', gc.get_count())

        return textResults

    def _getCDMatrix(self, xiInt, etaInt, xiRef, etaRef, weights=None, pOrder=1, scalerX=1.0, scalerY=1.0):
        H, _ = sip.buildModel(xiInt, etaInt, pOrder, scalerX=scalerX, scalerY=scalerY)

        reg = linear_model.LinearRegression(fit_intercept=False, copy_X=True)

        reg.fit(H, xiRef)

        C1 = reg.coef_

        reg.fit(H, etaRef)

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

T_MIN1 = 2002.1960
T_MAX1 = 2007.0720

T_MIN2 = 2009.4000
T_MAX2 = 2025.0000

class TimeDependentBSplineEstimator(SIPEstimator):
    def __init__(self, tMin, tMax, referenceCatalog, referenceWCS, tRef0, pOrderIndiv, pOrder, kOrder, nKnots,
                 qMax=0.5, min_n_app=3, max_pix_tol=1.0, min_n_refstar=100,
                 make_lithographic_and_filter_mask_corrections=True, cross_match=True):
        super().__init__(referenceCatalog, referenceWCS, tRef0, qMax=qMax, min_n_app=min_n_app, max_pix_tol=max_pix_tol,
                       min_n_refstar=min_n_refstar,
                       make_lithographic_and_filter_mask_corrections=make_lithographic_and_filter_mask_corrections,
                       cross_match=cross_match)
        self.pOrderIndiv = pOrderIndiv ## Maximum polynomial order that are inferred individually for each image
        self.pOrder      = pOrder      ## Total polynomial orders, including those with time-dependent model
        self.kOrder      = kOrder      ## B-spline order
        self.nKnots      = nKnots      ## Number of knots
        self.tMin        = tMin
        self.tMax        = tMax

        self.nParsPIndiv = sip.getUpperTriangularMatrixNumberOfElements(self.pOrderIndiv + 1)  ## Number of parameters PER AXIS!
        self.nParsP      = sip.getUpperTriangularMatrixNumberOfElements(self.pOrder + 1)       ## Number of parameters PER AXIS!
        self.nParsK      = self.nKnots + self.kOrder  ## Number of parameters include constant parameter (zero point)

        self.tKnot  = np.linspace(self.tMin, self.tMax, nKnots, endpoint=True)
        self.dtKnot = self.tKnot[1] - self.tKnot[0]

    def estimateTimeDependentBSplineCoefficients(self, hst1passFiles, imageFilenames, outDir='.', **kwargs):
        nOkay = 0
        nDataAll = np.zeros(2, dtype=int)

        okayIDs = []

        plateIDAll = []

        tAll = []

        rootnamesAll = []

        xiAll = []
        etaAll = []

        XpAll = []
        XkpAll = []

        xyRawAll = []

        dxAll = []
        dyAll = []
        rollAll = []

        nDataImages = []

        XtAll = []

        XAll = []

        matchResAll = []

        startTime = time.time()

        for chip in chips:
            jjj = chip - 1

            plateIDAll.append([])
            tAll.append([])
            rootnamesAll.append([])

            xiAll.append([])
            etaAll.append([])

            XpAll.append([])
            XkpAll.append([])

            xyRawAll.append([])

            dxAll.append([])
            dyAll.append([])
            rollAll.append([])

            nDataImages.append([])

            matchResAll.append([])

            XAll.append([])

        for i, (hst1passFile, imageFilename) in enumerate(zip(hst1passFiles, imageFilenames)):
            addendumFilename = hst1passFile.replace('.csv', '_addendum.csv')

            baseImageFilename = os.path.basename(hst1passFile).replace('_hst1pass_stand.csv', '')

            rootName = baseImageFilename.split('_')[0]

            if (os.path.exists(imageFilename)) and (os.path.exists(addendumFilename)):
                hduList = fits.open(imageFilename)

                tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
                t_acs = Time(tstring, scale='utc', format='fits')

                if ((t_acs.decimalyear >= self.tMin) and (t_acs.decimalyear <= self.tMax)):
                    print()
                    print(i, os.path.basename(hst1passFile), os.path.basename(addendumFilename), baseImageFilename,
                          rootName)

                    pa_v3 = float(hduList[0].header['PA_V3'])

                    dt = t_acs.decimalyear - self.tRef0

                    ## We use the observation time, in combination with the proper motions to move
                    ## the coordinates into the time
                    self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
                    self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

                    hst1pass = table.hstack([ascii.read(hst1passFile), ascii.read(addendumFilename)])

                    delX = hst1pass['xPred'] - hst1pass['xRef']
                    delY = hst1pass['yPred'] - hst1pass['yRef']

                    matchRes = np.sqrt(delX ** 2 + delY ** 2)

                    okays = np.zeros(chips.size, dtype='bool')
                    nDataBad = np.zeros(chips.size, dtype=int)
                    for chip in chips:
                        jjj = chip - 1

                        selection = (hst1pass['k'] == chip) & (hst1pass['refCatIndex'] >= 0) & (hst1pass['q'] > 0) & (
                                    hst1pass['q'] <= Q_MAX) & (~np.isnan(hst1pass['nAppearances'])) & (
                                                hst1pass['nAppearances'] >= self.min_n_app) & (~np.isnan(matchRes)) & (
                                                matchRes <= self.max_pix_tol)

                        nDataBad[jjj] = len(hst1pass[selection])

                        if (nDataBad[jjj] > self.min_n_refstar):
                            okays[jjj] = True

                        print(acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1], acsconstants.CHIP_POSITIONS[chip - 1]),
                              "N_STARS =", nDataBad[jjj], "OKAY:", okays[jjj])

                    del selection
                    gc.collect()

                    okayToProceed = np.prod(okays, dtype='bool')

                    print("OKAY TO PROCEED:", okayToProceed)

                    Xt = bspline.getForwardModelBSpline(t_acs.decimalyear, self.kOrder, self.tKnot)

                    if okayToProceed:
                        nOkay += 1

                        okayIDs.append(i)

                        XtAll.append(Xt)

                        for chip in chips:
                            jj = 2 - chip
                            jjj = chip - 1

                            chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1],
                                                                acsconstants.CHIP_POSITIONS[chip - 1])

                            hdu = hduList['SCI', chip]

                            ## Zero point of the y coordinates.
                            if (chip == 1):
                                yzp = 0.0
                                naxis2 = int(hdu.header['NAXIS2'])
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

                            print("CHIP:", chipTitle, "N_STARS:", nData)

                            nDataAll[jjj] += nData

                            xi  = self.refCat[refStarIdx]['xt'] / vaFactor
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

                                ## print(dcorr)
                                ## print(fcorr)

                                ## Apply the lithographic and filter mask correction
                                XC -= (dcorr[:, 2] - fcorr[:, 2])
                                YC -= (dcorr[:, 3] - fcorr[:, 3])

                                del dcorr
                                del fcorr

                            Xp, scalerArray = sip.buildModel(XC, YC, self.pOrder, scalerX=scalerX, scalerY=scalerY)

                            Xkp = np.zeros((Xp.shape[0], self.nParsK * (self.nParsP - self.nParsPIndiv)))

                            for p in range(self.nParsPIndiv, self.nParsP):
                                for k in range(self.nParsK):
                                    Xkp[:, (p - self.nParsPIndiv) * self.nParsK + k] = Xt[0, k] * Xp[:, p]

                            XpAll[jjj].append(Xp[:, :self.nParsPIndiv])
                            XkpAll[jjj].append(sparse.csr_matrix(Xkp, dtype='d'))

                            centerStar = np.argmin(np.sqrt(XC ** 2 + YC ** 2))

                            xiAll[jjj].append(xi)
                            etaAll[jjj].append(eta)

                            xyRawAll[jjj].append(
                                np.vstack([hst1pass['X'][selection], hst1pass['Y'][selection] - yzp]).T)

                            plateIDAll[jjj].append(np.full(nData, i, dtype=int))

                            tAll[jjj].append(np.full(nData, t_acs.decimalyear))

                            rootnamesAll[jjj].append(np.full(nData, rootName, dtype=object))

                            dxAll[jjj].append(xi[centerStar])
                            dyAll[jjj].append(eta[centerStar])
                            rollAll[jjj].append(np.deg2rad(orientat))

                            matchResAll[jjj].append(np.array([delX[selection], delY[selection]]).T)

                            nDataImages[jjj].append(nData)

                    hduList.close()

                    gc.set_threshold(2, 1, 1)
                    print('Thresholds:', gc.get_threshold())
                    print('Counts:', gc.get_count())

                    del hduList
                    del hst1pass
                    gc.collect()
                    print('Counts:', gc.get_count())
                else:
                    del hduList
                    gc.collect()