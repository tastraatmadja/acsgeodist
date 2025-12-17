from copy import deepcopy
import gc
import os
import time

from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy import wcs
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, ticker
import numpy as np
from photutils.aperture import CircularAperture
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree

from acsgeodist import acsconstants
from acsgeodist.tools import coords, corr, plotting, reader, sip, stat
from acsgeodist.tools.time import convertTime

class WCSAlignment:
    ## The grid of bins for the 2d histogram image
    delX = 1.0
    delY = delX

    ## KDE parameters. Width of the gaussian around the centroids.
    hx, hy = 1.0, 1.0

    MAX_SHIFT = np.inf

    cMap0 = 'Greys'

    dX0, dMX0 = 1000, 200
    dY0, dMY0 = 500, 100

    xLabel0, yLabel0 = r'$X$ [pix]', r'$Y$ [pix]'

    ## Plotting footprint
    xSize1 = 10
    ySize1 = xSize1

    markerSize1 = 1
    lineWidth1 = 2
    zorder1 = 5

    refStarColor    = '#ca0020'  ## RED
    nonRefStarColor = '#0571b0'  ## BLUE

    def __init__(self, referenceCatalog, referenceWCS, tRef0, refImage=None, n_image_stars=1000, n_ref_stars=1000,
                 in_footprint=True, qMax=0.3, min_n_good_sources=100, max_sep_init=5.0 * u.pix,
                 max_sep_refined=2.0 * u.pix, pOrder=1, n_iter=100):
        self.df_refCat = deepcopy(referenceCatalog)
        self.wcsRef    = deepcopy(referenceWCS)
        self.tRef0     = deepcopy(tRef0)
        self.refImage  = None
        if (refImage is not None):
            self.refImage = deepcopy(refImage)

        self.n_image_stars = n_image_stars
        self.n_ref_stars   = n_ref_stars
        self.in_footprint  = in_footprint

        self.qMax = qMax
        self.min_n_good_sources = min_n_good_sources
        self.max_sep_init = max_sep_init
        self.max_sep_refined = max_sep_refined
        self.pOrder = pOrder
        self.n_iter = n_iter

        self.n = self.pOrder + 1
        self.P = sip.getUpperTriangularMatrixNumberOfElements(self.n)


    def alignImage(self, hst1passFilenames, imageFilenames, outDir='.', detectorName='WFC'):
        if not isinstance(hst1passFilenames, list):
            hst1passFilenames = [hst1passFilenames]
        if not isinstance(imageFilenames, list):
            imageFilenames = [imageFilenames]

        self.detectorName = detectorName
        yAxisExtendFactor = 1.2
        if (self.detectorName == 'WFC'):
            self.n_chips        = acsconstants.N_CHIPS
            self.header_numbers = acsconstants.HEADER_NUMBER
            yAxisExtendFactor  = 2.5
        elif (self.detectorName == 'SBC'):
            self.n_chips        = acsconstants.SBC_N_CHIPS
            self.header_numbers = acsconstants.SBC_HEADER_NUMBER

        nRows0 = self.n_chips
        nCols0 = 1

        ## Plotting detected sources
        xSize0 = 12
        ySize0 = xSize0

        shiftInfoFilename = "{0:s}/shifts_pOrder{1:d}.txt".format(outDir, self.pOrder)

        if (not os.path.exists(shiftInfoFilename)):
            f = open(shiftInfoFilename, 'w')
            f.write("#rootname nSrc nMtch mtchFrac ")

            for chip in range(len(acsconstants.CHIP_POSITIONS)):
                for p in range(self.P):
                    for axis in range(acsconstants.NAXIS):
                        f.write("{0:16s}".format(
                            '{0:s}{1:d}_{2:d}'.format(acsconstants.COEFF_LABELS[axis], p + 1, chip + 1)))
            f.write("\n")
            f.close()

        gc.set_threshold(2, 1, 1)

        # Here we read the df_hst1pass results file by file and cross-match the detected sources with the external
        # reference catalogue. Only select stars with fit quality q <= Q_MAX. We use the WCS of both the external
        # reference catalogue and the given plate to transform the coordinates into RA and DEC. The RA and DEC are then
        # transformed into normal coordinates assuming a common reference point. We then use phase correlation to
        # estimate the shifts between the two catalogues (the scale and rotation are assumed to be close enough).
        # Initial match is then performed. We then use the stars in these initial match to derive an improved linear
        # transformation and performed the cross-matching the second time.
        for hst1passFilename, imageFilename in zip(hst1passFilenames, imageFilenames):
            print(os.path.basename(hst1passFilename), os.path.basename(imageFilename), self.detectorName)
            df_hst1pass = reader.readHST1PassFile(hst1passFilename, detector=self.detectorName)

            hduList = fits.open(imageFilename)

            rootname = hduList[0].header['ROOTNAME']

            outTableFilename = '{0:s}/{1:s}_hst1pass_stand.csv'.format(outDir, rootname)
            errTableFilename = '{0:s}/{1:s}_hst1pass_stand.err'.format(outDir, rootname)

            if (not os.path.exists(outTableFilename)) and (not os.path.exists(errTableFilename)):
                startTime0 = time.time()

                gc.set_threshold(2, 1, 1)

                tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
                t_acs = Time(tstring, scale='ut1', format='fits')

                dt = t_acs.tcb.decimalyear - self.tRef0.tcb.value

                self.df_refCat['xt'] = self.df_refCat['x'] + self.df_refCat['pm_x'] * dt
                self.df_refCat['yt'] = self.df_refCat['y'] + self.df_refCat['pm_y'] * dt

                self.df_refCat['in_footprint'] = False
                self.df_refCat['hasSource']    = False

                ## Create new columns with default values
                df_hst1pass['hasRefCat'] = False
                df_hst1pass['refCatID'] = -1
                df_hst1pass['refCatIndex'] = -1
                df_hst1pass['xPred'] = np.nan
                df_hst1pass['yPred'] = np.nan
                df_hst1pass['xRef'] = np.nan
                df_hst1pass['yRef'] = np.nan

                coeffs = np.zeros((acsconstants.N_CHIPS, 2 * self.P))

                fig1 = plt.figure(figsize=(WCSAlignment.xSize1, WCSAlignment.ySize1))

                ax1 = fig1.add_subplot(111, projection=self.wcsRef)
                ## ax1 = fig1.add_subplot(111)

                ax1.set_aspect('equal')

                ax1.set_xlabel(r'$\alpha$ [hms]')
                ax1.set_ylabel(r'$\delta$ [$^\circ$]')

                ax1.set_title(rootname + ' --- ' + t_acs.iso);

                if (self.refImage is not None):
                    vmin, vmax = 0, np.percentile(self.refImage[self.refImage > 0], 97)

                    ax1.imshow(self.refImage, cmap=WCSAlignment.cMap0, vmin=vmin, vmax=vmax, origin='lower')


                fig0, axes0 = plt.subplots(figsize=(xSize0, ySize0), nrows=nRows0, ncols=nCols0, rasterized=True,
                                           squeeze=False)

                plt.subplots_adjust(wspace=0.0, hspace=0.15)

                handles = ()

                phaseCorrelationFigures = []

                title = '{0:s}'.format(rootname)

                for jj in range(self.n_chips):
                    startTime = time.time()

                    ver = self.header_numbers[jj]
                    jjj = self.n_chips - ver ## Index for plotting

                    hdu = hduList['SCI', ver]

                    k = 1

                    chipLabel = None
                    if (self.detectorName == 'WFC'):
                        chipLabel = acsconstants.CHIP_LABEL(acsconstants.WFC[jj], acsconstants.CHIP_POSITIONS[jj])
                        title = '{0:s} --- {1:s}'.format(rootname, chipLabel)

                        k = int(hdu.header['CCDCHIP'])
                    elif (self.detectorName == 'SBC'):
                        chipLabel = self.detectorName

                    naxis1 = int(hdu.header['NAXIS1'])
                    naxis2 = int(hdu.header['NAXIS2'])

                    # Zero point of the y coordinates.
                    if (ver == 1):
                        yzp = 0.0
                    else:
                        yzp += float(naxis2)

                    ## Make a bunch of plots first.
                    ## Here we start to plot the image + detected sources
                    image0 = hdu.data

                    vmin0, vmax0 = ZScaleInterval(contrast=0.10, max_iterations=5).get_limits(image0)

                    axes0[jjj,0].imshow(image0, cmap=WCSAlignment.cMap0, aspect='equal', vmin=vmin0, vmax=vmax0,
                                        origin='lower')

                    axes0[jjj,0].set_title(title)

                    axes0[jjj,0].xaxis.set_major_locator(ticker.AutoLocator())
                    axes0[jjj,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())

                    axes0[jjj,0].yaxis.set_major_locator(ticker.AutoLocator())
                    axes0[jjj,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

                    ## Now we start to plot the image footprint on the celestial sky
                    wcsIm = wcs.WCS(hdu.header, fobj=hduList)

                    footprint, in_footprint = coords.calculateFootprintAndIfPointsAreInside(
                        wcsIm, self.wcsRef, points=self.df_refCat[['xt', 'yt']].values)

                    self.df_refCat.loc[in_footprint, ['in_footprint']] = True

                    ax1.plot(footprint[:, 0], footprint[:, 1], '-', color=acsconstants.WFC_COLORS[jj],
                         linewidth=WCSAlignment.lineWidth1, zorder=WCSAlignment.zorder1, label=chipLabel)

                    chipMidPoint = coords.getRectangleMidpoint(footprint)

                    ax1.text(chipMidPoint[0], chipMidPoint[1], chipLabel, color=acsconstants.WFC_COLORS[jj],
                             ha='center', va='center', zorder=WCSAlignment.zorder1)

                    if (((self.detectorName == 'WFC') and (acsconstants.CHIP_POSITIONS[jj] == 'bottom'))
                            or (self.detectorName == 'SBC')):
                        originPixRef = np.array(wcs.utils.pixel_to_pixel(wcsIm, self.wcsRef, 0, 0))
                        xAxisPixRef = np.array(wcs.utils.pixel_to_pixel(wcsIm, self.wcsRef, naxis1, 0))
                        yAxisPixRef = np.array(wcs.utils.pixel_to_pixel(wcsIm, self.wcsRef, 0, naxis2))

                        xAxisPixRef = originPixRef + 1.2 * (xAxisPixRef - originPixRef)
                        yAxisPixRef = originPixRef + yAxisExtendFactor * (yAxisPixRef - originPixRef)

                        ax1.annotate(r'$x$', color='r', xy=originPixRef, xycoords='data', xytext=xAxisPixRef,
                                     textcoords='data', ha='center', va='center',
                                     arrowprops=dict(arrowstyle="<-", color="r"), zorder=WCSAlignment.zorder1+1)
                        ax1.annotate(r'$y$', color='r', xy=originPixRef, xycoords='data', xytext=yAxisPixRef,
                                     textcoords='data', ha='center', va='center',
                                     arrowprops=dict(arrowstyle="<-", color="r"), zorder=WCSAlignment.zorder1+1)

                    selection  = (df_hst1pass['k'] == k) & (df_hst1pass['q'] > 0) & (df_hst1pass['q'] <= self.qMax)
                    nSelection = len(selection[selection])

                    nRefStarsFootprint = len(self.df_refCat[self.df_refCat['in_footprint']])

                    print('{0:s}: Found {1:d} out of {2:d} reference stars in image footprint.'.format(
                        chipLabel, nRefStarsFootprint, len(self.df_refCat)))

                    if (nSelection > self.min_n_good_sources) and (nRefStarsFootprint > 0):
                        nImStars = min(nSelection, self.n_image_stars)

                        x_im, y_im = df_hst1pass[selection][['X', 'Y']].values.T
                        y_im -= yzp

                        df_hst1pass.loc[selection, ['x_im', 'y_im']] = np.array(
                            wcs.utils.pixel_to_pixel(wcsIm, self.wcsRef, x_im, y_im)).T

                        x_im, y_im = df_hst1pass[selection].sort_values('m', ascending=True)[
                                         ['x_im', 'y_im']].values[:nImStars].T

                        ## x_ref, y_ref = df_refCat.sort_values('m', ascending=True)[['xt', 'yt']].values[:nRefStarSelected].T
                        if self.in_footprint:
                            nRefStarSelected = min(self.n_ref_stars,
                                                   len(self.df_refCat.loc[self.df_refCat['in_footprint']]))

                            x_ref, y_ref = self.df_refCat.loc[self.df_refCat['in_footprint']].sort_values(
                                'm', ascending=True)[['xt', 'yt']].values[:nRefStarSelected].T
                        else:
                            nRefStarSelected = min(self.n_ref_stars, len(self.df_refCat))

                            x_ref, y_ref = self.df_refCat.sort_values(
                                'm', ascending=True)[['xt', 'yt']].values[:nRefStarSelected].T

                        print("Selecting {0:d} reference stars (in footprint: {1:s})".format(x_ref.size,
                                                                                             str(self.in_footprint)))

                        shiftX, shiftY, corrIdx, corrIm = corr.phaseCorrelate2d(x_im, y_im, x_ref, y_ref,
                                                                                WCSAlignment.delX,
                                                                                WCSAlignment.delY,
                                                                                WCSAlignment.hx,
                                                                                WCSAlignment.hy)

                        print("SHIFT:", shiftX, shiftY)

                        if ((abs(shiftX) < WCSAlignment.MAX_SHIFT) and (abs(shiftY) < WCSAlignment.MAX_SHIFT)):
                            ## Display the 2d phase correlation image on the main plot and the 1d phase correlation on the top and right subplots
                            xSize2 = 8
                            ySize2 = xSize2

                            fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(xSize2, ySize2), sharex='col',
                                                     sharey='row', width_ratios=[1.0, 0.5],
                                                     height_ratios=[0.5, 1.0], rasterized=True)

                            ## ax2[1,0].set_aspect('equal')

                            vmin2, vmax2 = ZScaleInterval(contrast=0.1, max_iterations=5).get_limits(corrIm)

                            ax2[1, 0].imshow(corrIm, aspect='auto', interpolation='None', vmin=vmin2,
                                             vmax=vmax2)

                            ax2[1, 0].plot(corrIdx[1], corrIdx[0], 'wx')

                            ## We don't need this plot so set it to invisible
                            ax2[0, 1].set_visible(False)

                            ## This is the aux plot where we make the 1d phase correlation along x-axis, at the point that passes
                            ## through the maximum y.
                            ax2[0, 0].plot(np.arange(corrIm.shape[1]), corrIm[corrIdx[0]], 'k-')

                            ax2[0, 0].set_ylabel(r'Correlation')

                            ## This is the aux plot where we make the 1d phase correlation along y-axis, at the point that passes
                            ## through the maximum x.
                            ax2[1, 1].plot(corrIm[:, corrIdx[1]], np.arange(corrIm.shape[0]), 'k-')

                            ax2[1, 1].set_xlabel(r'Prob. Density')
                            ax2[1, 1].set_xlabel(r'Correlation')

                            plt.subplots_adjust(wspace=0.0, hspace=0.0)

                            axCommons2 = plotting.drawCommonLabel('', '', fig2, xPad=0, yPad=0)

                            axCommons2.set_title(title);

                            phaseCorrelationFigures.append(fig2)

                            del corrIm
                            gc.collect()

                            ## Prepare the KDTree for ALL reference stars
                            x_ref, y_ref = self.df_refCat[['xt', 'yt']].values.T

                            tree = KDTree(np.vstack([x_ref, y_ref]).T)

                            x_im, y_im = df_hst1pass[selection][['x_im', 'y_im']].values.T

                            dist, idx = tree.query(np.vstack([x_im - shiftX, y_im - shiftY]).T, k=1)

                            dist = dist.flatten()
                            idx = idx.flatten()

                            ## Create a mask for ACS sources thas has refCat counterpart, i.e. cross-matched
                            ## to better than (predefined) MAX_SEP
                            hasRefCat = (dist < self.max_sep_init.value)

                            ## Find doubly-identified sources and remove them. We use numpy unique
                            ## to identify refCat sources cross-matched within MAX_SEP to multiple sources
                            ## in the ACS image
                            doubleIndices, counts = np.unique(idx[hasRefCat], return_counts=True)

                            ## We now create mask for ACS sources that are cross-matched to the same
                            ## RefCat sources.
                            doubleSources = np.where(np.isin(idx, doubleIndices[counts > 1]))

                            ## We remove sources cross-matched to the same RefCat sources.
                            hasRefCat[doubleSources] = False

                            print("INITIAL_MATCH: {0:d} out of {1:d} selected sources".format(
                                hasRefCat[hasRefCat].size, hasRefCat.size))

                            ## We now make a (linear) forward model to transform observed xi and eta to
                            ## reference xi and eta
                            X, _ = sip.buildModel(x_im[hasRefCat], y_im[hasRefCat], self.pOrder)

                            ## Initialize the weights
                            weights = np.ones(X.shape[0])

                            previousWeightSum = np.sum(weights)

                            xRef = x_ref[idx[hasRefCat]]
                            yRef = y_ref[idx[hasRefCat]]

                            for iteration in range(self.n_iter):
                                ## Initialize the linear regression
                                reg = LinearRegression(fit_intercept=False, copy_X=True)

                                reg.fit(X, xRef, sample_weight=weights)

                                A = reg.coef_

                                reg.fit(X, yRef, sample_weight=weights)

                                B = reg.coef_

                                xPred = np.matmul(X, A)
                                yPred = np.matmul(X, B)

                                residuals = np.vstack([xRef - xPred, yRef - yPred]).T

                                rmsXi = np.sqrt(np.average(residuals[:, 0] ** 2, weights=weights))
                                rmsEta = np.sqrt(np.average(residuals[:, 1] ** 2, weights=weights))

                                ## Use the weights to estimate the mean and covariance matrix of the residual
                                ## distribution
                                mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(residuals, weights)

                                ## Calculate the Mahalanobis Distance, i.e. standardized distance
                                ## from the center of the gaussian distribution
                                z = stat.getMahalanobisDistances(residuals, mean, np.linalg.inv(cov))

                                ## We now use the z statistics to re-calculate the weights
                                weights = stat.wdecay(z)

                                weightSum = np.sum(weights)

                                weightSumDiff = np.abs(weightSum - previousWeightSum) / weightSum

                                ## Assign the current weight summation for the next iteration
                                previousWeightSum = weightSum

                                ## Select stars with zero weights
                                rejected = weights <= 0

                                if ((iteration == 0) or (((iteration + 1) % 10) == 0)):
                                    print(rootname, ver, self.pOrder, iteration + 1,
                                          "N_STARS: {0:d}/{1:d}".format(xPred[~rejected].size, (xPred.size)),
                                          "RMS: {0:.6f} {1:.6f}".format(rmsXi, rmsEta),
                                          "W_SUM: {0:0.6f}".format(weightSum))

                                if (weightSumDiff < 1.e-9):
                                    break
                                else:
                                    X = X[~rejected]

                                    weights = weights[~rejected]

                                    xRef = xRef[~rejected]
                                    yRef = yRef[~rejected]

                            print("A:", A)
                            print("B:", B)

                            ## Now that we have obtained a reasonably good transformation of xi and eta, for the
                            ## cross-matching we repeat the process of calculating the normal coordinates, but
                            ## now for ALL OBJECTS in the chip catalogue. We reuse the variables because we
                            ## don't need the old ones.
                            selection = (df_hst1pass['k'] == k)

                            nDataOriginal = len(selection[selection])

                            x_im, y_im = df_hst1pass[selection][['X', 'Y']].values.T
                            y_im -= yzp

                            df_hst1pass.loc[selection, ['x_im', 'y_im']] = np.array(
                                wcs.utils.pixel_to_pixel(wcsIm, self.wcsRef, x_im, y_im)).T

                            x_im, y_im = df_hst1pass.loc[selection, ['x_im', 'y_im']].values.T

                            X, _ = sip.buildModel(x_im, y_im, self.pOrder)

                            xPred = X @ A
                            yPred = X @ B

                            ## Consult the tree again, now with the refined coordinates
                            dist, idx = tree.query(np.vstack([xPred, yPred]).T, k=1)

                            dist = dist.flatten()
                            idx = idx.flatten()

                            ## Create a mask for ACS sources thas has refCat counterpart, i.e. cross-matched
                            ## to better than (predefined) MAX_SEP
                            hasRefCat = (dist < self.max_sep_refined.value)

                            ## Find doubly-identified sources and remove them. We use numpy unique
                            ## to identify refCat sources cross-matched within MAX_SEP to multiple sources
                            ## in the ACS image
                            doubleIndices, counts = np.unique(idx[hasRefCat], return_counts=True)

                            ## We now create mask for ACS sources that are cross-matched to the same
                            ## RefCat sources.
                            doubleSources = np.where(np.isin(idx, doubleIndices[counts > 1]))

                            ## We remove sources cross-matched to the same RefCat sources.
                            hasRefCat[doubleSources] = False

                            refCatID = self.df_refCat['id'][idx]

                            refCatID[~hasRefCat] = -1
                            idx[~hasRefCat] = -1

                            self.df_refCat.loc[idx[hasRefCat], 'hasSource'] = True

                            print("REFINED_MATCH: {0:d} out of {1:d} sources".format(hasRefCat[hasRefCat].size,
                                                                                     hasRefCat.size))

                            ax1.plot(x_ref[idx[hasRefCat]], y_ref[idx[hasRefCat]], '.',
                                     markersize=WCSAlignment.markerSize1, color=acsconstants.WFC_COLORS[jj],
                                     alpha=0.5)

                            ## Append new columns to the df_hst1pass results
                            df_hst1pass.loc[selection, ['hasRefCat']] = deepcopy(hasRefCat)
                            df_hst1pass.loc[selection, ['refCatID']] = deepcopy(refCatID.values)
                            df_hst1pass.loc[selection, ['refCatIndex']] = deepcopy(idx)
                            df_hst1pass.loc[selection, ['xPred']] = deepcopy(xPred)
                            df_hst1pass.loc[selection, ['yPred']] = deepcopy(yPred)
                            df_hst1pass.loc[selection, ['xRef']] = deepcopy(x_ref[idx])
                            df_hst1pass.loc[selection, ['yRef']] = deepcopy(y_ref[idx])

                            coeffs[jj, 0::2] = deepcopy(A)
                            coeffs[jj, 1::2] = deepcopy(B)

                            selection = (df_hst1pass['k'] == k) & (~df_hst1pass['hasRefCat'])

                            xyPos = df_hst1pass.loc[selection, ['X', 'Y']].values
                            xyPos[:, 1] -= yzp

                            apertures = CircularAperture(xyPos, r=20.0)

                            nrs_patches = apertures.plot(color=WCSAlignment.nonRefStarColor, lw=1, alpha=1,
                                                         ax=axes0[jjj,0], label='Non-ref. star');

                            selection = (df_hst1pass['k'] == k) & df_hst1pass['hasRefCat']

                            xyPos = df_hst1pass.loc[selection, ['X', 'Y']].values
                            xyPos[:, 1] -= yzp

                            apertures = CircularAperture(xyPos, r=20.0)

                            rs_patches = apertures.plot(color=WCSAlignment.refStarColor, lw=1, alpha=1,
                                                        ax=axes0[jjj,0], label='Ref. star');

                            handles = (rs_patches[0], nrs_patches[0])

                            del X
                            del reg
                            del hasRefCat
                            del refCatID
                            del idx
                            del xPred
                            del yPred
                            del xRef
                            del yRef
                            del xyPos
                            del apertures
                            del selection
                            gc.collect()
                        else:
                            errMessage = title+": SOMETHING FISHY IS GOING ON... UNABLE TO FIND SHIFT USING PHASE CORRELATION"
                            print(errMessage)

                            self._writeErrorMessage(errTableFilename, errMessage)

                            selection = (df_hst1pass['k'] == k)

                            xyPos = df_hst1pass.loc[selection, ['X', 'Y']].values
                            xyPos[:, 1] -= yzp

                            apertures = CircularAperture(xyPos, r=20.0)

                            nrs_patches = apertures.plot(color=WCSAlignment.nonRefStarColor, lw=1, alpha=1,
                                                         ax=axes0[jjj, 0], label='Non-ref. star');

                            handles = (nrs_patches[0])

                            del selection
                            del corrIm
                            del xyPos
                            del apertures
                            gc.collect()
                    else:
                        if (nSelection <= self.min_n_good_sources):
                            errMessage = title+": NOT ENOUGH DETECTED STARS WITH FIT QUALITY Q < {0:0.2f}: {1:d} (MIN: {2:d})".format(
                                self.qMax, nSelection, self.min_n_good_sources)
                            print(errMessage)
                            self._writeErrorMessage(errTableFilename, errMessage)

                        if (nRefStarsFootprint <= 0):
                            errMessage = title+": NO REFERENCE STARS FOUND! FOOTPRINT IS OUTSIDE THE RANGE OF EXTERNAL CATALOGUE!"
                            print(errMessage)
                            self._writeErrorMessage(errTableFilename, errMessage)

                        selection = (df_hst1pass['k'] == k)

                        xyPos = df_hst1pass.loc[selection, ['X', 'Y']].values
                        xyPos[:, 1] -= yzp

                        apertures = CircularAperture(xyPos, r=20.0)

                        nrs_patches = apertures.plot(color=WCSAlignment.nonRefStarColor, lw=1, alpha=1,
                                                     ax=axes0[jjj, 0], label='Non-ref. star');

                        handles = (nrs_patches[0])

                    elapsedTime = time.time() - startTime
                    print("CHIP DONE! Elapsed time:", convertTime(elapsedTime))

                if (len(phaseCorrelationFigures) > 0):
                    plotFilename2 = "{0:s}/plot_{1:s}_phaseCorrelation.pdf".format(outDir, rootname)

                    pp1 = PdfPages(plotFilename2)

                    for fig2 in phaseCorrelationFigures:
                        pp1.savefig(fig2, bbox_inches='tight', dpi=300)
                        plt.close(fig=fig2)

                        del fig2

                    pp1.close()
                    del phaseCorrelationFigures

                    print("Phase correlation plot saved to {0:s}".format(plotFilename2))

                ax1.plot(self.df_refCat.loc[~self.df_refCat['hasSource'], 'xt'],
                         self.df_refCat.loc[~self.df_refCat['hasSource'], 'yt'], '.', markersize=1, alpha=0.5,
                         color='#fc8d62', zorder=1)

                plotFilename1 = "{0:s}/plot_{1:s}_footprint.pdf".format(outDir, rootname)

                fig1.savefig(plotFilename1, bbox_inches='tight', dpi=300)

                print("Footprint image saved to {0:s}".format(plotFilename1))

                plt.close(fig=fig1)

                del fig1

                if (len(handles) > 0):
                    axes0[0, 0].legend(loc=(0.0, 1.0), handles=handles, handlelength=1, handletextpad=0.5, frameon=False)

                axCommons = plotting.drawCommonLabel(WCSAlignment.xLabel0, WCSAlignment.yLabel0, fig0, xPad=20, yPad=15)

                plotFilename0 = "{0:s}/plot_{1:s}_detectedSourcesWithRefCatCounterpart.pdf".format(outDir,
                                                                                                   rootname)

                fig0.savefig(plotFilename0, bbox_inches='tight', dpi=300)

                print("Image and detected sources saved to {0:s}".format(plotFilename0))

                plt.close(fig=fig0)

                del fig0
                del axes0

                hasRefCat = df_hst1pass['hasRefCat']

                nRefCat = hasRefCat[hasRefCat].size

                if (nRefCat > 0):
                    delX = df_hst1pass['xPred'][hasRefCat] - df_hst1pass['xRef'][hasRefCat]
                    delY = df_hst1pass['yPred'][hasRefCat] - df_hst1pass['yRef'][hasRefCat]

                    xyMatches = np.vstack([df_hst1pass['X'], df_hst1pass['Y']]).T[hasRefCat]
                    residuals = np.vstack([delX, delY]).T

                    nData         = len(df_hst1pass)
                    nMatches      = hasRefCat[hasRefCat].size
                    matchFraction = float(nMatches) / float(nData)

                    shiftText = "{0:s} {1:d} {2:d} {3:f}".format(rootname, nData, nMatches, matchFraction)

                    for iii in range(coeffs.shape[0]):
                        for jjj in range(coeffs.shape[1]):
                            shiftText += " {0:.9e}".format(coeffs[iii, jjj])
                    shiftText += "\n"

                    f = open(shiftInfoFilename, 'a')
                    f.write(shiftText)
                    f.close()

                    del f

                    plotFilename1 = "{0:s}/plot_{1:s}_matchResidualsDistribution.pdf".format(outDir, rootname)

                    self._plotMatchResidualsDistribution(delX, delY, plotFilename1, rootname)

                    plotFilename2 = "{0:s}/plot_{1:s}_matchResidualsXY.pdf".format(outDir, rootname)

                    self._plotMatchResidualsXY(xyMatches, residuals, plotFilename2, rootname, xMax=naxis2)

                    df_hst1pass.drop(columns=['x_im', 'y_im']).to_csv(outTableFilename, index=False)

                    print("CSV table written to", outTableFilename)

                else:
                    print("NO MATCHES FOUND! File skipped...")

                    self._writeErrorMessage(errTableFilename,
                                            rootname+" ALL CHIPS: NO MATCHES FOUND. N_REFCAT = {0:d}".format(nRefCat))

                del df_hst1pass
                del hduList
                gc.collect()

                elapsedTime0 = time.time() - startTime0
                print("IMAGE DONE! Elapsed time:", convertTime(elapsedTime0))

            else:
                if os.path.exists(outTableFilename):
                    print("File not processed because output table has been found:", outTableFilename)
                if os.path.exists(errTableFilename):
                    print("File not processed because an error was found in the previous run:", errTableFilename)


        print("ALL DONE!")


    def _writeErrorMessage(self, filename, errorMessage):
        f = open(filename, 'a')
        f.write(errorMessage+ "\n")
        f.close()

        del f
        gc.collect()

    def _plotMatchResidualsXY(self, coordinates, residuals, plotFilename, rootname, markerSize=1, xMax=4096):
        ## Plotting match residuals
        nRows2 = 2
        nCols2 = 2

        xSize2 = 8
        ySize2 = 0.5 * nCols2 * xSize2 / float(nRows2)

        fig2, axes2 = plt.subplots(figsize=(xSize2, ySize2), nrows=nRows2, ncols=nCols2, sharey=True)

        xLabels = [r'$X_{\rm raw}$ [pix]', r'$Y_{\rm raw}$ [pix]']
        yLabels = [r'$\Delta U$ [pix]', r'$\Delta V$ [pix]']

        xMin = 0

        dX, dMX = 1000, 200
        dY, dMY = 2, 0.5

        for axis1 in range(acsconstants.NAXIS):
            for axis2 in range(acsconstants.NAXIS):
                thisCoords = coordinates[:, axis2]
                thisRes    = residuals[:, axis1]

                mean = np.nanmean(thisRes)
                stdDev = np.nanstd(thisRes)

                axes2[axis1, axis2].plot(thisCoords, thisRes, 'k.', markersize=markerSize)

                axes2[axis1, axis2].axhline(0, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                                            linestyle='--')

                axes2[axis1, axis2].axhline(mean, color='r', linestyle='-')

                resMin = mean - stdDev
                resMax = mean + stdDev

                axes2[axis1, axis2].fill_between([xMin, xMax], [resMin, resMin], y2=[resMax, resMax],
                                                 alpha=0.2)

                if (axis1 == 1):
                    axes2[axis1, axis2].set_xlabel(xLabels[axis2])

                if (axis2 == 0):
                    axes2[axis1, axis2].set_ylabel(yLabels[axis1])

                axes2[axis1, axis2].set_ylim(-self.max_sep_refined.value, +self.max_sep_refined.value)

                axes2[axis1, axis2].xaxis.set_major_locator(ticker.AutoLocator())
                axes2[axis1, axis2].xaxis.set_minor_locator(ticker.AutoMinorLocator())

                axes2[axis1, axis2].yaxis.set_major_locator(ticker.AutoLocator())
                axes2[axis1, axis2].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axCommons = plotting.drawCommonLabel('', '', fig2, xPad=0, yPad=0)

        axCommons.set_title('{0:s}'.format(rootname))

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        fig2.savefig(plotFilename, bbox_inches='tight', dpi=300)

        print("Matching residuals plot saved to {0:s}".format(plotFilename))

        plt.close(fig=fig2)

        del fig2
        del axes2

    def _plotMatchResidualsDistribution(self, delX, delY, plotFilename, rootname):
        xSize = 4
        ySize = xSize

        markerSize = 1

        fig, ax = plt.subplots(figsize=(xSize, ySize), rasterized=True)

        ax.set_aspect('equal')

        ax.plot(delX, delY, 'k.', markersize=markerSize)

        ax.axhline()
        ax.axvline()

        ax.set_xlabel(r'$\Delta U$ [pix]')
        ax.set_ylabel(r'$\Delta V$ [pix]')

        ax.set_xlim(-self.max_sep_refined.value, +self.max_sep_refined.value)
        ax.set_ylim(-self.max_sep_refined.value, +self.max_sep_refined.value)

        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.set_title('{0:s}'.format(rootname))

        fig.savefig(plotFilename, bbox_inches='tight', dpi=300)

        print("Match residuals plot saved to {0:s}".format(plotFilename))

        plt.close(fig=fig)

        del fig
        del ax