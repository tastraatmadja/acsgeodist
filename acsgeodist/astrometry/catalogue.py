import gc
import glob
import os
import time

from astropy import coordinates
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.gaia import Gaia
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import linalg, sparse, stats
from sklearn.linear_model import LinearRegression

from acsgeodist import acsconstants
from acsgeodist.tools import astro, coords, reader, sip, stat
from acsgeodist.tools.time import convertTime

N_WRITE = 100

MIN_N_OBS = 3

ONE_SIGMA   = 68.2689492
THREE_SIGMA = 99.7300204

N_SIGMA_THRESHOLD = np.array([0.0, 3.0, 5.0])

N_PARAMS = np.array([2, 4, 5])

PM = np.full_like(N_PARAMS, 1.0 / 3.0, dtype=float)

LOG_PM = np.log(PM)

HEIGHT  = 0.5 * u.deg
WIDTH   = HEIGHT

MIN_RUWE = 0.8
MAX_RUWE = 1.2

class SourceCollector:
    def __init__(self, wcsRef, c0, qMin=None, qMax=None, min_t_exp=None, min_n_stars=None, max_pos_targs=None, max_sep=None,
                 min_n_epoch=None, min_n_obs=None, min_t_range=1.0, tRef=Time(2016.00, format='jyear', scale='tcb')):
        self.qMin              = 1.e-6
        self.qMax              = 0.5
        self.min_t_exp         = 99.0
        self.min_n_stars       = 10000
        self.max_pos_targs     = np.inf
        self.max_sep           = 1.0 * u.pix
        self.min_n_epoch       = 5
        self.min_n_obs         = 10
        self.min_t_range       = 1.0
        self.n_sigma_threshold = deepcopy(N_SIGMA_THRESHOLD)
        self.tRef              = deepcopy(tRef)
        self.wcsRef            = deepcopy(wcsRef)
        self.c0                = deepcopy(c0)
        self.pqr0              = coords.getNormalTriad(self.c0)
        self.reg               = LinearRegression(fit_intercept=False, copy_X=False)

        if (qMin is not None):
            self.qMin = qMin
        if (qMax is not None):
            self.qMax = qMax
        if (min_t_exp is not None):
            self.min_t_exp = min_t_exp
        if (min_n_stars is not None):
            self.min_n_stars = min_n_stars
        if (max_pos_targs is not None):
            self.max_pos_targs = max_pos_targs
        if (max_sep is not None):
            self.max_sep = max_sep
        if (min_n_epoch is not None):
            self.min_n_epoch = min_n_epoch
        if (min_n_obs is not None):
            self.min_n_obs = min_n_obs
        if (min_t_range is not None):
            self.min_t_range = min_t_range

    def collectIndividualSources(self, fitSummaryTableFilenames, residsFilenames, pOrder=5, outDir='./'):
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        df_fitResults = reader.readFitResults(fitSummaryTableFilenames, pOrder)

        df_fitResults['i']   = [rootname.replace('_flc', '')[0] for rootname in df_fitResults['rootname']]
        df_fitResults['ppp'] = [rootname.replace('_flc', '')[1:4] for rootname in df_fitResults['rootname']]
        df_fitResults['ss']  = [rootname.replace('_flc', '')[4:6] for rootname in df_fitResults['rootname']]
        df_fitResults['oo']  = [rootname.replace('_flc', '')[6:8] for rootname in df_fitResults['rootname']]
        df_fitResults['t']   = [rootname.replace('_flc', '')[8] for rootname in df_fitResults['rootname']]

        tBins = []
        tMins = []
        tMaxs = []

        nImages = 0
        for ppp in df_fitResults.loc[df_fitResults['chip'] == 1, ['ppp']].value_counts().sort_index().index:
            ppp = ppp[0]

            for ss in df_fitResults.loc[
                (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp), ['ss']].value_counts().sort_index().index:
                ss = ss[0]

                selected_time = df_fitResults.loc[
                    (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp) & (df_fitResults['ss'] == ss), ['time']]

                tMin = selected_time.min().values[0]
                tMax = selected_time.max().values[0]
                dt = (tMax - tMin) * 365.25 * 24 * u.hour

                tMins.append(tMin)
                tMaxs.append(tMax)

                nImages += selected_time.size

                print(ppp, ss, selected_time.size, selected_time.min().values, selected_time.max().values, dt)
            print()

        tMins = np.sort(np.array(tMins))
        tMaxs = np.sort(np.array(tMaxs))

        tBins = np.hstack([tMins, tMaxs[-1] + 1.0 / 365.25])

        print("TOTAL_NUMBER_OF_IMAGES:", nImages)
        print("NUMBER_OF_TIME_BINS:", tBins.size)

        df_fitResults['epochID'] = np.digitize(df_fitResults['time'].values, tBins, right=False) - 1

        df_timeBins = pd.DataFrame(
            data={'epochID': np.arange(0, tBins.size - 1, dtype=int), 'tMin': tBins[0:-1], 'tMax': tBins[1:]})

        df_timeBins.to_csv('{0:s}/time_bins.csv'.format(outDir))

        nFiles = len(residsFilenames)

        print("NUMBER OF FILES:", nFiles)

        startTimeAll = time.time()

        nWrite = 10000

        names = []
        catalog = None
        obsData = []
        nObsData = []
        nEpochs = []
        deltaT   = []
        epochIDs = []

        fileDone = []

        ## TODO: This part need to be re-written once all is done
        fileDoneFilename = '{0:s}/fileDone.csv'.format(outDir)
        if os.path.exists(fileDoneFilename):
            df_fileDone = pd.read_csv(fileDoneFilename)
            fileDone = df_fileDone['rootname'].values.tolist()

        nObsDataFilename = '{0:s}/nObsData.csv'.format(outDir)
        if os.path.exists(nObsDataFilename):
            df_nObs = pd.read_csv(nObsDataFilename)

            names    = df_nObs['sourceID']
            catalog  = SkyCoord(ra=df_nObs['alpha'].values * u.deg, dec=df_nObs['delta'].values * u.deg,
                                pm_ra_cosdec=df_nObs['pm_ra'].values * u.mas / u.yr,
                                pm_dec=df_nObs['pm_dec'].values * u.mas / u.yr,
                                obstime=Time(df_nObs['epoch'].values, format='jyear', scale='tcb'),
                                distance=1, frame='icrs')
            nObsData = df_nObs['nObs'].values.tolist()
            nEpochs  = df_nObs['nEpochs'].values.tolist()
            deltaT   = df_nObs['dT'].values.tolist()

        obsDataFilenameAll = '{0:s}/allSources_individualObservations.h5'.format(outDir)

        nRowsTotal   = 0
        thisFileDone = []
        iterate      = range(nFiles)
        timeRange    = 0.0
        prevTime     = 0.0
        for i in iterate:
            startTime = time.time()

            residsFile = residsFilenames[i]

            rootname = os.path.basename(residsFile).split('_')[0]

            df_coeffs = df_fitResults[(df_fitResults['rootname'].str.contains(rootname))]

            tObs     = Time(df_coeffs['time'].values[0], format='decimalyear', scale='ut1')
            tExp     = df_coeffs['tExp'].values[0]
            vaFactor = df_coeffs['vaFactor'].values[0]
            nStars   = df_coeffs['nStars'].values[0]
            epochID  = df_coeffs['epochID'].values[0]
            posTarg1 = df_coeffs['posTarg1'].values[0]
            posTarg2 = df_coeffs['posTarg2'].values[0]
            posTarg  = np.sqrt(posTarg1**2 + posTarg2**2)

            process = (tExp > self.min_t_exp) and (nStars > self.min_n_stars) and (posTarg <= self.max_pos_targs)

            print(i, os.path.basename(residsFile), rootname, epochID, tExp, nStars, posTarg, "PROCESS:", process, end='')

            if process:
                done = False

                if (rootname in fileDone):
                    done = True

                print(" DONE:", done, end='')

                if (not done):
                    print('.')
                    print("Processing {0:s}... ".format(rootname), end='')

                    df_resids = pd.read_csv(residsFile)

                    df_resids['time_tcb'] = tObs.tcb.jyear
                    df_resids['vaFactor'] = vaFactor
                    df_resids['rootname'] = rootname
                    df_resids['epochID']  = epochID

                    xi  = df_resids['xi'].values  * df_resids['vaFactor'].values
                    eta = df_resids['eta'].values * df_resids['vaFactor'].values

                    argsel = np.argwhere(
                        ~np.isnan(xi) & ~np.isnan(eta) & (df_resids['q'] > self.qMin) & (df_resids['q'] <= self.qMax)).flatten()

                    ## c = coords.getCelestialCoordinatesFromNormalCoordinates(xi[argsel], eta[argsel], self.c0, frame='icrs')
                    alpha, delta = self.wcsRef.wcs_pix2world(xi[argsel], eta[argsel], 1)

                    c = SkyCoord(ra=alpha * u.deg, dec=delta * u.deg, frame='icrs')

                    sourceIDs = list(df_resids.iloc[argsel]['sourceID'])
                    obstime   = df_resids.iloc[argsel]['time_tcb'].values

                    ## Once we get the list of sourceIDs we want and their corresponding indices, we drop the sourceID column to save space
                    df_resids = df_resids.drop(columns=['sourceID'])

                    ## Populate the reference catalog, names list, and observation data with the data from the first plate
                    if (len(names) <= 0):
                        names    = deepcopy(sourceIDs)
                        catalog  = SkyCoord(ra=c.ra, dec=c.dec, pm_ra_cosdec=np.zeros_like(c.ra.value) * u.mas / u.yr,
                                            pm_dec=np.zeros_like(c.dec.value) * u.mas / u.yr,
                                            obstime=Time(obstime, format='jyear', scale='tcb'),
                                            distance=1, frame='icrs')
                        obsData  = [pd.DataFrame([df_resids.iloc[index]]) for index in argsel]
                        nObsData = np.ones(len(sourceIDs), dtype=int).tolist()
                        nEpochs  = np.ones(len(sourceIDs), dtype=int).tolist()
                        deltaT   = np.zeros(len(sourceIDs), dtype=float).tolist()
                        epochIDs = np.full((len(sourceIDs), 1), epochID, dtype=int).tolist()
                        prevTime = tObs.tcb.jyear
                    else:
                        catalog_obs = catalog.apply_space_motion(tObs)

                        ## Now we do cross-matching
                        idx, d2d, _ = c.match_to_catalog_sky(catalog_obs)

                        d2d_pix = (d2d / acsconstants.ACS_PLATESCALE).decompose()

                        ## Select matching sources
                        matches = d2d_pix < self.max_sep

                        ra_new   = []
                        dec_new  = []
                        time_new = []

                        ## Cycle over the rows, matches, and the indices of the reference catalog
                        for ii, (index, match, matchIdx) in enumerate(zip(argsel, matches, idx)):
                            ## Grab the corresponding row
                            row = pd.DataFrame([df_resids.iloc[index]])

                            ## If it's a match, add to the reference catalog and the observation data
                            if match:
                                ## Concatenate the new row to the corresponding matching row(s).
                                obsData[matchIdx] = pd.concat([obsData[matchIdx], row], ignore_index=True)
                                obsData[matchIdx].reset_index()

                                deltaT[matchIdx] = (obsData[matchIdx]['time_tcb'].max() -
                                                    obsData[matchIdx]['time_tcb'].min())

                                nObsData[matchIdx] += 1

                                ## Extend the list here so as not introduce a NoneType later
                                epochIDs[matchIdx].extend(obsData[matchIdx].epochID.unique())

                                epochIDs[matchIdx] = list(set(epochIDs[matchIdx]))

                                nEpochs[matchIdx] = len(epochIDs[matchIdx])

                            ## Otherwise this is a new observation and create a new entry for this observation
                            else:
                                names.append(sourceIDs[ii])  ## Append the source Id to the list of names
                                obsData.append(row)  ## Append the row to list of dataframe
                                nObsData.append(1)
                                nEpochs.append(1)
                                epochIDs.append([epochID])
                                deltaT.append(0.0)

                                ra_new.append(c[ii].ra.deg)
                                dec_new.append(c[ii].dec.deg)
                                time_new.append(tObs.tcb.jyear)

                        ra_new   = np.array(ra_new)
                        dec_new  = np.array(dec_new)
                        time_new = np.array(time_new)

                        ## Update the catalogs coordinates
                        ## catalog = coordinates.concatenate([catalog, new_catalog])
                        catalog = self._concatenateCatalogues(catalog,
                                                              SkyCoord(ra=ra_new * u.deg, dec=dec_new * u.deg,
                                                                       pm_ra_cosdec=np.zeros_like(ra_new) * u.mas / u.yr,
                                                                       pm_dec=np.zeros_like(dec_new) * u.mas / u.yr,
                                                                       obstime=Time(time_new, format='jyear', scale='tcb'),
                                                                       distance=1, frame='icrs'))

                    nRowsTotal += xi.size

                    timeRange = tObs.tcb.jyear - prevTime

                    calculatePM = False
                    if (timeRange > 1.0):
                        prevTime = tObs.tcb.jyear

                        calculatePM = True

                    catalog = self._generateNewCatalog(catalog, obsData, np.array(nObsData), np.array(deltaT),
                                                       calculate_proper_motion=calculatePM)

                    print(len(names), len(catalog), len(obsData), len(deltaT), nRowsTotal, "{0:0.3}".format(timeRange),
                          end='')
                    thisFileDone.append(rootname)

                    elapsedTime = time.time() - startTime

                    print(" DONE PROCESSING {0:s}! Elapsed time: {1:s}".format(rootname, convertTime(elapsedTime)))
                else:
                    print(". File has been worked out. Skipping file...")
            else:
                print(".")

            if ((((i % nWrite) == 0) and (i > 0)) or (i == iterate[-1])):
                if ((len(names) > 0) and (len(obsData) > 0)):
                    print("Writing observation data and flushing out the dictionary...")

                    startTimeFlush = time.time()

                    print("NUMBER OF SOURCES SO FAR:", len(names))

                    mode = 'w'

                    if os.path.exists(obsDataFilenameAll):
                        mode = 'a'

                    store = pd.HDFStore(obsDataFilenameAll, mode)

                    for index, name, df in enumerate(zip(names, obsData)):
                        store.append(name, df, index=False, append=True, format='table', complevel=None)

                        ## Drop the whole rows once they're flushed to save memory, but keep the dataframe header so later
                        ## we can load them with new rows
                        obsData[index] = obsData[index][0:0]

                    store.close()

                    ## Expand the list of files done after the collected data have been flushed
                    fileDone.extend(thisFileDone)

                    thisFileDone.clear()

                    ## Writing the number of observations for individual sources
                    df_nObs = pd.DataFrame.from_dict({'sourceID': names,
                                                      'alpha': catalog.ra.value,
                                                      'delta': catalog.dec.value,
                                                      'pm_ra': catalog.pm_ra_cosdec.value,
                                                      'pm_dec': catalog.pm_dec.value,
                                                      'nObs': nObsData, 'nEpochs': nEpochs, 'dT': deltaT})
                    df_nObs.to_csv(nObsDataFilename, index=False)

                    df_fileDone = pd.DataFrame.from_dict({'rootname': fileDone})
                    df_fileDone.to_csv(fileDoneFilename, index=False)

                    gc.collect()

                    elapsedTime = time.time() - startTimeFlush
                    print("DONE FLUSHING! Elapsed time:", convertTime(elapsedTime))


        elapsedTime = time.time() - startTimeAll

        print("ALL DONE! Elapsed time:", convertTime(elapsedTime))

        return obsDataFilenameAll, nObsDataFilename

    def calculateAstrometricSolutions(self, obsDataFilename, nObsDataFilename, refPix_x=None, refPix_y=None,
                                      maxNModel=2, nIter=20, minNEpoch=None, minNObs=None, minDT=None,
                                      nSigmaThreshold=None, tRef=None, outDir='./', imageDir=None, reOrientFrame=True,
                                      nPrint=10000, acceptMaximalSolution=False):

        if (tRef is not None):
            self.tRef = deepcopy(tRef)

        properMotionFilename = '{0:s}/PPMPLXCatalogue_tRef{1:0.1f}.csv'.format(outDir, self.tRef.tcb.jyear)

        if (not os.path.exists(properMotionFilename)):
            triu_indices = []
            for model in range(maxNModel):
                triu_indices.append(np.triu_indices(N_PARAMS[model]))

            triu_indices_all = np.triu_indices(N_PARAMS[-1])

            df_nObs = pd.read_csv(nObsDataFilename)

            df_nObs[['nObs', 'nEpochs']].describe()

            if (minDT is not None):
                self.min_t_range = minDT

            if (minNEpoch is not None):
                self.min_n_epoch = minNEpoch

            selection = (df_nObs['nEpochs'] >= self.min_n_epoch) & (df_nObs['dT'] >= self.min_t_range)

            if (minNObs is not None):
                self.min_n_obs = minNObs

                selection = (df_nObs['nObs'] >= self.min_n_obs) & (df_nObs['dT'] >= self.min_t_range)

            if (nSigmaThreshold is not None) and hasattr(nSigmaThreshold, '__iter__'):
                self.n_sigma_threshold    = deepcopy(nSigmaThreshold)
                self.n_sigma_threshold[0] = 0.0

            if (tRef is not None):
                self.tRef = deepcopy(tRef)

            print("SELECTED {0:d} SOURCES!".format(selection[selection].shape[0]))
            print("P_VALUE THRESHOLD FOR MODEL SELECTION:", self.n_sigma_threshold, "SIGMA")

            columns = ['sourceID', 'tMin', 'tMax', 'nObs', 'nEpoch', 'meanMag', 'medianQ', 'bestModel', 'LR', 'pVal',
                       'nSigma', 'pMD', 'rms', 'nEff', 'xi0', 'eta0', 'pm_xi', 'pm_eta', 'parallax']

            N_COVS = 0
            for i in range(N_PARAMS[-1]):
                for j in range(i, N_PARAMS[-1]):
                    N_COVS += 1
                    columns.append('cov_{0:d}{1:d}'.format(i, j))

            storeIn = pd.HDFStore(obsDataFilename, 'r')

            data = []

            startTimeAll = time.time()

            for i, sourceID in enumerate(df_nObs[selection]['sourceID'].values):
                if (sourceID in storeIn):
                    df = storeIn[sourceID]

                    refCatID = df['refCatID'].iloc[0]

                    nEpochs = df['epochID'].unique().size

                    xi  = (df['xi'] * df['vaFactor']).values
                    eta = (df['eta'] * df['vaFactor']).values
                    t   = Time(df['time_tcb'].values, format='jyear', scale='tcb')

                    weights_init = np.repeat(df['weights'].values, 2)

                    if (0.5 * np.nansum(weights_init) < MIN_N_OBS):
                        weights_init = np.repeat(np.ones_like(xi), 2)

                    '''
                    if (np.sum(weights) <= 0.0):
                        print(i, sourceID, refCatID, nEpochs, nObs)
                    ''';

                    '''
                    if (i == 59714):
                        print(df['weights'])
                        print(weights_init)
                    ''';

                    nObs = t.size

                    ## print(i, sourceID, refCatID, nEpochs, nObs)

                    if (maxNModel > 2) and (imageDir is not None):
                        rootnames = df['rootname']

                        sptFilenames = ['{0:s}/mastDownload/HST/{1:s}/{1:s}_spt.fits'.format(imageDir, rootname) for
                                        rootname in rootnames]
                        flcFilenames = ['{0:s}/mastDownload/HST/{1:s}/{1:s}_flc.fits'.format(imageDir, rootname) for
                                        rootname in rootnames]

                        pv, tObs = astro.getHSTPosVelTime(sptFilenames, flcFilenames)

                        X = astro.getAstrometricModels(tObs, self.tRef, maxNModel=maxNModel, pqr0=self.pqr0, pv=pv)

                    else:
                        X = astro.getAstrometricModels(t, self.tRef, maxNModel=maxNModel, pqr0=self.pqr0)

                    y = np.zeros((2 * nObs, 1))

                    y[0::2, 0] = xi
                    y[1::2, 0] = eta

                    chiSq = np.zeros(maxNModel)
                    lnLL = np.zeros(maxNModel)
                    LR = np.zeros(maxNModel)
                    lnPMD = np.zeros(maxNModel)
                    pVal = np.zeros(maxNModel)
                    nSigma = np.zeros(maxNModel)
                    nuEff  = np.zeros(maxNModel)
                    prevNPars = 0
                    prevLnLL = -np.inf
                    lnPSum = -np.inf
                    solutions = []
                    covs = []
                    rms = []
                    nEff = []
                    for model in range(maxNModel):
                        astro_solution, weights, res = self._solveAstrometry(X[model], y, weights_init, nIter=nIter)

                        W = sparse.diags(weights)

                        solutions.append(astro_solution)

                        XTWX = X[model].T @ W @ X[model]

                        XTWXInv =  linalg.inv(XTWX)

                        H = X[model] @ XTWXInv @ X[model].T @ W

                        nuEff[model] = np.nansum(weights) - np.trace(H)

                        mse = (res.T @ W @ res)[0, 0] / np.sum(weights)

                        covs.append(mse * XTWXInv)

                        rms.append(np.sqrt(mse))

                        nEff.append(0.5 * np.nansum(weights))

                        resStdDev = np.sqrt(np.nansum(weights * res**2) / np.nansum(weights) / nuEff[model])

                        ## covs.append(list(cov[triu_indices[model]].flatten()))

                        chiSq[model] = stat.getChiSq(res, weights, np.ones_like(weights))
                        lnLL[model]  = stat.getLnLL(res, weights)

                        ## print(X[model].shape, weights.shape, np.nansum(weights), resStdDev, np.sqrt(mse), mse, chiSq[model], np.nansum(res**2), np.nansum(weights), np.trace(H), nuEff[model], chiSq[model] / nuEff[model])

                        lnPMD[model] = stat.getLnPMD(N_PARAMS[model], res, LOG_PM[model], weights=weights)

                        LR[model] = -2.0 * (prevLnLL - lnLL[model])

                        pVal[model] = stats.chi2.sf(LR[model], N_PARAMS[model] - prevNPars)

                        nSigma[model] = stats.norm.ppf(1.0 - 0.5 * pVal[model])

                        if acceptMaximalSolution:
                            if np.isfinite(rms[model]):
                                bestModel = model
                        elif (nSigma[model] > self.n_sigma_threshold[model]):
                            bestModel = model

                        ## print(model, astro_solution, np.sqrt(np.nanmean(res**2)), lnLL[model], LR[model], pVal[model], nSigma[model], lnPMD[model])

                        lnPSum = stat.addLnProb(lnPSum, lnPMD[model])

                        prevLnLL = lnLL[model]
                        prevNPars = N_PARAMS[model]

                    pMD = np.exp(lnPMD - lnPSum)

                    ## bestModel = np.argmax(lnPMD)

                    bestSolution = np.zeros(N_PARAMS[-1])
                    bestSolution[0:solutions[bestModel].size] = solutions[bestModel]

                    bestCov = np.zeros((N_PARAMS[-1], N_PARAMS[-1]))

                    bestCov[triu_indices[bestModel]] = covs[bestModel][triu_indices[bestModel]]

                    bestCov = bestCov[triu_indices_all].flatten()

                    bestRMS = rms[bestModel]

                    bestNEff = nEff[bestModel]

                    bestChiSq    = chiSq[bestModel]
                    bestRedChiSq = chiSq[bestModel] / nuEff[bestModel]

                    ## print("BEST_MODEL:", bestModel, nSigma[bestModel], pMD[bestModel], bestSolution)
                    ## print(bestSolution.shape, bestCov.shape)

                    tMin = df['time_tcb'].min()
                    tMax = df['time_tcb'].max()

                    df['flux'] = 10.00 ** (-0.4 * df['W'])

                    meanMag = -2.5 * np.log10(df['flux'].mean())
                    medianQ = df['q'].median()

                    dataLine = [sourceID, tMin, tMax, nObs, nEpochs, meanMag, medianQ, bestModel + 1, LR[bestModel],
                                pVal[bestModel], nSigma[bestModel], pMD[bestModel], bestRMS, bestNEff]
                    dataLine.extend(list(bestSolution))
                    dataLine.extend(list(bestCov))

                    data.append(dataLine)

                if ((((i+1) % nPrint) == 0) and (i > 0)):
                    elapsedTime = time.time() - startTimeAll
                    print("DONE PROCESSING {0:d} SOURCES! ELAPSED TIME: {1:s}".format(i + 1, convertTime(elapsedTime)))

                '''
                if (i > 500):
                    break
                ''' ;

            storeIn.close()

            df_ppm = pd.DataFrame(data, columns=columns)

            df_ppm = df_ppm.sort_values(by='xi0', ascending=True, ignore_index=True)

            df_ppm.to_csv(properMotionFilename, index=False, mode='w')

            elapsedTime = time.time() - startTimeAll

            print("ALL DONE! Elapsed time:", convertTime(elapsedTime))
        else:
            print("PROPER MOTION CATALOGUE EXISTS ALREADY!")

        df_ppm = pd.read_csv(properMotionFilename)

        df_new = pd.DataFrame(columns=['m', 'x', 'y', 'pm_x', 'pm_y'])

        df_new['m']    = df_ppm['meanMag']
        df_new['x']    = df_ppm['xi0']
        df_new['y']    = df_ppm['eta0']
        df_new['pm_x'] = df_ppm['pm_xi']
        df_new['pm_y'] = df_ppm['pm_eta']

        propMotionFilenameGDR3 = '{0:s}/PPMPLXCatalogue_tRef{1:0.1f}_GDR3Corr.txt'.format(outDir,
                                                                                          self.tRef.tcb.jyear)

        if reOrientFrame:
            if (not os.path.exists(propMotionFilenameGDR3)):
                print("RE-ORIENTING CELESTIAL FRAME USING GAIA DR3 STARS...")
                print("QUERYING THE CATALOGUE...")
                Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
                Gaia.ROW_LIMIT = -1

                g = Gaia.query_object_async(coordinate=self.c0, width=WIDTH, height=HEIGHT)

                gaia_selection = (g['ruwe'] >= MIN_RUWE) & (g['ruwe'] <= MAX_RUWE)

                g = g[gaia_selection]

                print("FOUND {0:d} GAIA SOURCES!".format(len(g)))

                gdr3_id = np.array(['GDR3_{0:d}'.format(sourceID) for sourceID in g['SOURCE_ID']], dtype=str)

                c_gdr3 = SkyCoord(ra=g['ra'].value * u.deg, dec=g['dec'].value * u.deg,
                                  pm_ra_cosdec=g['pmra'].value * u.mas / u.yr, pm_dec=g['pmdec'].value * u.mas / u.yr,
                                  obstime=Time(g['ref_epoch'].value, format='jyear', scale='tcb'))

                print("GETTING THE SHIFT AND ROTATION MATRIX...")

                xi0_gdr3, eta0_gdr3 = coords.getNormalCoordinates(c_gdr3, coords.getNormalTriad(self.c0))

                xi0_gdr3_pix  = (xi0_gdr3 / acsconstants.ACS_PLATESCALE).decompose().value
                eta0_gdr3_pix = (eta0_gdr3 / acsconstants.ACS_PLATESCALE).decompose().value

                gdr3Sources = df_ppm['sourceID'].str.contains('GDR3')

                df_ppm['xi0_gdr3'] = np.nan
                df_ppm['eta0_gdr3'] = np.nan

                hasGDR3_indices_original = np.argwhere(gdr3Sources).flatten()

                if (hasGDR3_indices_original.size > 0):
                    hasGDR3_indices = []
                    foundIndices = []
                    for index in hasGDR3_indices_original:
                        sourceID = df_ppm.iloc[index]['sourceID']
                        matchIndex = np.argwhere(gdr3_id == sourceID).flatten()[0]
                        bestModel = df_ppm.iloc[index]['bestModel']

                        if (not matchIndex in foundIndices):
                            if (bestModel < 3):
                                '''
                                print(index, sourceID, matchIndex, gdr3_id[matchIndex], df_ppm.iloc[index]['xi0'],
                                      xi0_gdr3_pix[matchIndex], df_ppm.iloc[index]['eta0'], eta0_gdr3_pix[matchIndex])
                                ''';

                                df_ppm.at[index, 'xi0_gdr3']  = xi0_gdr3_pix[matchIndex]
                                df_ppm.at[index, 'eta0_gdr3'] = eta0_gdr3_pix[matchIndex]

                                hasGDR3_indices.append(index)
                            foundIndices.append(matchIndex)
                else:
                    gdr3_id = np.array(['GDR3_{0:d}'.format(sourceID) for sourceID in g['SOURCE_ID']], dtype=str)

                    xi0_pix  = (df_ppm['xi0'].values * u.pix * acsconstants.ACS_PLATESCALE).decompose()
                    eta0_pix = (df_ppm['eta0'].values * u.pix * acsconstants.ACS_PLATESCALE).decompose()

                    idx, sep, _ = coords.getCelestialCoordinatesFromNormalCoordinates(xi0_pix, eta0_pix, self.c0).match_to_catalog_sky(c_gdr3)

                    ## c_refCat.match_to_catalog_sky(c_gdr3)

                    sep_pix = sep.to(u.mas) / acsconstants.ACS_PLATESCALE

                    selection_gdr3 = sep_pix < 3.0 * u.pix

                    hasGDR3_indices = np.argwhere(selection_gdr3).flatten()

                    df_ppm.loc[hasGDR3_indices, 'sourceID']  = gdr3_id[idx[selection_gdr3]]
                    df_ppm.loc[hasGDR3_indices, 'xi0_gdr3']  = xi0_gdr3_pix[idx[selection_gdr3]]
                    df_ppm.loc[hasGDR3_indices, 'eta0_gdr3'] = eta0_gdr3_pix[idx[selection_gdr3]]

                    nMatch = selection_gdr3[selection_gdr3].size

                    print("Found {0:d} GDR3 matches!".format(nMatch))

                hasGDR3_indices = np.array(hasGDR3_indices)

                xi0, eta0 = df_ppm.iloc[hasGDR3_indices]['xi0'], df_ppm.iloc[hasGDR3_indices]['eta0']
                xi0_gdr3, eta0_gdr3 = df_ppm.iloc[hasGDR3_indices]['xi0_gdr3'], df_ppm.iloc[hasGDR3_indices]['eta0_gdr3']

                X, scaler       = sip.buildModel(xi0, eta0, 1)
                XAll, scalerAll = sip.buildModel(df_ppm['xi0'], df_ppm['eta0'], 1)

                for i in range(acsconstants.NAXIS):
                    if (i == 0):
                        y = xi0_gdr3
                    elif (i == 1):
                        y = eta0_gdr3

                    self.reg.fit(X, y)

                    print(self.reg.coef_)

                    new_coords = self.reg.predict(XAll)
                    new_pm = df_ppm['pm_xi'] * self.reg.coef_[1] + df_ppm['pm_eta'] * self.reg.coef_[2]

                    if ((refPix_x is not None) and (refPix_y is not None)):
                        ## Add 1 because in pixel space, coordinates starts from 1, NOT 0!
                        if (i == 0):
                            df_new['x']    = refPix_x - new_coords + 1.0
                            df_new['pm_x'] = -new_pm
                        elif (i == 1):
                            df_new['y']    = refPix_y + new_coords + 1.0
                            df_new['pm_y'] = new_pm
                    else:
                        if (i == 0):
                            df_new['x']    = new_coords
                            df_new['pm_x'] = new_pm
                        elif (i == 1):
                            df_new['y']    = new_coords
                            df_new['pm_y'] = new_pm

                print("CELESTIAL FRAME ALREADY RE-ORIENTED USING GAIA DR3 STARS!")

        else:
            propMotionFilenameGDR3 = '{0:s}/PPMPLXCatalogue_tRef{1:0.1f}_noGDR3Corr.txt'.format(outDir,
                                                                                                self.tRef.tcb.jyear)
        if (not os.path.exists(propMotionFilenameGDR3)):
            df_new.index += 1

            df_new.to_csv(propMotionFilenameGDR3, sep=' ', header=False, index=True)

        return properMotionFilename, propMotionFilenameGDR3

    def _concatenateCatalogues(self, catalog1, catalog2):
        ra     = np.hstack([catalog1.ra.deg, catalog2.ra.deg]) * u.deg
        dec    = np.hstack([catalog1.dec.deg, catalog2.dec.deg]) * u.deg
        pm_ra  = np.hstack([catalog1.pm_ra_cosdec.value, catalog2.pm_ra_cosdec.value]) * u.mas / u.yr
        pm_dec = np.hstack([catalog1.pm_dec.value, catalog2.pm_dec.value]) * u.mas / u.yr
        time   = np.hstack([catalog1.obstime.tcb.jyear, catalog2.obstime.tcb.jyear])

        return SkyCoord(ra=ra, dec=dec, pm_ra_cosdec=pm_ra, pm_dec=pm_dec,
                        obstime=Time(time, format='jyear', scale='tcb'), distance=1, frame='icrs')


    def _getMedianMeasurements(self, df):
        xi   = np.nanmedian((df['xi']  * df['vaFactor']).values)
        eta  = np.nanmedian((df['eta'] * df['vaFactor']).values)
        time = np.nanmedian(df['time_tcb'].values)

        alpha, delta = self.wcsRef.wcs_pix2world(xi, eta, 1)

        return alpha, delta, time

    def _solveAstrometry(self, X, y, w, nIter=20):
        astro_solution = np.zeros(X.shape[1])
        res            = np.zeros(X.shape[0])
        for iter in range(nIter):
            self.reg.fit(X, y, sample_weight=w)

            astro_solution = self.reg.coef_[0]

            res = y - self.reg.predict(X)

            mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(res.reshape((-1, 2)), w[::2])

            z = stat.getMahalanobisDistances(res.reshape((-1, 2)), mean, np.linalg.inv(cov))

            ## We now use the z statistics to re-calculate the weights
            w = np.repeat(stat.wdecay(z), 2)

        return astro_solution, w, res

    def _generateNewCatalog(self, catalog, obsData, nObs, dt, calculate_proper_motion=True, median=True):
        alpha  = catalog.ra.value
        delta  = catalog.dec.value
        pm_ra  = catalog.pm_ra_cosdec.value
        pm_dec = catalog.pm_dec.value
        time   = catalog.obstime.tcb.jyear

        if calculate_proper_motion:
            process1 = (nObs >= self.min_n_obs) & (dt >= self.min_t_range)
            indices1 = np.argwhere(process1).flatten()

            process2 = ~process1 & (nObs > 1)
            indices2 = np.argwhere(process2).flatten()

            for index in indices1:
                xi  = (obsData[index]['xi'] * obsData[index]['vaFactor']).values
                eta = (obsData[index]['eta'] * obsData[index]['vaFactor']).values
                t   = Time(obsData[index]['time_tcb'].values, format='jyear', scale='tcb')

                X = astro.getAstrometricModels(t, self.tRef, maxNModel=2, pqr0=self.pqr0)[1]
                y = np.zeros((X.shape[0], 1))

                y[0::2, 0] = xi
                y[1::2, 0] = eta

                w = np.repeat(obsData[index]['weights'].values, 2)

                if 0.5 * np.nansum(w) < self.min_n_obs:
                    w = np.ones_like(w)

                astro_solution, _, _ = self._solveAstrometry(X, y, w, nIter=5)

                alpha[index], delta[index]  = self.wcsRef.wcs_pix2world(astro_solution[0], astro_solution[1], 1)
                pm_ra[index], pm_dec[index] = -astro_solution[2] * acsconstants.ACS_PLATESCALE.value, astro_solution[
                    3] * acsconstants.ACS_PLATESCALE.value
                time[index] = self.tRef.tcb.jyear

            for index in indices2:
                alpha[index], delta[index], time[index] = self._getMedianMeasurements(obsData[index])

        elif median:
            for index in np.argwhere(nObs > 1).flatten():
                alpha[index], delta[index], time[index] = self._getMedianMeasurements(obsData[index])

        else:
            for index in np.argwhere(nObs > 1).flatten():
                xi  = np.nanmean(obsData[index]['xi'].values  * obsData[index]['vaFactor'].values)
                eta = np.nanmean(obsData[index]['eta'].values * obsData[index]['vaFactor'].values)

                alpha[index], delta[index] = self.wcsRef.wcs_pix2world(xi, eta, 1)

                time[index] = np.nanmean(obsData[index]['time_tcb'])

        return SkyCoord(ra=alpha * u.deg, dec=delta * u.deg, pm_ra_cosdec=pm_ra * u.mas / u.yr,
                        pm_dec=pm_dec * u.mas / u.yr, obstime=Time(time, format='jyear', scale='tcb'), distance=1,
                        frame='icrs')

class CrossMatcher:
    def __init__(self, referenceCatalog, referenceWCS, tRef0, max_sep=None):
        self.refCat  = deepcopy(referenceCatalog)
        self.wcsRef  = deepcopy(referenceWCS)
        self.tRef0   = deepcopy(tRef0)
        self.max_sep = 1.0 * u.pix

        ## Now we take the normal triad pqr_0 of the reference coordinate
        self.pqr0 = coords.getNormalTriad(SkyCoord(ra=self.wcsRef.wcs.crval[0] * u.deg,
                                                   dec=self.wcsRef.wcs.crval[1] * u.deg, frame='icrs'))

        if (max_sep is not None):
            self.max_sep = max_sep

    def crossMatch(self, hst1passFiles, imageFilenames, outDir='./'):
        for i, (hst1passFile, imageFilename) in enumerate(zip(hst1passFiles, imageFilenames)):
            print(i, os.path.basename(hst1passFile))

            hduList = fits.open(imageFilename)

            baseFilename = hduList[0].header['FILENAME'].replace('.fits', '')

            outFilename = '{0:s}/{1:s}_hst1pass_stand.csv'.format(outDir, baseFilename)

            if (not os.path.exists(outFilename)):
                tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
                t_acs   = Time(tstring, scale='utc', format='fits')

                dt = t_acs.tcb.jyear - self.tRef0.tcb.jyear

                ## We use the observation time, in combination with the proper motions to move
                ## the coordinates into the time
                self.refCat['xt'] = self.refCat['x'].values + self.refCat['pm_x'].values * dt
                self.refCat['yt'] = self.refCat['y'].values + self.refCat['pm_y'].values * dt

                self.refCat = coords.getNormalCoordinatesInTable(self.refCat, 'xt', 'yt', self.wcsRef, self.pqr0)

                c_refCat = SkyCoord(ra=self.refCat['alpha'].values * u.deg, dec=self.refCat['delta'].values * u.deg,
                                    frame='icrs')

                xiRef, etaRef = coords.getNormalCoordinates(c_refCat, self.pqr0)

                xiRef  = (xiRef / acsconstants.ACS_PLATESCALE).decompose().value
                etaRef = (etaRef / acsconstants.ACS_PLATESCALE).decompose().value

                df_hst1pass = pd.read_csv(hst1passFile)

                alpha = df_hst1pass['alpha'].values
                delta = df_hst1pass['delta'].values

                selection = ~np.isnan(alpha) & ~np.isnan(delta)

                c_image = SkyCoord(ra=alpha[selection] * u.deg, dec=delta[selection] * u.deg, frame='icrs')

                xi_init, eta_init = coords.getNormalCoordinates(c_image, self.pqr0)

                xi  = np.full(len(df_hst1pass), np.nan) * xi_init.unit
                eta = np.full(len(df_hst1pass), np.nan) * eta_init.unit

                xi[selection]  = deepcopy(xi_init)
                eta[selection] = deepcopy(eta_init)

                xi  = (xi / acsconstants.ACS_PLATESCALE).decompose().value
                eta = (eta / acsconstants.ACS_PLATESCALE).decompose().value

                idx_init, sep_init, _ = c_image.match_to_catalog_sky(c_refCat)

                idx = np.full(len(df_hst1pass), -1, dtype=int)
                sep = np.full(len(df_hst1pass), np.inf) * sep_init.unit

                idx[selection] = deepcopy(idx_init)
                sep[selection] = deepcopy(sep_init)

                sep_pix = sep.to(u.mas) / acsconstants.ACS_PLATESCALE

                hasRefCat = sep_pix < self.max_sep

                ## Find doubly-identified sources and remove them. We use numpy unique
                ## to identify refCat sources cross-matched within MAX_SEP to multiple sources
                ## in the ACS image
                doubleIndices, counts = np.unique(idx[hasRefCat], return_counts=True)

                ## We now create mask for ACS sources that are cross-matched to the same
                ## RefCat sources.
                doubleSources = np.where(np.isin(idx, doubleIndices[counts > 1]))

                ## We remove sources cross-matched to the same RefCat sources.
                hasRefCat[doubleSources] = False

                print("MATCH: {0:d} out of {1:d} selected sources".format(hasRefCat[hasRefCat].size, hasRefCat.size))

                ## Replace the old values with default values
                df_hst1pass['hasRefCat']   = False
                df_hst1pass['refCatID']    = -1
                df_hst1pass['refCatIndex'] = -1
                df_hst1pass['xPred']       = np.nan
                df_hst1pass['yPred']       = np.nan
                df_hst1pass['xRef']        = np.nan
                df_hst1pass['yRef']        = np.nan

                ## Append new columns to the hst1pass results
                df_hst1pass['hasRefCat']                  = deepcopy(hasRefCat)
                df_hst1pass.loc[hasRefCat, 'refCatID']    = self.refCat.iloc[idx[hasRefCat]]['id'].values
                df_hst1pass.loc[hasRefCat, 'refCatIndex'] = deepcopy(idx[hasRefCat])
                df_hst1pass.loc[hasRefCat, 'xPred']       = deepcopy(xi[hasRefCat])
                df_hst1pass.loc[hasRefCat, 'yPred']       = deepcopy(eta[hasRefCat])
                df_hst1pass.loc[hasRefCat, 'xRef']        = deepcopy(xiRef[idx[hasRefCat]])
                df_hst1pass.loc[hasRefCat, 'yRef']        = deepcopy(etaRef[idx[hasRefCat]])

                df_hst1pass.drop(columns=['nAppearances', 'refCatMag', 'dx', 'dy', 'retained', 'weights', 'xi', 'eta',
                                          'xiRef', 'etaRef', 'resXi', 'resEta', 'alpha', 'delta', 'sourceID']).to_csv(
                    outFilename, index=False)

                print("CSV table written to", outFilename)

        ## We now count the number of appearances for each reference star.
        hst1passFilesCSV = sorted(glob.glob("{0:s}/*_stand.csv".format(outDir)))

        print("Found {0:d} crossmatched HST1PASS results files!".format(len(hst1passFilesCSV)))

        ## Initialize table
        nAppearances = np.zeros(len(self.refCat), dtype=int)

        for i, hst1passFilename in enumerate(hst1passFilesCSV):
            print(i, hst1passFilename)

            df_hst1pass = pd.read_csv(hst1passFilename)

            nSources = len(df_hst1pass)

            print("Read {0:d} detected sources".format(nSources))

            selection = df_hst1pass['refCatIndex'] >= 0

            refCatIndices = df_hst1pass[selection]['refCatIndex'].values

            nRefStars = len(refCatIndices)

            print("Found {0:d} detected sources with reference star counterpart".format(nRefStars))

            nAppearances[refCatIndices] += 1

            gc.set_threshold(2, 1, 1)
            print('Thresholds:', gc.get_threshold())
            print('Counts:', gc.get_count())

            del df_hst1pass
            gc.collect()
            print('Counts:', gc.get_count())

        print("ALL DONE!")

        ## Now, for each hst1pass file, we provide an addendum column giving the fractional appearances
        ## of each reference star and its instrumental magnitude (according to the reference catalogue)
        ## We now count the number of appearances for each reference star.
        addendumFilenames = []
        for i, hst1passFilename in enumerate(hst1passFilesCSV):
            print(i, hst1passFilename)

            baseName = os.path.basename(hst1passFilename).replace('.csv', '')

            addendumFilename = '{0:s}/{1:s}_addendum.csv'.format(outDir, baseName)

            if (not os.path.exists(addendumFilename)):
                df_hst1pass = pd.read_csv(hst1passFilename)

                nSources = len(df_hst1pass)

                print("Read {0:d} detected sources".format(nSources))

                selection     = df_hst1pass['refCatIndex'] >= 0
                refCatIndices = df_hst1pass[selection]['refCatIndex'].values
                nRefStars     = refCatIndices.size

                print("Found {0:d} detected sources with reference star counterpart".format(nRefStars))

                df_hst1pass['nAppearances'] = np.nan
                df_hst1pass['refCatMag']    = np.nan

                df_hst1pass.loc[selection, 'nAppearances'] = nAppearances[refCatIndices]
                df_hst1pass.loc[selection, 'refCatMag']    = self.refCat.iloc[refCatIndices]['m'].values

                df_hst1pass[['nAppearances', 'refCatMag']].to_csv(addendumFilename, index=False)

                print("Addendum written to {0:s}".format(addendumFilename))

            addendumFilenames.append(addendumFilename)

        print("ALL DONE!")
        return hst1passFilesCSV, addendumFilenames