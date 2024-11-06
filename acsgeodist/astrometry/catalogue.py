import gc
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

MIN_N_EPOCH = 5

MIN_N_OBS = 3

ONE_SIGMA   = 68.2689492
THREE_SIGMA = 99.7300204

SIGMA_THRESHOLD = 3.0

N_PARAMS = np.array([2, 4, 5])

PM = np.full_like(N_PARAMS, 1.0 / 3.0, dtype=float)

LOG_PM = np.log(PM)

HEIGHT  = 0.5 * u.deg
WIDTH   = HEIGHT

MIN_RUWE = 0.8
MAX_RUWE = 1.2

class SourceCollector:
    def __init__(self, qMin=None, qMax=None, min_t_exp=None, min_n_stars=None, max_pos_targs=None, max_sep=None,
                 min_n_epoch=None):
        self.qMin          = 1.e-6
        self.qMax          = 0.5
        self.min_t_exp     = 99.0
        self.min_n_stars   = 10000
        self.max_pos_targs = np.inf
        self.max_sep       = 1.0 * u.pix
        self.min_n_epoch   = 5

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

    def collectIndividualSources(self, fitSummaryTableFilenames, residsFilenames, c0, pOrder=5, outDir='./'):
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        df_fitResults = reader.readFitResults(fitSummaryTableFilenames, pOrder)

        df_fitResults['i'] = [rootname.replace('_flc', '')[0] for rootname in df_fitResults['rootname']]
        df_fitResults['ppp'] = [rootname.replace('_flc', '')[1:4] for rootname in df_fitResults['rootname']]
        df_fitResults['ss'] = [rootname.replace('_flc', '')[4:6] for rootname in df_fitResults['rootname']]
        df_fitResults['oo'] = [rootname.replace('_flc', '')[6:8] for rootname in df_fitResults['rootname']]
        df_fitResults['t'] = [rootname.replace('_flc', '')[8] for rootname in df_fitResults['rootname']]

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

            nObsData = df_nObs['nObs'].values.tolist()
            nEpochs = df_nObs['nEpochs'].values.tolist()

        obsDataFilenameAll = '{0:s}/47Tuc_allSources_individualObservations.h5'.format(outDir)

        nRowsTotal   = 0
        thisFileDone = []
        iterate      = range(nFiles)
        for i in iterate:
            startTime = time.time()

            residsFile = residsFilenames[i]

            rootname = os.path.basename(residsFile).split('_')[0]

            df_coeffs = df_fitResults[(df_fitResults['rootname'].str.contains(rootname))]

            tObs     = df_coeffs['time'].values[0]
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

                    df_resids['time_ut1'] = tObs
                    df_resids['vaFactor'] = vaFactor
                    df_resids['rootname'] = rootname
                    df_resids['epochID'] = epochID

                    xi = (df_resids['xi'].values * u.pix) * acsconstants.ACS_PLATESCALE
                    eta = (df_resids['eta'].values * u.pix) * acsconstants.ACS_PLATESCALE

                    argsel = np.argwhere(
                        ~np.isnan(xi) & ~np.isnan(eta) & (df_resids['q'] > self.qMin) & (df_resids['q'] <= self.qMax)).flatten()

                    c = coords.getCelestialCoordinatesFromNormalCoordinates(xi[argsel], eta[argsel], c0, frame='icrs')

                    sourceIDs = list(df_resids.iloc[argsel]['sourceID'])

                    ## Once we get the list of sourceIDs we want and their corresponding indices, we drop the sourceID column to save space
                    df_resids = df_resids.drop(columns=['sourceID'])

                    ## Populate the reference catalog, names list, and observation data with the data from the first plate
                    if (len(names) <= 0):
                        names = deepcopy(sourceIDs)
                        catalog = SkyCoord(ra=c.ra, dec=c.dec, distance=1, frame='icrs')
                        obsData = [pd.DataFrame([df_resids.iloc[index]]) for index in argsel]
                        nObsData = np.ones(len(sourceIDs), dtype=int).tolist()
                        nEpochs = np.ones(len(sourceIDs), dtype=int).tolist()
                        epochIDs = np.full((len(sourceIDs), 1), epochID, dtype=int).tolist()
                    else:
                        ## Now we do cross-matching
                        idx, d2d, _ = c.match_to_catalog_sky(catalog)

                        d2d_pix = (d2d / acsconstants.ACS_PLATESCALE).decompose()

                        ## Select matching sources
                        matches = d2d_pix < self.max_sep

                        ra_new = []
                        dec_new = []

                        ## Cycle over the rows, matches, and the indices of the reference catalog
                        for ii, (index, match, matchIdx) in enumerate(zip(argsel, matches, idx)):
                            ## Grab the corresponding row
                            row = pd.DataFrame([df_resids.iloc[index]])

                            ## If it's a match, add to the reference catalog and the observation data
                            if match:
                                ## Concatenate the new row to the corresponding matching row(s).
                                obsData[matchIdx] = pd.concat([obsData[matchIdx], row], ignore_index=True)
                                obsData[matchIdx].reset_index()

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

                                ra_new.append(c[ii].ra.deg)
                                dec_new.append(c[ii].dec.deg)

                        ra_new = np.array(ra_new)
                        dec_new = np.array(dec_new)

                        ## Update the catalogs coordinates
                        catalog = coordinates.concatenate([catalog,
                                                           SkyCoord(ra=ra_new * u.deg, dec=dec_new * u.deg, distance=1,
                                                                    frame='icrs')])

                    nRowsTotal += xi.size

                    print(len(names), len(catalog), len(obsData), nRowsTotal, end='')
                    thisFileDone.append(rootname)

                    elapsedTime = time.time() - startTime

                    print(" DONE PROCESSING {0:s}! Elapsed time: {1:s}".format(rootname, convertTime(elapsedTime)))
                else:
                    print(". File has been worked out. Skipping file...")
            else:
                print(".")

            if ((((i % nWrite) == 0) and (i > 0)) or (i == iterate[-1])):
                print("Writing observation data and flushing out the dictionary...")

                startTimeFlush = time.time()

                argsel = np.argwhere(np.array(nEpochs) >= self.min_n_epoch).flatten()

                print("NUMBER OF SOURCES SO FAR (SELECTED):", len(nEpochs), len(argsel))

                mode = 'w'

                if os.path.exists(obsDataFilenameAll):
                    mode = 'a'

                store = pd.HDFStore(obsDataFilenameAll, mode)

                for index in argsel:
                    name = names[index]

                    store.append(name, obsData[index], index=False, append=True, format='table', complevel=None)

                    ## obsData[index].to_hdf(obsDataFilenameAll, key=name, mode='a', append=True, complevel=None)

                    ## Drop the whole rows once they're flushed to save memory, but keep the dataframe header so later
                    ## we can load them with new rows
                    obsData[index] = obsData[index][0:0]

                store.close()

                ## Expand the list of files done after the collected data have been flushed
                fileDone.extend(thisFileDone)

                thisFileDone.clear()

                ## Writing the number of observations for individual sources
                df_nObs = pd.DataFrame.from_dict({'source_id': names, 'nObs': nObsData, 'nEpochs': nEpochs})
                df_nObs.to_csv(nObsDataFilename, index=False)

                df_fileDone = pd.DataFrame.from_dict({'rootname': fileDone})
                df_fileDone.to_csv(fileDoneFilename, index=False)

                gc.collect()

                elapsedTime = time.time() - startTimeFlush
                print("DONE FLUSHING! Elapsed time:", convertTime(elapsedTime))


        elapsedTime = time.time() - startTimeAll

        print("ALL DONE! Elapsed time:", convertTime(elapsedTime))

        return obsDataFilenameAll, nObsDataFilename

    def calculateAstrometricSolutions(self, obsDataFilename, nObsDataFilename, c0, refPix_x=None, refPix_y=None,
                                      maxNModel=2, nIter=5, tRef=2016.0, outDir='./', reOrientFrame=True):
        properMotionFilename = '{0:s}/47Tuc_PPMPLXCatalogue_modelSelection_weightedRegression_tRef{1:0.1f}.csv'.format(
            outDir, tRef)

        if (not os.path.exists(properMotionFilename)):
            triu_indices = []
            for model in range(maxNModel):
                triu_indices.append(np.triu_indices(N_PARAMS[model]))

            triu_indices_all = np.triu_indices(N_PARAMS[-1])

            df_nObs = pd.read_csv(nObsDataFilename)

            df_nObs[['nObs', 'nEpochs']].describe()

            selection = df_nObs['nEpochs'] >= MIN_N_EPOCH

            print("SELECTED {0:d} SOURCES!".format(selection[selection].shape[0]))

            ## Now we take the normal triad pqr_0 of the reference coordinate
            pqr0 = coords.getNormalTriad(c0)

            columns = ['sourceID', 'tMin', 'tMax', 'nObs', 'nEpoch', 'meanMag', 'medianQ', 'bestModel', 'LR', 'pVal',
                       'nSigma', 'pMD', 'rms', 'nEff', 'xi0', 'eta0', 'pm_xi', 'pm_eta', 'parallax']

            N_COVS = 0
            for i in range(N_PARAMS[-1]):
                for j in range(i, N_PARAMS[-1]):
                    N_COVS += 1
                    columns.append('cov_{0:d}{1:d}'.format(i, j))

            df_ppm = pd.DataFrame(columns=columns)

            storeIn = pd.HDFStore(obsDataFilename, 'r')

            nPrint = 10000
            write  = True

            reg = LinearRegression(fit_intercept=False, copy_X=False)

            startTimeAll = time.time()

            for i, sourceID in enumerate(df_nObs[selection]['source_id'].values):
                if (sourceID in storeIn):
                    df = storeIn[sourceID]

                    refCatID = df['refCatID'].iloc[0]

                    nEpochs = df['epochID'].unique().size

                    xi = (df['xi'] * df['vaFactor']).values
                    eta = (df['eta'] * df['vaFactor']).values
                    t = Time(df['time_ut1'].values, format='decimalyear', scale='ut1')

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

                    X = astro.getAstrometricModels(t, tRef, maxNModel=maxNModel, pqr0=pqr0)

                    y = np.zeros((2 * nObs, 1))

                    y[0::2, 0] = xi
                    y[1::2, 0] = eta

                    lnLL = np.zeros(maxNModel)
                    LR = np.zeros(maxNModel)
                    lnPMD = np.zeros(maxNModel)
                    pVal = np.zeros(maxNModel)
                    nSigma = np.zeros(maxNModel)
                    prevNPars = 0
                    prevLnLL = -np.inf
                    lnPSum = -np.inf
                    solutions = []
                    covs = []
                    rms = []
                    nEff = []
                    for model in range(maxNModel):
                        weights = deepcopy(weights_init)
                        W = sparse.diags(weights)

                        for iter in range(nIter):
                            reg.fit(X[model], y, sample_weight=weights)

                            astro_solution = reg.coef_[0]

                            res = y - reg.predict(X[model])

                            mse = (res.T @ W @ res)[0, 0] / np.sum(weights)

                            '''
                            if (np.sum(weights) <= 0.0):
                                print(i, sourceID, refCatID, nEpochs, nObs, mse, model, iter, astro_solution)
                                print(weights)
                            ''';

                            '''
                            if (i == 59714):
                                print(i, sourceID, refCatID, nEpochs, nObs, mse, model, iter, astro_solution)
                                print(weights)
                            ''';

                            mean, cov = stat.estimateMeanAndCovarianceMatrixRobust(res, weights)

                            z = stat.getMahalanobisDistances(res, mean, np.array([1.0 / cov]))

                            '''
                            if (np.nansum(stat.wdecay(z)) <= 0.0):
                                break
                            ''';

                            ## We now use the z statistics to re-calculate the weights
                            weights = stat.wdecay(z)

                            W = sparse.diags(weights)

                        solutions.append(astro_solution)

                        XTWX = X[model].T @ W @ X[model]

                        covs.append(mse * linalg.inv(XTWX))

                        rms.append(np.sqrt(mse))

                        nEff.append(0.5 * np.nansum(weights))

                        ## covs.append(list(cov[triu_indices[model]].flatten()))

                        lnLL[model] = stat.getLnLL(res, weights)

                        lnPMD[model] = stat.getLnPMD(N_PARAMS[model], res, LOG_PM[model], weights=weights)

                        LR[model] = -2.0 * (prevLnLL - lnLL[model])

                        pVal[model] = stats.chi2.sf(LR[model], N_PARAMS[model] - prevNPars)

                        nSigma[model] = stats.norm.ppf(1.0 - 0.5 * pVal[model])

                        if (nSigma[model] > SIGMA_THRESHOLD):
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

                    ## print("BEST_MODEL:", bestModel, nSigma[bestModel], pMD[bestModel], bestSolution)
                    ## print(bestSolution.shape, bestCov.shape)

                    tMin = df['time_ut1'].min()
                    tMax = df['time_ut1'].max()

                    df['flux'] = 10.00 ** (-0.4 * df['W'])

                    meanMag = -2.5 * np.log10(df['flux'].mean())
                    medianQ = df['q'].median()

                    dataLine = [sourceID, tMin, tMax, nObs, nEpochs, meanMag, medianQ, bestModel + 1, LR[bestModel],
                                pVal[bestModel], nSigma[bestModel], pMD[bestModel], bestRMS, bestNEff]
                    dataLine.extend(list(bestSolution))
                    dataLine.extend(list(bestCov))

                    df_ppm = pd.concat([df_ppm, pd.DataFrame([dataLine], columns=columns)], ignore_index=True)

                if (((i % nPrint) == 0) and (i > 0)):
                    elapsedTime = time.time() - startTimeAll
                    print("DONE PROCESSING {0:d} SOURCES! ELAPSED TIME: {1:s}".format(i + 1, convertTime(elapsedTime)))

                '''
                if (i > 1000):
                    break
                ''';

            storeIn.close()

            df_ppm = df_ppm.sort_values(by='xi0', ascending=True, ignore_index=True)

            df_ppm.to_csv(properMotionFilename, index=False, mode='w')

            elapsedTime = time.time() - startTimeAll

            print("ALL DONE! Elapsed time:", convertTime(elapsedTime))
        else:
            print("FILE EXISTS ALREADY!")

        if reOrientFrame:
            print("RE-ORIENTING CELESTIAL FRAME USING GAIA DR3 STARS...")
            print("QUERYING THE CATALOGUE...")
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            Gaia.ROW_LIMIT = -1

            coord = SkyCoord(ra=c0.ra, dec=c0.dec, frame='icrs')

            g = Gaia.query_object_async(coordinate=coord, width=WIDTH, height=HEIGHT)

            gaia_selection = (g['ruwe'] >= MIN_RUWE) & (g['ruwe'] <= MAX_RUWE)

            g = g[gaia_selection]

            gdr3_id = np.array(['GDR3_{0:d}'.format(sourceID) for sourceID in g['SOURCE_ID']], dtype=str)

            c_gdr3 = SkyCoord(ra=g['ra'].value * u.deg, dec=g['dec'].value * u.deg,
                              pm_ra_cosdec=g['pmra'].value * u.mas / u.yr, pm_dec=g['pmdec'].value * u.mas / u.yr,
                              obstime=Time(g['ref_epoch'].value, format='jyear', scale='tcb'))

            print("GETTING THE SHIFT AND ROTATION MATRIX...")

            xi0_gdr3, eta0_gdr3 = coords.getNormalCoordinates(c_gdr3, coords.getNormalTriad(c0))

            xi0_gdr3_pix  = (xi0_gdr3 / acsconstants.ACS_PLATESCALE).decompose().value
            eta0_gdr3_pix = (eta0_gdr3 / acsconstants.ACS_PLATESCALE).decompose().value

            df_ppm = pd.read_csv(properMotionFilename)

            gdr3Sources = df_ppm['sourceID'].str.contains('GDR3')

            df_ppm['xi0_gdr3'] = np.nan
            df_ppm['eta0_gdr3'] = np.nan

            hasGDR3_indices_original = np.argwhere(gdr3Sources).flatten()

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

            hasGDR3_indices = np.array(hasGDR3_indices)

            df_new = pd.DataFrame(columns=['m', 'x', 'y', 'pm_x', 'pm_y'])

            df_new['m'] = df_ppm['meanMag']

            xi0, eta0 = df_ppm.iloc[hasGDR3_indices]['xi0'], df_ppm.iloc[hasGDR3_indices]['eta0']
            xi0_gdr3, eta0_gdr3 = df_ppm.iloc[hasGDR3_indices]['xi0_gdr3'], df_ppm.iloc[hasGDR3_indices]['eta0_gdr3']

            X, scaler       = sip.buildModel(xi0, eta0, 1)
            XAll, scalerAll = sip.buildModel(df_ppm['xi0'], df_ppm['eta0'], 1)

            reg = LinearRegression(fit_intercept=False)

            for i in range(acsconstants.NAXIS):
                if (i == 0):
                    y = xi0_gdr3
                elif (i == 1):
                    y = eta0_gdr3

                reg.fit(X, y)

                print(reg.coef_)

                new_coords = reg.predict(XAll)
                new_pm = df_ppm['pm_xi'] * reg.coef_[1] + df_ppm['pm_eta'] * reg.coef_[2]

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

            propMotionFilenameGDR3 = '{0:s}/PPMPLXCatalogue_tRef{1:0.1f}_GDR3Corr.txt'.format(outDir, tRef)

            df_new.to_csv(propMotionFilenameGDR3, sep=' ', header=False, index=True)

            return properMotionFilename, propMotionFilenameGDR3
        else:
            return properMotionFilename

class CrossMatcher:
    def __init__(self, referenceCatalog, referenceWCS, tRef0, max_sep=None):
        self.refCat  = deepcopy(referenceCatalog)
        self.wcsRef  = deepcopy(referenceWCS)
        self.tRef0   = tRef0
        self.max_sep = 1.0 * u.pix

        if (max_sep is not None):
            self.max_sep = max_sep



    def crossMatch(self, hst1passFiles, imageFilenames):
        for i, (hst1passFile, imageFilename) in enumerate(zip(hst1passFiles, imageFilenames)):
            addendumFilename = hst1passFile.replace('.csv', '_addendum.csv')

            hduList = fits.open(imageFilename)

            tstring = hduList[0].header['DATE-OBS'] + 'T' + hduList[0].header['TIME-OBS']
            t_acs   = Time(tstring, scale='utc', format='fits')

            dt = t_acs.decimalyear - self.tRef0.utc.value

            ## We use the observation time, in combination with the proper motions to move
            ## the coordinates into the time
            self.refCat['xt'] = self.refCat['x'].value + self.refCat['pm_x'].value * dt
            self.refCat['yt'] = self.refCat['y'].value + self.refCat['pm_y'].value * dt

            df_hst1pass = pd.read_csv(hst1passFile)

            ra = df_hst1pass['alpha'] * u.deg
            de = df_hst1pass['delta'] * u.deg