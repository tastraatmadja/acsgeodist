import gc
import os
import time

from astropy import coordinates
from astropy import units as u
from astropy.coordinates import SkyCoord
from copy import deepcopy
import numpy as np
import pandas as pd

from acsgeodist import acsconstants
from acsgeodist.tools import astro, coords, reader
from acsgeodist.tools.time import convertTime

N_WRITE = 100
class SourceCollector():
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
                df_nObs.to_csv('{0:s}/nObsData.csv'.format(outDir), index=False)

                df_fileDone = pd.DataFrame.from_dict({'rootname': fileDone})
                df_fileDone.to_csv(fileDoneFilename, index=False)

                gc.collect()

                elapsedTime = time.time() - startTimeFlush
                print("DONE FLUSHING! Elapsed time:", convertTime(elapsedTime))


        elapsedTime = time.time() - startTimeAll

        print("ALL DONE! Elapsed time:", convertTime(elapsedTime))