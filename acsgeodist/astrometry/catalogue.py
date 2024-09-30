from acsgeodist import acsconstants
from acsgeodist.tools import astro, coords, reader
from acsgeodist.tools.time import convertTime
from astropy import coordinates
from astropy import units as u
from copy import deepcopy
import gc
import numpy as np
import os
import pandas as pd
import time

N_WRITE = 100
class SourceCollector():
    def __init__(self):
        self.min_t_exp     = 0.0
        self.min_n_stars   = 0
        self.max_pos_targs = np.inf
        self.max_sep       = 1.0 * u.pix
        self.min_n_epoch   = 3

    def collectIndividualSources(self, fitSummaryTableFilename, residsFilenames, c0, pmDir, pOrder=5, min_t_exp=None,
                                 min_n_stars=None, max_pos_targs=None, max_sep=None, min_n_epoch=None):
        startTimeAll = time.time()
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

        df_fitResults = reader.readFitResults(fitSummaryTableFilename, pOrder)

        ## Organize the filenames by epoch
        df_fitResults['i']   = [rootname.replace('_flc', '')[0] for rootname in df_fitResults['rootname']]
        df_fitResults['ppp'] = [rootname.replace('_flc', '')[1:4] for rootname in df_fitResults['rootname']]
        df_fitResults['ss']  = [rootname.replace('_flc', '')[4:6] for rootname in df_fitResults['rootname']]
        df_fitResults['oo']  = [rootname.replace('_flc', '')[6:8] for rootname in df_fitResults['rootname']]
        df_fitResults['t']   = [rootname.replace('_flc', '')[8] for rootname in df_fitResults['rootname']]

        nImages = 0
        for ppp in df_fitResults.loc[df_fitResults['chip'] == 1, ['ppp']].value_counts().sort_index().index:
            ppp = ppp[0]

            for ss in df_fitResults.loc[
                (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp), ['ss']].value_counts().sort_index().index:
                ss = ss[0]

                for oo in df_fitResults.loc[
                    (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp) & (df_fitResults['ss'] == ss), [
                        'oo']].value_counts().sort_index().index:
                    oo = oo[0]

                    selected_time = df_fitResults.loc[
                        (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp) & (df_fitResults['ss'] == ss) & (
                                    df_fitResults['oo'] == oo), ['time']]

                    tMin = selected_time.min().values[0]
                    tMax = selected_time.max().values[0]
                    dt = (tMax - tMin) * 365.25 * 24 * u.hour

                    nImages += selected_time.size

                    print(ppp, ss, oo, selected_time.size, selected_time.min().values, selected_time.max().values, dt)
                print()

        print("TOTAL_NUMBER_OF_IMAGES:", nImages)

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
                    (df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp) & (df_fitResults['ss'] == ss), [
                        'time']]

                ## oo   = df_fitResults.loc[(df_fitResults['chip'] == 1) & (df_fitResults['ppp'] == ppp) & (df_fitResults['ss'] == ss), ['oo']].value_counts().sort_index().index.values

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

        df_timeBins.to_csv('{0:s}/time_bins.csv'.format(pmDir))

        nFiles = len(residsFilenames)

        if not os.path.exists(pmDir):
            os.makedirs(pmDir)

        obsData  = {}
        nObsData = {}
        nEpochs  = {}

        names   = []
        catalog = None

        fileDone = []

        fileDoneFilename = '{0:s}/fileDone.csv'.format(pmDir)
        if os.path.exists(fileDoneFilename):
            df_fileDone = pd.read_csv(fileDoneFilename)
            fileDone = list(df_fileDone['rootname'])

        nObsDataFilename = '{0:s}/nObsData.csv'.format(pmDir)
        if os.path.exists(nObsDataFilename):
            df_nObs = pd.read_csv(nObsDataFilename)

            for i, name in enumerate(list(df_nObs['source_id'])):
                nObsData[name] = df_nObs.iloc[i]['nObs']

        obsDataFilename = '{0:s}/47Tuc_allSources_individualObservations.h5'.format(pmDir)

        nRowsTotal   = 0
        thisFileDone = []
        for i in range(nFiles):
            startTime = time.time()

            residsFilename = residsFilenames[i]

            rootname = os.path.basename(residsFilename).split('_')[0]

            df_coeffs = df_fitResults[(df_fitResults['rootname'].str.contains(rootname))]

            tObs     = df_coeffs['time'].values[0]
            tExp     = df_coeffs['tExp'].values[0]
            vaFactor = df_coeffs['vaFactor'].values[0]
            nStars   = df_coeffs['nStars'].values[0]
            epochID  = df_coeffs['epochID'].values[0]

            posTarg1 = df_coeffs['posTarg1'].values[0]
            posTarg2 = df_coeffs['posTarg2'].values[0]

            posTargResultant = np.sqrt(posTarg1 ** 2 + posTarg2 ** 2)

            process = (tExp > self.min_t_exp) and (nStars > self.min_n_stars) and (posTargResultant < self.max_pos_targs)

            print(i, os.path.basename(residsFilename), rootname, epochID, tExp, nStars, posTargResultant, "PROCESS:", process, end='')

            if process:
                done = False

                if (rootname in fileDone):
                    done = True

                print(" DONE:", done, end='')

                if (not done):
                    print('.')
                    df_resids = pd.read_csv(residsFilename)

                    xi  = (df_resids['xi'].values * u.pix)  * acsconstants.ACS_PLATESCALE
                    eta = (df_resids['eta'].values * u.pix) * acsconstants.ACS_PLATESCALE

                    argsel = np.argwhere(~np.isnan(xi) & ~np.isnan(eta)).flatten()

                    c = coords.getCelestialCoordinatesFromNormalCoordinates(xi[argsel], eta[argsel], c0, frame='icrs')

                    sourceIds = astro.generateSourceID(c)

                    ## Populate the reference catalog, names list, and observation data with the data from the first plate
                    if (len(names) <= 0):
                        names = deepcopy(sourceIds)
                        catalog = coordinates.SkyCoord(ra=c.ra, dec=c.dec, distance=1, frame='icrs')

                        for index, name in zip(argsel, sourceIds):
                            row = df_resids.iloc[index].copy()

                            ## Add the time, VA factor, and file rootname
                            row['time_ut1'] = tObs
                            row['vaFactor'] = vaFactor
                            row['rootname'] = rootname
                            row['epochID'] = epochID

                            obsData[name] = pd.DataFrame([row])
                            nObsData[name] = 1
                            nEpochs[name] = 0
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
                            row = df_resids.iloc[index].copy()

                            ## Add the time, VA factor, and file rootname
                            row['time_ut1'] = tObs
                            row['vaFactor'] = vaFactor
                            row['rootname'] = rootname
                            row['epochID'] = epochID

                            ## If it's a match, add to the reference catalog and the observation data
                            if match:
                                name = names[matchIdx]

                                if name in obsData:
                                    obsData[name] = pd.concat([obsData[name], pd.DataFrame([row])], ignore_index=True)
                                    obsData[name].reset_index()
                                else:
                                    obsData[name] = pd.DataFrame([row])

                                nObsData[name] += 1

                            ## Otherwise this is a new observation and create a new entry for this observation
                            else:
                                name = sourceIds[ii]

                                obsData[name] = pd.DataFrame([row])

                                ## Append the source Id to the list of names
                                names.append(name)

                                ra_new.append(c[ii].ra.deg)
                                dec_new.append(c[ii].dec.deg)

                                nObsData[name] = 1
                                nEpochs[name] = 0

                        ra_new = np.array(ra_new)
                        dec_new = np.array(dec_new)

                        ## Update the catalogs coordinates
                        catalog = coordinates.concatenate([catalog,
                                                           coordinates.SkyCoord(ra=ra_new * u.deg, dec=dec_new * u.deg,
                                                                                distance=1, frame='icrs')])

                    nRowsTotal += xi.size

                    print(len(names), len(catalog), len(obsData), nRowsTotal)
                    thisFileDone.append(rootname)

                    elapsedTime = time.time() - startTime

                    print("DONE PROCESSING {0:s}! Elapsed time: {1:s}".format(rootname, convertTime(elapsedTime)))
                else:
                    print(". File has been worked out. Skipping file...")
            else:
                print(".")

            if ((((i % N_WRITE) == 0) and (i > 0)) or ((i + 1) == nFiles)):
                print("Writing observation data and flushing out the dictionary...")

                startTimeFlush = time.time()

                currentNames = list(obsData.keys())

                print("NUMBER OF SOURCES (SO FAR):", len(currentNames))

                for name in currentNames:
                    ## if (obsData[name].shape[0] >= MIN_N_OBS):
                    ## print(name, obsData[name]['epochID'].unique().size, nEpochs[name])
                    if ((obsData[name]['epochID'].unique().size >= self.min_n_epoch) or (nEpochs[name] >= self.min_n_epoch)):
                        obsData[name].to_hdf(obsDataFilename, key=name, mode='a', append=True, complevel=None)
                        '''

                        obsDataFilename = '{0:s}/{1:s}_individualObservations.parquet'.format(PMDIR, name)

                        append = False

                        if os.path.exists(obsDataFilename):
                            append = True

                        obsData[name].to_parquet(obsDataFilename, compression=None, engine='fastparquet', append=append)

                        thisObsData = pd.read_parquet(obsDataFilename)
                        ''';

                        '''
                        header = True
                        mode   = 'w'

                        if os.path.exists(obsDataFilename):
                            header = False
                            mode   = 'a'

                        obsData[name].to_csv(obsDataFilename, index=False, mode=mode, header=header)
                        ''';

                        del obsData[name]

                        ## Read again the current observation data to get the most current number of epochs for this source
                        thisObsData = pd.read_hdf(obsDataFilename, key=name, mode='r')

                        nEpochs[name] = thisObsData['epochID'].unique().size

                        del thisObsData

                ## Expand the list of files done after the collected data have been flushed
                fileDone.extend(thisFileDone)

                thisFileDone.clear()

                ## Writing the number of observations for individual sources
                df_nObs = pd.DataFrame.from_dict({'source_id': list(nObsData.keys()), 'nObs': list(nObsData.values()),
                                                  'nEpochs': list(nEpochs.values())})
                df_nObs.to_csv('{0:s}/nObsData.csv'.format(pmDir), index=False)

                df_fileDone = pd.DataFrame.from_dict({'rootname': fileDone})
                df_fileDone.to_csv(fileDoneFilename, index=False)

                gc.collect()

                elapsedTime = time.time() - startTimeFlush
                print("DONE FLUSHING! Elapsed time:", convertTime(elapsedTime))

        elapsedTime = time.time() - startTimeAll

        print("ALL DONE! Elapsed time:", convertTime(elapsedTime))