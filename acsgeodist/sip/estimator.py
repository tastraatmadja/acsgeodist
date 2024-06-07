from acsgeodist import acsconstants
from astropy import table
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import ascii, fits
from astropy.time import Time
import gc
import numpy as np
import os

chips = np.array([1, 2], dtype=int) ## hdu [SCI, x]
class SIPEstimator():
    def __init__(self, pOrder, referenceCatalog, referenceWCS, tRef0, qMax=0.5, min_n_app=3, max_pix_tol=1.0, min_n_refstar=100):
        self.pOrder        = pOrder
        self.refCat        = referenceCatalog
        self.wcsRef        = referenceWCS
        self.tRef0         = tRef0
        self.qMax          = qMax
        self.min_n_app     = min_n_app
        self.max_pix_tol   = max_pix_tol
        self.min_n_refstar = min_n_refstar


    def processHST1PassFile(self, hst1passFile, imageFilename):
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
            textResults = ""
            for chip in chips:
                chipTitle = acsconstants.CHIP_LABEL(acsconstants.WFC[chip - 1], acsconstants.CHIP_POSITIONS[chip - 1])

                hdu = hduList['SCI', chip]

                orientat = Angle(float(hdu.header['ORIENTAT']) * u.deg).wrap_at('360d').value
                vaFactor = float(hdu.header['VAFACTOR'])

                textResults += "{0:s} {1:d} {2:.8f} {3:.6f} {4:.13f} {5:.12e} {6:0.2f} {7:f} {8:f}".format(
                    baseImageFilename, chip, t_acs.decimalyear, pa_v3, orientat, vaFactor, tExp, posTarg1, posTarg2)




                textResults += "\n"

        return textResults



