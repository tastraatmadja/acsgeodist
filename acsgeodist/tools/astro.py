from astropy import units as u
import healpy as hp
import numpy as np

HP_RESOLUTION = 28
HP_NSIDES     = 2 ** HP_RESOLUTION
HP_NESTED     = True

N_PARAMS = np.array([2, 4, 5])

NAXIS = 2

'''
Given a SkyCoord object c, return the source ID of an ACS source using healpix, assuming that the healpix resolution is
28, giving n_sides = 2 ** 28. We also use nested scheme for generating the ID. The resolution is small enough so that 
even sources close to each other can have unique id.
'''
def generateSourceID(c):
    pixIds = hp.pixelfunc.ang2pix(HP_NSIDES, c.dec.to_value(u.radian), c.ra.to_value(u.radian), nest=HP_NESTED, lonlat=True)

    return ['ACS_{0:018d}'.format(pixId) for pixId in pixIds]

def getAstrometricModels(t, t_ref, maxNModel=3):
    dt = t - t_ref

    nData = 2 * t.size

    X = []

    for model in range(maxNModel):
        X.append(np.zeros((nData, N_PARAMS[model])))

        for axis in range(NAXIS):
            X[model][axis::2, axis] = 1.0

            if (model > 0):
                X[model][axis::2, NAXIS + axis] = dt
    return X