from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric
import healpy as hp
import numpy as np

## Astronomical unit (m) IAU 2012
DAU = 149597870.7e3 * u.m

## Speed of light (m/s)
CMPS = 299792458.0 * u.m / u.s

## Light time for 1 au (s)
AULT = DAU / CMPS

## Seconds per day
DAYSEC = 86400.0 * u.s

## Julian year
DJY  = 365.25 * u.yr

## Light time for 1 au, Julian years
AULTY = AULT/DAYSEC/DJY

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

def getAstrometricModels(t, t_ref, maxNModel=3, pqr0=None, site=None, pv=None):
    if pqr0 is None:
        pqr0 = np.zeros((3,3))
    if site is None:
        site = EarthLocation.from_geocentric(0.0 * u.m, 0.0 * u.m, 0.0 * u.m)

    ## Earth barycentric position at time t
    ebp = get_body_barycentric('earth', t)

    ## Position and velocity of the observer at site
    if pv is None:
        pv  = site.get_gcrs_posvel(t)

    ## Barycentric position of the observer
    eb = ebp.xyz.T.to(u.au) + pv[0].xyz.T.to(u.au)

    ## Proper motion time including Roemer's effect
    dt = (t.tdb.value - t_ref + (eb.to_value(u.au) @ pqr0[2].reshape((3, -1))).flatten() * AULTY.value) * u.yr

    nData = 2 * t.size

    X = []

    for model in range(maxNModel):
        X.append(np.zeros((nData, N_PARAMS[model])))

        for axis in range(NAXIS):
            X[model][axis::2, axis] = 1.0

            if (model > 0):
                X[model][axis::2, NAXIS + axis] = dt

                if (model > 1):
                    X[model][axis::2, 4] = -(eb.to_value(u.au) @ pqr0[axis].reshape((3,-1))).flatten()

    return X