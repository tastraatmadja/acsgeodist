from astropy import units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, get_body_barycentric
from astropy.io import fits
from astropy.time import Time, TimeDelta
from calcos import orbit
import healpy as hp
import numpy as np

from acsgeodist.tools import coords

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

'''
Given a list of HST SPT and FLC files, return the position and velocity of the telescope at the middle of the exposure.
Position and Velocity is return as a CartesianRepresentation object. Also return the mid-exposure time in the form of 
a Time object.
'''
def getHSTPosVelTime(sptFiles, dataFiles):
    x = []
    y = []
    z = []

    v_x = []
    v_y = []
    v_z = []

    times = []
    for sptFile, dataFile in zip(sptFiles, dataFiles):
        orb = orbit.HSTOrbit(sptFile)
        hdu = fits.open(dataFile)

        expStart = Time(hdu[0].header['EXPSTART'], format='mjd')
        expTime  = hdu[0].header['EXPTIME']
        timeMid  = expStart + TimeDelta(0.5 * expTime, format='sec')

        times.append(timeMid.mjd)

        (rect_hst, vel_hst) = orb.getPos(timeMid.mjd)

        x.append(rect_hst[0])
        y.append(rect_hst[1])
        z.append(rect_hst[2])

        v_x.append(vel_hst[0])
        v_y.append(vel_hst[1])
        v_z.append(vel_hst[2])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    v_x = np.array(v_x)
    v_y = np.array(v_y)
    v_z = np.array(v_z)

    times = np.array(times)

    pv = (CartesianRepresentation(x, y, z, unit=u.km), CartesianRepresentation(v_x, v_y, v_z, unit=u.km / u.s))
    t  = Time(times, scale='ut1', format='mjd')
    return pv, t

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
    dt = (t.tdb.decimalyear - t_ref.tdb.decimalyear + (eb.to_value(u.au) @ pqr0[2].reshape((3, -1))).flatten() * AULTY.value) * u.yr

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