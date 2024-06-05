from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

'''
Given a SkyCoord coordinate, return the normal triad pqr
at the coordinate. First row = p, second row = q, third
row = r
'''
def getNormalTriad(c):
    sinRA = np.sin(c.ra)
    cosRA = np.cos(c.ra)
    sinDE = np.sin(c.dec)
    cosDE = np.cos(c.dec)

    return np.array([[-sinRA, cosRA, 0.0],
                     [-sinDE * cosRA, -sinDE * sinRA, cosDE],
                     [+cosDE * cosRA, cosDE * sinRA, sinDE]])


'''
Given a SkyCoord coordinate, return the r vector at the
point.
'''
def getRTriad(c):
    sinRA = np.sin(c.ra)
    cosRA = np.cos(c.ra)
    sinDE = np.sin(c.dec)
    cosDE = np.cos(c.dec)

    return np.vstack([+cosDE * cosRA, cosDE * sinRA, sinDE]).transpose()


'''
Given a SkyCoord object and a normal triad pqr0 around a 
reference point, return the normal coordinates
'''
def getNormalCoordinates(c, pqr0):
    r = getRTriad(c)
    r0DotR = np.sum(pqr0[2] * r, axis=1)

    ## Normal coordinates in arcsec
    xi = ((np.sum(pqr0[0] * r, axis=1) / r0DotR) * u.radian).to(u.arcsec)
    eta = ((np.sum(pqr0[1] * r, axis=1) / r0DotR) * u.radian).to(u.arcsec)

    del r
    del r0DotR

    return xi, eta

'''
From normal coordinates (xi, eta), calculate its observed
celestial coordinates r = {alpha, delta}. The celestial
coordinate (alpha0, delta0) of the center of the plate must be
known. The calculation assumes gnomonic projection.
@param xi
@param eta
@param c0 the reference coordinate in the form a SkyCoord object
@return Celestial coordinates (alpha, delta) in the form of SkyCoord object
'''
def getCelestialCoordinatesFromNormalCoordinates(xi, eta, c0, frame='icrs'):
    sinDelta0 = np.sin(c0.dec)
    cosDelta0 = np.cos(c0.dec)

    alpha = c0.ra + np.arctan2(xi.to_value(u.radian) / cosDelta0, 1.0 - eta.to_value(u.radian) * np.tan(c0.dec))
    delta = np.arcsin((sinDelta0 + eta.to_value(u.radian) * cosDelta0) / np.sqrt(
        1.0 + (xi.to_value(u.radian)) ** 2 + (eta.to_value(u.radian)) ** 2)).to(u.deg)

    return SkyCoord(ra=alpha, dec=delta, frame=frame)

def shift_rotate_coords(x, y, sx, sy, theta):
    return shift_rotate_scale_coords(x, y, sx, sy, 1.0, 1.0, theta)

def shift_rotate_scale_coords(x, y, sx, sy, px, py, theta):
    dx = x - sx
    dy = y - sy
    xn =  dx * px * np.cos(theta) + dy * px * np.sin(theta)
    yn = -dx * py * np.sin(theta) + dy * py * np.cos(theta)
    return xn, yn