from astropy import units as u
import healpy as hp

HP_RESOLUTION = 28
HP_NSIDES     = 2 ** HP_RESOLUTION
HP_NESTED     = True

'''
Given a SkyCoord object c, return the source ID of an ACS source using healpix, assuming that the healpix resolution is
28, giving n_sides = 2 ** 28. We also use nested scheme for generating the ID. The resolution is small enough so that 
even sources close to each other can have unique id.
'''
def generateSourceID(c):
    pixIds = hp.pixelfunc.ang2pix(HP_NSIDES, c.dec.to_value(u.radian), c.ra.to_value(u.radian), nest=HP_NESTED, lonlat=True)

    return ['ACS_{0:018d}'.format(pixId) for pixId in pixIds]

## def getAstrometricDesignMatrix()