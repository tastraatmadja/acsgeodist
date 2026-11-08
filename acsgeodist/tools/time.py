'''
Created on Oct 25, 2021

@author: tastraatmadja
'''

from astropy.time import Time

def getObservingTimeFromHeader(header):
    tstring = header['DATE-OBS'] + 'T' + header['TIME-OBS']
    return Time(tstring, scale='ut1', format='fits')

def convertTime(sec):
    minutes = sec // 60
    seconds = sec % 60
    hours   = minutes // 60
    minutes = minutes % 60
    return "{0:.0f} h {1:.0f} m {2:.3f} s".format(hours, minutes, seconds)