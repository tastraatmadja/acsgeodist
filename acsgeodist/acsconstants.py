'''
Created on Apr 13 2024

@author tastraatmadja
'''
from astropy import units as u

ACS_PLATESCALE = 50.0 * u.mas / u.pix

NAXIS = 2

AXIS_NAMES = [r'$X$', r'$Y$']

COEFF_LABELS = ['A', 'B']

WFC = ['WFC2', 'WFC1']

CHIP_POSITIONS = ['bottom', 'top']

CHIP_LABEL = lambda wfc, pos : '{0:s} ({1:s})'.format(wfc, pos)