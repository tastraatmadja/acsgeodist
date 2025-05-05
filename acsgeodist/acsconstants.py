'''
Created on Apr 13 2024

@author tastraatmadja
'''
from astropy import units as u

ACS_PLATESCALE = 50.0 * u.mas / u.pix

NAXIS = 2

AXIS_NAMES = [r'$X$', r'$Y$']

COEFF_LABELS = ['A', 'B']

CHIP_NUMBER = [2, 1]

WFC = ['WFC2', 'WFC1']

WFC_COLORS = ['#1f78b4', '#33a02c']

CHIP_POSITIONS = ['bottom', 'top']

N_CHIPS = len(WFC)

CHIP_LABEL = lambda wfc, pos : '{0:s} ({1:s})'.format(wfc, pos)