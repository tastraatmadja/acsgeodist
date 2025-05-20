'''
Created on Apr 13 2024

@author tastraatmadja
'''
from astropy import units as u
import numpy as np

ACS_PLATESCALE = 50.0 * u.mas / u.pix

NAXIS = 2

AXIS_NAMES = [r'$X$', r'$Y$']

COEFF_LABELS = ['A', 'B']

CHIP_NUMBER   = np.array([2, 1], dtype=int)
HEADER_NUMBER = np.array([1, 2], dtype=int)

CHIP_POSITIONS = ['bottom', 'top']

WFC = ['WFC{0:d}'.format(chipNumber) for chipNumber in CHIP_NUMBER] ## ['WFC2', 'WFC1']

WFC_COLORS = ['#1f78b4', '#33a02c']

CHIP_LABEL = lambda wfc, pos : '{0:s} ({1:s})'.format(wfc, pos)
WFC_LABELS = [CHIP_LABEL(wfc, pos) for wfc, pos in zip(WFC, CHIP_POSITIONS)]

N_CHIPS = len(WFC)
