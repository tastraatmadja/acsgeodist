'''
Created on Apr 13 2024

@author tastraatmadja
'''
from astropy import units as u
import numpy as np

ACS_PLATESCALE = 50.0 * u.mas / u.pix

SBC_PLATESCALE = np.array([31.21, 32.74]) * u.mas / u.pix

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

FILTERS = ["F435W", "F606W", "F814W"]

N_FILTERS = len(FILTERS)

##                    CH2 (BOTTOM) CH1 (TOP) COMBINED_PROPERTIES
FILTER_CHIP_COLORS = [['#2c7fb8', '#a1dab4', '#253494'], ## F435W
                      ['#31a354', '#c2e699', '#006837'], ## F606W
                      ['#dd1c77', '#d7b5d8', '#980043']] ## F814W
