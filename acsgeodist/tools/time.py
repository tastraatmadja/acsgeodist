'''
Created on Oct 25, 2021

@author: tastraatmadja
'''

def convertTime(sec):
    minutes = sec // 60
    seconds = sec % 60
    hours   = minutes // 60
    minutes = minutes % 60
    return "{0:.0f} h {1:.0f} m {2:.3f} s".format(hours, minutes, seconds)