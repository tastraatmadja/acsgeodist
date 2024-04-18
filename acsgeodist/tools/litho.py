import numpy as np
from scipy.interpolate import griddata
def get_dtab_ftab_data(path):
    tabdata = np.loadtxt(path)
    x,y,dx,dy = tabdata.T
    return x,y,dx,dy

def interp_dtab_ftab_data(path, x, y, x_divby, y_divby):
    xtab,ytab,dxtab,dytab = get_dtab_ftab_data(path)
    xshape,yshape = int(xtab.max()),int(ytab.max())
    x_eval = x/x_divby * (xshape - 1) + 1
    y_eval = y/y_divby * (yshape - 1) + 1
    dx_interp = griddata((xtab,ytab), dxtab, np.array([x_eval,y_eval]).T, method='cubic')
    dy_interp = griddata((xtab,ytab), dytab, np.array([x_eval,y_eval]).T, method='cubic')
    return x_eval, y_eval, dx_interp, dy_interp