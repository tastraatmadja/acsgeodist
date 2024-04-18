'''
Created on Apr 13, 2024

@author: tastraatmadja
'''

import math

from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch

import numpy as np


def plot_values(fontsize):
    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=fontsize)
    plt.rc('axes', titlesize='large')
    plt.rc('axes', labelsize='medium')
    plt.rc('xtick', labelsize='x-small', top=True, bottom=True)
    plt.rc('ytick', labelsize='x-small', left=True, right=True)
    plt.rc('legend', fontsize='small')


# Default plot values
def def_plot_values():
    plot_values(12)


def def_plot_values_large():
    plot_values(16)


def def_plot_values_extra_large():
    plot_values(20)


def getLogTickMarks(xmin, xmax, dx=1):
    minTicks = math.floor(math.log10(xmin))
    maxTicks = math.ceil(math.log10(xmax))

    mantissa = np.arange(0, 10, dx)
    mantissa[0] = 1

    tickMarks = []
    for i in np.arange(minTicks, maxTicks + 1, 1):
        for j in mantissa:
            thisTick = j * math.pow(10., i)
            if ((thisTick > xmin) & (thisTick < xmax)):
                tickMarks.append(thisTick)
    return np.asarray(tickMarks)


def getLogLabels(axWhichAxis, sign='', extent=None, axis=True, logScaleAlready=True, verbose=False):
    if axis:
        if logScaleAlready:
            tickVal = axWhichAxis.get_majorticklocs()
        else:
            tickVal = np.log10(axWhichAxis.get_majorticklocs())
    else:
        if logScaleAlready:
            tickVal = axWhichAxis
        else:
            tickVal = np.log10(axWhichAxis)
    tickLabels = []
    for logLabel in tickVal:
        if (extent == None):
            if (logLabel == -2):
                tickLabel = sign + r'$0.01$'
            elif (logLabel == -1):
                tickLabel = sign + r'$0.1$'
            elif (logLabel == 0):
                tickLabel = sign + r'$1$'
            elif (logLabel == 1):
                tickLabel = sign + r'$10$'
            else:
                tickLabel = r'$' + sign + r'10^{' + '{0:0.0f}'.format(logLabel) + r'}$'
        else:
            if ((logLabel >= extent[0]) and (logLabel <= extent[1])):
                if (logLabel == -3):
                    tickLabel = sign + r'$0.001$'
                elif (logLabel == -2):
                    tickLabel = sign + r'$0.01$'
                elif (logLabel == -1):
                    tickLabel = sign + r'$0.1$'
                elif (logLabel == 0):
                    tickLabel = sign + r'$1$'
                elif (logLabel == 1):
                    tickLabel = sign + r'$10$'
                elif (logLabel == 2):
                    tickLabel = sign + r'$100$'
                elif (logLabel == 3):
                    tickLabel = sign + r'$1000$'
                else:
                    tickLabel = r'$' + sign + r'10^{' + '{0:0.0f}'.format(logLabel) + r'}$'
            else:
                tickLabel = r'$' + sign + r'10^{' + '{0:0.0f}'.format(logLabel) + r'}$'
        if verbose:
            print(logLabel, tickLabel)
        tickLabels.append(tickLabel)
    return tickLabels


def drawCommonLabel(xlabel, ylabel, fig, xPad=15, yPad=20, xLabelPosition=None, yLabelPosition=None):
    ax = fig.add_subplot(111)  # The big subplot

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.patch.set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    if (xLabelPosition is not None):
        ax.xaxis.set_label_position(xLabelPosition)
    if (yLabelPosition is not None):
        ax.yaxis.set_label_position(yLabelPosition)

    ax.set_xlabel(xlabel, labelpad=xPad)
    ax.set_ylabel(ylabel, labelpad=yPad)
    return ax


class _TransformedBboxWithCallback(TransformedBbox):
    """
    Variant of `.TransformBbox` which calls *callback* before returning points.
    Used by `.mark_inset` to unstale the parent axes' viewlim as needed.
    """

    def __init__(self, *args, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback

    def get_points(self):
        self._callback()
        return super().get_points()


class BboxConnector(Patch):
    @staticmethod
    def get_bbox_edge_pos(bbox, loc, upper=False):
        """
        Return the ``(x, y)`` coordinates of corner *loc* of *bbox*; parameters
        behave as documented for the `.BboxConnector` constructor.
        """
        x0, y0, x1, y1 = bbox.extents
        if loc == 1:
            if upper:
                return x1, y0
            else:
                return x1, y1
        elif loc == 2:
            if upper:
                return x0, y0
            else:
                return x0, y1
        elif loc == 3:
            if upper:
                return x0, y1
            else:
                return x0, y0
        elif loc == 4:
            if upper:
                return x1, y1
            else:
                return x1, y0

    @staticmethod
    def connect_bbox(bbox1, bbox2, loc1, loc2=None, upper=False):
        """
        Construct a `.Path` connecting corner *loc1* of *bbox1* to corner
        *loc2* of *bbox2*, where parameters behave as documented as for the
        `.BboxConnector` constructor.
        """
        if isinstance(bbox1, Rectangle):
            bbox1 = TransformedBbox(Bbox.unit(), bbox1.get_transform())
        if isinstance(bbox2, Rectangle):
            bbox2 = TransformedBbox(Bbox.unit(), bbox2.get_transform())
        if loc2 is None:
            loc2 = loc1
        x1, y1 = BboxConnector.get_bbox_edge_pos(bbox1, loc1, upper=upper)
        x2, y2 = BboxConnector.get_bbox_edge_pos(bbox2, loc2)
        return Path([[x1, y1], [x2, y2]])

    def __init__(self, bbox1, bbox2, loc1, loc2=None, upper=False, **kwargs):
        """
        Connect two bboxes with a straight line.
        Parameters
        ----------
        bbox1, bbox2 : `matplotlib.transforms.Bbox`
            Bounding boxes to connect.
        loc1, loc2 : {1, 2, 3, 4}
            Corner of *bbox1* and *bbox2* to draw the line. Valid values are::
                'upper right'  : 1,
                'upper left'   : 2,
                'lower left'   : 3,
                'lower right'  : 4
            *loc2* is optional and defaults to *loc1*.
        **kwargs
            Patch properties for the line drawn. Valid arguments include:
            %(Patch:kwdoc)s
        """
        if "transform" in kwargs:
            raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        if 'fill' in kwargs:
            super().__init__(**kwargs)
        else:
            fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
            super().__init__(fill=fill, **kwargs)
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.loc1 = loc1
        self.loc2 = loc2
        self.upper = upper

    def get_path(self):
        # docstring inherited
        return self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2, upper=self.upper)


def mark_inset(parent_axes, inset_axes, loc1, loc2, upper=False, **kwargs):
    rect = _TransformedBboxWithCallback(
        inset_axes.viewLim, parent_axes.transData,
        callback=parent_axes._unstale_viewLim)

    if 'fill' in kwargs:
        pp = BboxPatch(rect, **kwargs)
    else:
        fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
        pp = BboxPatch(rect, fill=fill, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, upper=upper, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, upper=upper, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2
