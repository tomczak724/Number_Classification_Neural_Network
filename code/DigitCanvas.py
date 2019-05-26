
import os
import pdb
import sys
import time
import json
import utils
import numpy
import matplotlib
from scipy import signal
from matplotlib import pyplot
pyplot.ion()





class DigitCanvas(object):
    '''Class for creating an interactive canvas for draing digits'''

    def __init__(self, image_size=28):

        self.fig = pyplot.figure(figsize=(7,5))
        self.ax_canvas = pyplot.subplot2grid((7,7), (0,0), rowspan=7, colspan=5, aspect='equal', label='canvas')
        self.ax_clear = pyplot.subplot2grid((7,7), (2,5), rowspan=1, colspan=2, label='clear button')
        self.ax_submit = pyplot.subplot2grid((7,7), (3,5), rowspan=1, colspan=2, label='submit button')
        self.fig.subplots_adjust(left=0.03, right=0.97, hspace=0.4, wspace=0.2)


        ###  turning off axis labels
        for ax in [self.ax_canvas, self.ax_clear, self.ax_submit]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)


        ###  initializing 2D image array
        self.image_size = image_size
        self.image_data = numpy.zeros((image_size, image_size))


        ###  adding gridlines for pixels
        self.ax_canvas.axis([-0.5, image_size-0.5, -0.5, image_size-0.5])
        for i in range(image_size):
            self.ax_canvas.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
            self.ax_canvas.axhline(i-0.5, color='gray', lw=1, alpha=0.3)

        ###  adding button labels
        self.ax_canvas.set_title('Draw a single-digit number', size=14)
        self.ax_clear.text(0.5, 0.5, 'CLEAR', fontweight='bold', size=14, ha='center', va='center')
        self.ax_submit.text(0.5, 0.5, 'SUBMIT', fontweight='bold', size=14, ha='center', va='center')

        self.mouse_depressed = False
        self.sketch_coords = {'x':[], 'y':[]}

        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)
        cid = self.fig.canvas.mpl_connect('button_release_event', self._onClickRelease)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self._onMotion)


        self.sketch_plot = self.ax_canvas.plot(0, 0, color='k', lw=10)[0]
        self.image_plot = self.ax_canvas.imshow(self.image_data, interpolation='none', vmin=0, vmax=1, cmap=pyplot.cm.Greens)


    def _onClick(self, event):

        ###  handling clicks in the main canvas
        if (event.inaxes is not None) and (event.inaxes.get_label() == 'canvas'):
            self.mouse_depressed = True
            self.sketch_coords['x'].append(event.xdata)
            self.sketch_coords['y'].append(event.ydata)

        ###  handling CLEAR button click
        if (event.inaxes is not None) and (event.inaxes.get_label() == 'clear button'):
            self.sketch_coords = {'x':[], 'y':[]}
            self.sketch_plot.set_data((0, 0))

            self.image_data = numpy.zeros((self.image_size, self.image_size))
            self.image_plot.set_data(self.image_data)
            pyplot.draw()

        ###  handling SUBMIT button click
        if (event.inaxes is not None) and (event.inaxes.get_label() == 'submit button'):

            xdata = numpy.array(self.sketch_coords['x']).astype(float)
            ydata = numpy.array(self.sketch_coords['y']).astype(float)

            inds_keep = (~numpy.isnan(xdata)) & (~numpy.isnan(xdata)) & (xdata>=0) & (ydata>=0)
            xdata = xdata[inds_keep]
            ydata = ydata[inds_keep]

            ###  interpolating by factor of 100x
            ###  looping over all adjacent pairs of coordinates
            xdata_final = []
            ydata_final = []
            for i in range(len(xdata)-1):
                x1, y1 = xdata[i], ydata[i]
                x2, y2 = xdata[i+1], ydata[i+1]

                ###  interpolating by factor of 100x
                x_100 = numpy.linspace(x1, x2, 100)
                y_100 = numpy.linspace(y1, y2, 100)
                xdata_final += x_100.tolist()
                ydata_final += y_100.tolist()

            xdata_final = numpy.array(xdata_final).round().astype(int)
            ydata_final = numpy.array(ydata_final).round().astype(int)

            ###  digitizing sketch into image array
            self.image_data[(ydata_final, xdata_final)] = 1

            ###  smoothing and normalizing image data
            kernel = [[1,3,1],[3,2,3],[1,3,1]]
            self.image_data = signal.convolve2d(self.image_data, kernel, mode='same')
            self.image_data /= self.image_data.max()

            self.image_plot.set_data(self.image_data)
            self.sketch_plot.set_data((0,0))
            pyplot.draw()



    def _onClickRelease(self, event):
        self.mouse_depressed = False


    def _onMotion(self, event):
        if self.mouse_depressed:
            self.sketch_coords['x'].append(event.xdata)
            self.sketch_coords['y'].append(event.ydata)

            self.sketch_plot.set_data((self.sketch_coords['x'], self.sketch_coords['y']))
            pyplot.draw()


if __name__ == '__main__':

    dc = DigitCanvas()


