
import os
import pdb
import sys
import time
import json
import utils
import numpy
import matplotlib
from matplotlib import pyplot
pyplot.ion()





class DigitCanvas(object):
    '''Class for creating an interactive canvas for draing digits'''

    def __init__(self):

        self.fig, self.ax = pyplot.subplots(figsize=(5,5))
        #self.fig.subplots_adjust()

        self.ax.set_title('Draw a single-digit number', size=14)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        for i in range(28+1):
            self.ax.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
            self.ax.axhline(i-0.5, color='gray', lw=1, alpha=0.3)



        self.button_depressed = False
        self.sketch_coords = {'x':[], 'y':[]}

        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)
        cid = self.fig.canvas.mpl_connect('button_release_event', self._onClickRelease)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self._onMotion)


        self.sketch_data = pyplot.plot(0, 0, color='k', lw=8)[0]



    def _onClick(self, event):
        self.button_depressed = True
        self.sketch_coords['x'].append(event.xdata)
        self.sketch_coords['y'].append(event.ydata)


    def _onClickRelease(self, event):
        self.button_depressed = False


    def _onMotion(self, event):
        if self.button_depressed:
            self.sketch_coords['x'].append(event.xdata)
            self.sketch_coords['y'].append(event.ydata)

            self.sketch_data.set_data((self.sketch_coords['x'], self.sketch_coords['y']))
            pyplot.draw()

if __name__ == '__main__':

    dc = DigitCanvas()


