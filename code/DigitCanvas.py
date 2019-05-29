
import os
import pdb
import sys
import time
import json
import utils
import numpy
import matplotlib
from NumberClassifier import NumberClassifier
from scipy import signal, ndimage
from matplotlib import pyplot
pyplot.ion()





class DigitCanvas(object):
    '''Class for creating an interactive canvas for draing digits'''

    def __init__(self, image_size=28):

        self.fig = pyplot.figure(figsize=(7,5))
        self.ax_canvas = pyplot.subplot2grid((7,7), (0,0), rowspan=7, colspan=5, aspect='equal', label='canvas')
        self.ax_clear = pyplot.subplot2grid((7,7), (0,5), rowspan=1, colspan=2, facecolor='#cccccc', label='clear button')
        self.ax_submit = pyplot.subplot2grid((7,7), (1,5), rowspan=1, colspan=2, facecolor='#cccccc', label='submit button')
        self.fig.subplots_adjust(left=0.03, right=0.97, hspace=0.4, wspace=0.2)

        self.neural_network = None

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
        self.sketch_collection = []

        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)
        cid = self.fig.canvas.mpl_connect('button_release_event', self._onClickRelease)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self._onMotion)


        self.sketch_plot = self.ax_canvas.plot(0, 0, color='k', lw=10)[0]
        self.image_plot = self.ax_canvas.imshow(self.image_data, interpolation='none', vmin=0, vmax=1, cmap=pyplot.cm.Greens)
        self.text_probs_plot = self.ax_canvas.text(1.1, 0.05, '', size=12, transform=self.ax_canvas.transAxes)
        self.text_guess_plot = self.ax_canvas.text(0.5, 0.98, '', size=16, transform=self.ax_canvas.transAxes, ha='center', va='top', fontweight='bold', family='sans-serif', color='r')

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

            self.text_probs_plot.set_text('')
            self.text_guess_plot.set_text('')
            pyplot.draw()

        ###  handling SUBMIT button click
        if (event.inaxes is not None) and (event.inaxes.get_label() == 'submit button'):

            xdata = numpy.array(self.sketch_coords['x']).astype(float)
            ydata = numpy.array(self.sketch_coords['y']).astype(float)

            ###  interpolating by factor of 100x
            ###  looping over all adjacent pairs of coordinates
            xdata_final = []
            ydata_final = []
            for i in range(len(xdata)-1):
                x1, y1 = xdata[i], ydata[i]
                x2, y2 = xdata[i+1], ydata[i+1]

                ###  interpolating by factor of 100x
                if not numpy.isnan([x1, x2, y1, y2]).any():
                    x_100 = numpy.linspace(x1, x2, 100)
                    y_100 = numpy.linspace(y1, y2, 100)
                    xdata_final += x_100.tolist()
                    ydata_final += y_100.tolist()

            xdata_final = numpy.array(xdata_final).round().astype(int)
            ydata_final = numpy.array(ydata_final).round().astype(int)

            ###  digitizing sketch into image array
            self.image_data[(ydata_final, xdata_final)] = 1

            ###  smoothing and normalizing image data
            kernel = [[0,1,0],
                      [1,1,1],
                      [0,1,0]]
            self.image_data = signal.convolve2d(self.image_data, kernel, mode='same')

            ###  centering and normalizing image
            self.image_data = utils.center_image(self.image_data.flatten())
            self.image_data = self.image_data.reshape((self.image_size, self.image_size))
            self.image_data /= self.image_data.max()

            self.image_plot.set_data(self.image_data)
            self.sketch_plot.set_data((0,0))
            pyplot.draw()

            if self.neural_network is not None:

                ###  inverting image along y-axis (MNIST data format)
                im = self.image_data[::-1, :].flatten()
                guess = self.neural_network.guess_digit(im)

                ###  indicating best guess
                text1 = 'Looks like a %i to me!' % guess
                self.text_guess_plot.set_text(text1)

                ###  getting probabilities
                zs, activations = self.neural_network._forward_propagate(im)
                probs = activations[-1] / activations[-1].sum()

                text = 'Digit probabilities\n'
                for i, p in enumerate(probs):
                    text += '\n    %i  =  %.1f %%' % (i, p*100)

                self.text_probs_plot.set_text(text)


    def _onClickRelease(self, event):
        self.mouse_depressed = False
        self.sketch_coords['x'].append(numpy.nan)
        self.sketch_coords['y'].append(numpy.nan)


    def _onMotion(self, event):
        if self.mouse_depressed:
            self.sketch_coords['x'].append(event.xdata)
            self.sketch_coords['y'].append(event.ydata)

            self.sketch_plot.set_data((self.sketch_coords['x'], self.sketch_coords['y']))
            pyplot.draw()


    def load_neural_network(self, fname):
        self.neural_network = NumberClassifier()
        self.neural_network.load_weights_biases(fname)




if __name__ == '__main__':

    dc = DigitCanvas()
    dc.load_neural_network('../output/neural_network_28x28_backprop9000.json')



