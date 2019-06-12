
import utils
import numpy
import matplotlib
from scipy import signal
from matplotlib import pyplot
from NumberClassifier import NumberClassifier


class DigitCanvas(object):
    '''Class for creating an interactive canvas for draing digits'''

    def __init__(self, image_size=28):

        self.fig = pyplot.figure(figsize=(10,5))
        self.ax_canvas = pyplot.subplot2grid((7,12), (0,0), rowspan=7, colspan=5, aspect='equal', label='canvas')
        self.ax_clear = pyplot.subplot2grid((7,12), (0,6), rowspan=1, colspan=2, facecolor='#cccccc', label='clear button')
        self.ax_submit = pyplot.subplot2grid((7,12), (1,6), rowspan=1, colspan=2, facecolor='#cccccc', label='submit button')
        self.ax_probs = pyplot.subplot2grid((7,12), (3, 6), rowspan=4, colspan=6, label='probabilities')
        self.fig.subplots_adjust(left=0.03, right=0.97, hspace=0.4, wspace=0.2)

        self.neural_network = None

        ###  turning off axis labels for canvas and buttons
        for ax in [self.ax_canvas, self.ax_clear, self.ax_submit]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ###  labeling probabilities panel
        self.ax_probs.set_xlabel('Digit Label', size=14, family='serif')
        self.ax_probs.set_ylabel('Probability (%)', size=14, family='serif')
        self.ax_probs.minorticks_on()
        self.ax_probs.set_ylim(-4, 112)
        self.ax_probs.tick_params(axis='x', which='minor', bottom=False)

        self.ax_probs.xaxis.set_ticks(range(10))
        for i in numpy.arange(-0.5,10,1):
            self.ax_probs.axvline(i, color='gray', lw=1, alpha=0.5)

        ###  initializing 2D image array
        self.image_size = image_size
        self.image_data = numpy.zeros((image_size, image_size))


        ###  adding gridlines for pixels
        self.ax_canvas.axis([-0.5, image_size-0.5, -0.5, image_size-0.5])
        for i in range(image_size):
            self.ax_canvas.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
            self.ax_canvas.axhline(i-0.5, color='gray', lw=1, alpha=0.3)

        ###  adding button labels
        self.ax_canvas.set_title('Draw a single-digit number', size=14, family='serif')
        self.ax_clear.text(0.5, 0.5, 'CLEAR', fontweight='bold', family='sans-serif', size=14, ha='center', va='center')
        self.ax_submit.text(0.5, 0.5, 'SUBMIT', fontweight='bold', family='sans-serif', size=14, ha='center', va='center')

        self.last_panel_clicked = None
        self.mouse_depressed = False
        self.sketch_coords = {'x':[], 'y':[]}
        self.sketch_collection = []

        cid = self.fig.canvas.mpl_connect('button_press_event', self._onClick)
        cid = self.fig.canvas.mpl_connect('button_release_event', self._onClickRelease)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self._onMotion)


        self.sketch_plot = self.ax_canvas.plot(0, 0, color='k', lw=10)[0]
        self.image_plot = self.ax_canvas.imshow(self.image_data, interpolation='none', vmin=0, vmax=1, cmap=pyplot.cm.Greens)
        self.pdf_curve = self.ax_probs.step(range(-1,11), numpy.zeros(12)-100, where='mid', color='#1f77b4', lw=2)[0]
        self.text_probs_plot = [self.ax_probs.text(x, -10, '', size=12, ha='center', va='bottom',  color='#1f77b4', fontweight='bold', family='sans-serif') for x in range(10)]


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

            self.pdf_curve.set_ydata(numpy.zeros(12)-100)
            for t in self.text_probs_plot:
                t.set_text('')

            self.ax_probs.set_title('')
            pyplot.draw()

        ###  handling SUBMIT button click
        if (event.inaxes is not None) and (event.inaxes.get_label() == 'submit button') and (self.last_panel_clicked != 'submit button'):

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

                ###  inverting image along y-axis and flattening (MNIST data format)
                im = self.image_data[::-1, :].flatten()

                ###  guessing and getting probabilities
                guess = self.neural_network.guess_digit(im)
                probs = 100 * self.neural_network.get_probabilities(im)

                ###  plotting PDF curve and probabilities text
                self.pdf_curve.set_ydata([0]+probs.tolist()+[0])
                for i in range(10):
                    t = self.text_probs_plot[i]
                    t.set_text('%i' % probs[i])
                    t.set_y(probs[i])

                ###  indicating best guess
                if probs.max() > 80:
                    title = 'Looks like %i to me!' % guess
                elif probs.max() > 70:
                    title = 'Looks mostly like %i to me' % guess
                elif probs.max() > 60:
                    title = 'Looks kind of like %i to me' % guess
                else:
                    title = 'I\'m not really sure, but I guess %i?' % guess

                self.ax_probs.set_title(title, color='r', fontweight='bold', family='serif', size=16)

        if (event.inaxes is not None):
            self.last_panel_clicked = event.inaxes.get_label()


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
    dc.load_neural_network('../output/neural_network_28x28.json')
    pyplot.show()
