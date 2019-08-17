
import os
import sys
import glob
import gzip
import json
import numpy
import struct
import matplotlib
from scipy import ndimage
from matplotlib import pyplot


def load_MNIST_data(set_type='train', normalize=True):
    '''Reads the MNIST data binaries and returns as arrays.
    Specify either "train" or "test" data'''

    if set_type.lower() not in ['train', 'test']:
        raise ValueError('Parameter set_type must be in [train, test]')

    ###  reading binaries
    if set_type.lower() == 'train':
        fopen_labels = gzip.open('../data/train-labels-idx1-ubyte.gz', 'rb')
        fopen_images = gzip.open('../data/train-images-idx3-ubyte.gz', 'rb')
    elif set_type.lower() == 'test':
        fopen_labels = gzip.open('../data/t10k-labels-idx1-ubyte.gz', 'rb')
        fopen_images = gzip.open('../data/t10k-images-idx3-ubyte.gz', 'rb')

    junk, Ntot = struct.unpack(">II", fopen_labels.read(8))
    junk, Ntot, Nrows, Ncols = struct.unpack('>IIII', fopen_images.read(16))

    ###  reading digit labels
    digits_labels = numpy.array(struct.unpack('B'*Ntot, fopen_labels.read()))

    ###  initializing output data array
    digits_data = numpy.zeros((Ntot, Nrows*Ncols))

    ###  looping over digits
    print('')
    for i in range(Ntot):

        text = '\r[%2i%%] reading MNIST %s data' % (i*100./(Ntot-1), set_type)
        sys.stdout.write(text)
        sys.stdout.flush()

        byte_data = fopen_images.read(Nrows*Ncols)
        image_data = struct.unpack('B'*(Nrows*Ncols), byte_data)
        digits_data[i] = numpy.array(image_data)

    print('')
    fopen_labels.close()
    fopen_images.close()

    ###  normalizing data
    if normalize == True:
        for d in digits_data:
            d /= d.max()

    return (digits_data, digits_labels)


def logistic(x):
    return 1. / (1 + numpy.exp(-x))


def logistic_prime(x):
    return logistic(x) * (1 - logistic(x))


def print_data_stats(data, labels):
    '''Prints some meta stats associated with the input dataset'''

    n_digits, n_pixels = data.shape


    text  = 'Number of images: %i\n' % n_digits
    text += '      Image size: %ix%i pixels\n' % (n_pixels**0.5, n_pixels**0.5)

    text += '\n'
    text += 'Numeral breakdown\n'
    for i in range(10):
        text += '  Instances of "%i" is %i (%i%%)\n' % (i, labels.tolist().count(i), 100.*labels.tolist().count(i)/n_digits)

    print(text)


def center_image(data):
    '''Calculates the intensity-weighted center and returns a centered image'''

    s = int(data.shape[0]**0.5)
    xgrid, ygrid = numpy.meshgrid(numpy.arange(0.5, s+0.5), numpy.arange(0.5, s+0.5))

    xcenter = numpy.average(xgrid, weights=data.reshape((s,s)))
    ycenter = numpy.average(ygrid, weights=data.reshape((s,s)))

    dx = s/2. - xcenter
    dy = s/2. - ycenter

    ###  applying translational shifts
    data = xshift(data.reshape((s,s)), dx)
    data = yshift(data.reshape((s,s)), dy)

    return data.flatten()


def xshift(data, dx):
    '''
    Description
    -----------
    Performs a translational shift on the input image
    along the x-axis. Allows for subpixel precision.

    Parameters
    ----------
    data : numpy.array
        Two dimensional image
    dx : float
        Size of translational shift in pixels

    Returns
    -------
    output : numpy.array
        Shifted image
    '''

    sy, sx = data.shape
    output = numpy.zeros(data.shape)

    for i_x in range(data.shape[1]):

        weights = numpy.zeros(data.shape)

        if 0 <= int(i_x-numpy.ceil(dx)) < sx:
            weights[:, int(i_x-numpy.ceil(dx))] = dx%1
        if 0 <= int(i_x-numpy.floor(dx)) < sx:
            weights[:, int(i_x-numpy.floor(dx))] = 1-dx%1

        if weights.sum() > 0:
            output[:, i_x] = numpy.sum(data*weights, axis=1)

    return output


def yshift(data, dy):
    '''
    Description
    -----------
    Performs a translational shift on the input image
    along the y-axis. Allows for subpixel precision.

    Parameters
    ----------
    data : numpy.array
        Two dimensional image
    dy : float
        Size of translational shift in pixels

    Returns
    -------
    output : numpy.array
        Shifted image
    '''

    sy, sx = data.shape
    output = numpy.zeros(data.shape)

    for i_y in range(data.shape[0]):

        weights = numpy.zeros(data.shape)

        if 0 <= int(i_y-numpy.ceil(dy)) < sy:
            weights[int(i_y-numpy.ceil(dy)), :] = dy%1
        if 0 <= int(i_y-numpy.floor(dy)) < sy:
            weights[int(i_y-numpy.floor(dy)), :] = 1-dy%1

        if weights.sum() > 0:
            output[i_y, :] = numpy.sum((data*weights).T, axis=1)

    return output


def calc_xy_center(data):
    '''Calculates the intensity-weighted center of the input image'''

    s = int(data.shape[0]**0.5)
    xgrid, ygrid = numpy.meshgrid(numpy.arange(0.5, s+0.5), numpy.arange(0.5, s+0.5))

    xcenter = numpy.average(xgrid, weights=data.reshape((s,s)))
    ycenter = numpy.average(ygrid, weights=data.reshape((s,s)))

    return xcenter, ycenter


def plot_example_images(data, labels):
    '''Plots a handful of example MNIST images'''

    nrows, ncols = 2, 6
    fig, axs = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9*1.08*nrows/ncols))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, hspace=0.2, wspace=0.05)

    ###  configuring panels
    for ax in axs.flatten():

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        for i in range(28):
            ax.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
            ax.axhline(i-0.5, color='gray', lw=1, alpha=0.3)

    ###  selecting random images to plot
    n = len(axs.flatten())
    for i_ax, i_digit in enumerate(numpy.random.randint(0, len(labels), n)):
        data_i = data[i_digit]
        label_i = labels[i_digit]

        ax = axs.flatten()[i_ax]
        d = data_i.reshape((28,28))[::-1, :]
        ax.imshow(d, interpolation='none', cmap=pyplot.cm.Greens)

        ax.text(0.03, 1.01, 'label = %i' % label_i, size=11, weight='bold', color='r', transform=ax.transAxes, ha='left', va='bottom')

    return fig, axs


def plot_centering_example():
    '''Generates a plot that illustrates the centering of the MNIST digit images'''

    data_test, labels_test = load_MNIST_data(set_type='test', centering=False)
    data = data_test[9322]


    fig, (ax1, ax2) = pyplot.subplots(ncols=2, figsize=(9.5,4.5))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.92)

    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    ax1.set_title('Original image', size=16)
    ax2.set_title('Centered image', size=16)

    cmap = pyplot.cm.Greens
    ax1.imshow(data.reshape((28,28)), interpolation='none', cmap=cmap, vmin=0, vmax=data.max())
    ax2.imshow(center_image(data).reshape((28,28)), interpolation='none', cmap=cmap, vmin=0, vmax=data.max())


    ax1.axvline((28-1)/2., color='k', lw=2)
    ax1.axhline((28-1)/2., color='k', lw=2)
    ax2.axvline((28-1)/2., color='k', lw=2)
    ax2.axhline((28-1)/2., color='k', lw=2)
    for i in range(28+1):
        ax1.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
        ax1.axhline(i-0.5, color='gray', lw=1, alpha=0.3)
        ax2.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
        ax2.axhline(i-0.5, color='gray', lw=1, alpha=0.3)

    xc, yc = calc_xy_center(data)
    ax1.plot(xc-0.5, yc-0.5, color='r', marker='x', ms=8, mew=3)

    return fig, (ax1, ax2)


def plot_centering_offsets():
    '''Generates a plot that shows the distribution of centering offsets'''

    if os.path.exists('../output/centering_offsets.csv'):
        d = numpy.loadtxt('../output/centering_offsets.csv', skiprows=1, delimiter=',')
        xoffsets = d[:,1]
        yoffsets = d[:,2]

    else:
        data_train, labels_train = load_MNIST_data(set_type='train', centering=False)

        xoffsets = numpy.zeros(len(labels_train))
        yoffsets = numpy.zeros(len(labels_train))

        for i_digit in range(len(labels_train)):

            text = '\r[%2i%%] calculating centering offsets' % (i_digit*100./(len(labels_train)-1))
            sys.stdout.write(text)
            sys.stdout.flush()

            xc, yc = calc_xy_center(data_train[i_digit])
            xoffsets[i_digit] = xc - (28/2.)
            yoffsets[i_digit] = yc - (28/2.)


    fig = pyplot.figure(figsize=(6., 6.))
    ax1 = pyplot.subplot2grid((4, 4), (0, 0), rowspan=1, colspan=3)
    ax2 = pyplot.subplot2grid((4, 4), (1, 3), rowspan=3, colspan=1)
    ax0 = pyplot.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3, sharex=ax1, sharey=ax2)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.13, right=0.98, bottom=0.13, top=0.98)

    for ax in [ax0, ax1, ax2]:
        ax.minorticks_on()

    ax0.set_xlabel('$\Delta$x (pixels)', size=14)
    ax0.set_ylabel('$\Delta$y (pixels)', size=14)
    ax1.set_ylabel('Number', size=14)
    ax2.set_xlabel('Number', size=14)

    xlo, xhi = xoffsets.min(), xoffsets.max()
    ylo, yhi = yoffsets.min(), yoffsets.max()
    ax0.axis([xlo-(xhi-xlo)*0.9, xhi+(xhi-xlo)*0.9, 
              ylo-(yhi-ylo)*0.9, yhi+(yhi-ylo)*0.9])

    ax0.axvline(0, color='k', lw=1)
    ax0.axhline(0, color='k', lw=1)
    ax1.axvline(0, color='k', lw=1)
    ax2.axhline(0, color='k', lw=1)

    ax0.plot(xoffsets, yoffsets, ls='', marker='o', ms=2, mfc='r', mec='k', mew=1)
    ax1.hist(xoffsets, histtype='step', color='r', lw=2, bins=35)
    ax2.hist(yoffsets, histtype='step', color='r', lw=2, bins=35, orientation='horizontal')

    ax1.set_ylim(10, ax1.axis()[3])
    ax2.set_xlim(10, ax2.axis()[1])

    ax0.xaxis.set_tick_params(rotation=30)
    ax0.yaxis.set_tick_params(rotation=30)
    ax1.yaxis.set_tick_params(rotation=30)
    ax2.xaxis.set_tick_params(rotation=30)



    return fig, (ax0, ax1, ax2)


def plot_cartoon_neural_network():


    fig, ax = pyplot.subplots(figsize=(6.5, 6.5))
    fig.subplots_adjust(left=0, top=1, right=1, bottom=0)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')

    ###  plotting input layer of nodes
    n_input_nodes = 8
    x_input = 0
    y_input = numpy.arange(n_input_nodes) - (n_input_nodes-1)/2.
    radius = 0.3
    for i in range(n_input_nodes):
        c = matplotlib.patches.Circle((x_input, y_input[i]), radius=radius, color='gray')
        ax.add_artist(c)

    ###  adding title
    t = ax.text(x_input, (n_input_nodes+1)/2., 'Input\nLayer', size=12, ha='center', va='center')



    ###  plotting layer 1 of nodes
    n_layer1_nodes = 5
    x_layer1 = 4
    y_layer1 = numpy.arange(n_layer1_nodes) - (n_layer1_nodes-1)/2.
    for i in range(n_layer1_nodes):
        c = matplotlib.patches.Circle((x_layer1, y_layer1[i]), radius=radius, color='gray')
        ax.add_artist(c)

    ###  adding title
    t = ax.text(x_layer1, (n_layer1_nodes+1)/2., 'Layer 1', size=12, ha='center', va='center')



    ###  plotting layer 2 of nodes
    n_layer2_nodes = 5
    x_layer2 = 8
    y_layer2 = numpy.arange(n_layer2_nodes) - (n_layer2_nodes-1)/2.
    for i in range(n_layer2_nodes):
        c = matplotlib.patches.Circle((x_layer2, y_layer2[i]), radius=radius, color='gray')
        ax.add_artist(c)

    ###  adding title
    t = ax.text(x_layer2, (n_layer2_nodes+1)/2., 'Layer 2', size=12, ha='center', va='center')



    ###  plotting output layer of nodes
    n_output_nodes = 3
    x_output = 12
    y_output = numpy.arange(n_output_nodes) - (n_output_nodes-1)/2.
    for i in range(n_output_nodes):
        c = matplotlib.patches.Circle((x_output, y_output[i]), radius=radius, color='gray')
        ax.add_artist(c)

    ###  adding title
    t = ax.text(x_output, (n_output_nodes+1)/2., 'Output\nLayer', size=12, ha='center', va='center')



    ax.axis([-1, 13, -7, 7])
    return fig


def plot_training_progress_cascade():


    fig, axs = pyplot.subplots(nrows=5, ncols=5, figsize=(7,6), sharex=True, sharey=True)

    fig.subplots_adjust(hspace=0, wspace=0, left=0.13, top=0.87, bottom=0.01, right=0.99)

    n = [10, 15, 20, 25, 30]
    for i, n1 in enumerate(n):
        for j, n2 in enumerate(n):
            print('plotting %ix%i network' % (n1, n2))
            fnames =  glob.glob('/media/removable/DEMOGORGON/storage/training_progress_%ix%i*' % (n1, n2))
            data_all = []
            for fname in fnames:
                data = json.load(open(fname, 'r'))
                data_all.append(data['f_correct'])
                axs[i][j].plot(data['iteration'], data['f_correct'], color='k', lw=1, alpha=0.15)

                axs[i][j].xaxis.set_ticks([])
                axs[i][j].yaxis.set_ticks([])

                if j == 0:
                    axs[i][j].set_ylabel('%i'%n1, size=18, rotation=0, ha='right', va='center')
                if i == 0:
                    axs[i][j].set_title('%i'%n2, size=18)

            axs[i][j].plot(data['iteration'], numpy.array(data_all).mean(axis=0), color='r', alpha=0.7, lw=1)


    ###  adding axes to host meta-labels for x- and y- axes
    ax_meta = fig.add_axes([0,0,1,1], facecolor='none', xticks=[], yticks=[])
    ax_meta.axis([0,1,0,1])

    ax_meta.text(0.55, 0.99, 'Neurons in Layer 2', size=22, ha='center', va='top')
    ax_meta.text(0.01, 0.45, 'Neurons in Layer 1', size=22, ha='left', va='center', rotation=90)

    return fig

