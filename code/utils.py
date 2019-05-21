
import sys
import time
import gzip
import glob
import json
import numpy
import struct
from astropy.io import fits
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages




def load_data(set_type='train'):
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