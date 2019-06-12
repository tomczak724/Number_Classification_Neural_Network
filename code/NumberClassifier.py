
import os
import pdb
import sys
import time
import json
import utils
import numpy
import itertools
from matplotlib import pyplot


def main():

    data_train, labels_train = utils.load_MNIST_data(set_type='train', centering=False)
    #data_test, labels_test = utils.load_MNIST_data(set_type='test', centering=False)

    n_neurons = [10, 15, 20, 25, 30]
    for n1, n2 in itertools.product(n_neurons, n_neurons):

        print('\nTraining %ix%i network\n' % (n1, n2))

        NN = NumberClassifier(N_neurons=[n1, n2], N_backprop=500)
        NN.train_on(data_train, labels_train)



class NumberClassifier(object):

    def __init__(self, N_neurons=[16, 16], N_backprop=10, eta=5., verbose=True):
        '''
        Description
        -----------
            This class initializes and executes the infrastructure
            designed to train a neural network aimed at classifying
            hand-written digets from the MNIST dataabse [1].

        Parameters
        ----------
            N_neurons : array
                Number of neurons in hidden layers
            N_backprop : int
                Number of back-propagations to perform for training
            eta : float
                Learning rate

        References
        ----------
            [1] MNIST database
                http://yann.lecun.com/exdb/mnist/
            [2] 3Blue1Brown
                https://www.youtube.com/watch?v=aircAruvnKk
        '''

        self.N_neurons = N_neurons
        self.N_layers = len(N_neurons) + 2
        self.N_backprop = N_backprop
        self.eta = eta
        self.verbose = verbose

        ###  weights and biases for each neuron in each layer
        self.weights = None
        self.biases = None

        self.training_progress = {'iteration': [], 
                                  'n_unique': [], 
                                  'n_correct': [], 
                                  'f_correct': []}


    def load_weights_biases(self, fname='../output/neuron_weights_biases.json'):
        '''Loads existing neuron weights and biases from previous training.
        Will automatically adopt the corresponding dimensionality
        '''

        neurons = json.load(open(fname, 'r'))
        self.weights = [numpy.array(w) for w in neurons['weights']]
        self.biases = [numpy.array(b) for b in neurons['biases']]

        self.N_neurons = [len(b) for b in self.biases[:-1]]
        self.N_layers = len(self.N_neurons) + 2


    def load_training_progress(self, fname='../output/training_progress.json'):
        '''Loads existing progress file from previous training'''

        self.training_progress = json.load(open(fname, 'r'))


    def load_training_progress_csv(self, fname='../output/training_progress.json'):
        '''Loads existing progress file from previous training'''

        arr = numpy.loadtxt(fname, delimiter=',', skiprows=1)
        self.training_progress['iteration'] = arr[:,0].astype(int)
        self.training_progress['n_unique'] = arr[:,1].astype(int)
        self.training_progress['n_correct'] = arr[:,2].astype(int)
        self.training_progress['f_correct'] = arr[:,3].astype(int)


    def train_on(self, data_train, labels_train, n_subsamples=10):

        if self.verbose == True:
            utils.print_data_stats(data_train, labels_train)

        ###  reading dimensions of training data
        n_digits = data_train.shape[0]
        n_pixels = data_train.shape[1]

        ###  breaking training data into subsamples
        subsample_inds = numpy.linspace(0, n_digits, n_subsamples+1).astype(int)

        ###  initializing random weights/biases if not provided
        if self.weights is None:
            self.weights = self._initialize_random_weights(n_pixels)
        if self.biases is None:
            self.biases = self._initialize_random_biases(n_pixels)


        ###  running through training iterations
        if self.verbose == True:
            print('running %i iterations...' % self.N_backprop)
        for i_backprop in range(self.N_backprop):

            tstart = time.time()

            ###  grabbing a random subsample from the training data
            i_ss = numpy.random.randint(0, n_subsamples)
            i_1, i_2 = subsample_inds[i_ss], subsample_inds[i_ss+1]
            data_train_sub = data_train[i_1:i_2]
            labels_train_sub = labels_train[i_1:i_2]
            n_digits_sub = data_train_sub.shape[0]

            ###  array for storing guesses
            guesses = numpy.zeros(n_digits_sub)

            ###  arrays for storing gradients
            grad_weights = [numpy.zeros(w.shape) for w in self.weights]
            grad_biases = [numpy.zeros(b.shape) for b in self.biases]


            ###  looping through digits
            for i_digit in range(n_digits_sub):

                ###  setting target activations for output layer
                target = numpy.zeros(10)
                target[labels_train_sub[i_digit]] = 1

                ###  forward propagating data through network
                zs, activations = self._forward_propagate(data_train_sub[i_digit])

                ###  back propagating, calculating gradients to biases and weights
                grad_weights_i, grad_biases_i = self._back_propagate(zs, activations, target)

                ###  adding this digit's gradients to master gradient array
                for i_layer in range(self.N_layers-1):
                    grad_weights[i_layer] += grad_weights_i[i_layer]
                    grad_biases[i_layer] += grad_biases_i[i_layer]

                ###  adding best-guess for this digit to array
                a = activations[-1]
                guesses[i_digit] = a.tolist().index(a.max())


            ###  applying gradient descent
            for i_layer in range(self.N_layers-1):

                ###  averaging the gradient sums
                grad_biases[i_layer] /= n_digits_sub
                grad_weights[i_layer] /= n_digits_sub

                ###  updating weights and biases
                self.biases[i_layer] -= self.eta * grad_biases[i_layer]
                self.weights[i_layer] -= self.eta * grad_weights[i_layer]


            ###  writing summary stats to file
            n_unique = numpy.unique(guesses).shape[0]
            n_correct = (guesses - labels_train_sub).tolist().count(0)

            self.training_progress['iteration'].append(i_backprop)
            self.training_progress['n_unique'].append(n_unique)
            self.training_progress['n_correct'].append(n_correct)
            self.training_progress['f_correct'].append(n_correct/n_digits_sub)

            self.save_network()
            self.save_training_progress()

            tstop = time.time()

            if self.verbose:
                print('%4i, %2i, %4i, %.3f (%i seconds elapsed)' % (i_backprop+1, n_unique, n_correct, n_correct/n_digits_sub, tstop-tstart))


    def _initialize_random_weights(self, len_data):
        '''Initializes random weights for all layers of the network'''
        weights = []
        for n1, n2 in zip([len_data]+self.N_neurons, self.N_neurons+[10]):
            w = numpy.random.randn(n2, n1) / n1**0.5
            weights.append(w)

        return weights


    def _initialize_random_biases(self, len_data):
        '''Initializes random biases for all layers of the network'''
        biases = []
        for n1, n2 in zip([len_data]+self.N_neurons, self.N_neurons+[10]):
            b = numpy.random.randn(n2)
            biases.append(b)

        return biases


    def _forward_propagate(self, data):

        ###  looping over network layers
        zs, activations = [], [data]
        for i_layer in range(self.N_layers-1):

            ###  calculating weighted sum + bias
            z = numpy.dot(self.weights[i_layer], data) + self.biases[i_layer]
            zs.append(z)

            ###  applying logistic transform
            data = utils.logistic(z)

            ###  storing activations
            activations.append(data)

        return zs, activations


    def _back_propagate(self, zs, activations, target):

        ###  backpropagating, calculating gradients to biases and weights
        grad_biases_i = [numpy.zeros(b.shape) for b in self.biases]
        grad_weights_i = [numpy.zeros(w.shape) for w in self.weights]

        delta = 2 * (activations[-1] - target) * utils.logistic_prime(zs[-1])
        grad_weights_i[-1] = numpy.array([delta*a for a in activations[-2]]).T
        grad_biases_i[-1] = delta

        for i_layer in range(2, self.N_layers):
            z = zs[-i_layer]
            delta = numpy.dot(self.weights[-i_layer+1].T, delta) * utils.logistic_prime(z)
            grad_weights_i[-i_layer] = numpy.array([delta*a for a in activations[-i_layer-1]]).T
            grad_biases_i[-i_layer] = delta

        return grad_weights_i, grad_biases_i


    def save_network(self, fname=None):
        '''Save the current neural network weights and biases to JSON file'''

        ###  generate output filename if not provided
        if fname is None:
            str_layers = ''
            for n in self.N_neurons:
                str_layers += '%ix' % n

            fname = '../output/neural_network_%s.json' % str_layers[:-1]

        weights = [w.tolist() for w in self.weights]
        biases = [b.tolist() for b in self.biases]

        neurons = {'weights':weights,
                   'biases':biases}

        with open(fname, 'w') as outer_json:
            json.dump(neurons, outer_json)


    def save_training_progress(self, fname=None):
        '''Save the progress from training to JSON file'''

        ###  generate output filename if not provided
        if fname is None:
            str_layers = ''
            for n in self.N_neurons:
                str_layers += '%ix' % n

            fname = '../output/training_progress_%s.json' % str_layers[:-1]

        with open(fname, 'w') as outer_json:
            json.dump(self.training_progress, outer_json)


    def save_training_progress_csv(self, fname=None):
        '''Save the progress from training to csv file'''

        ###  generate output filename if not provided
        if fname is None:
            str_layers = ''
            for n in self.N_neurons:
                str_layers += '%ix' % n

            fname = '../output/training_progress_%s.csv' % str_layers[:-1]

        ###  converting training progress to array
        arr = numpy.zeros((len(self.training_progress['iteration']), len(self.training_progress)))
        for i, vals in enumerate(self.training_progress.values()):
            arr[:,i] = vals

        numpy.savetxt(fname, arr, delimiter=',', header=','.join(self.training_progress.keys()), fmt=('%i', '%i', '%i', '%.4f'))


    def get_probabilities(self, data):
        '''Returns probabilities for all digits from the data for the given digit'''
        zs, activations = self._forward_propagate(data)
        return activations[-1] / activations[-1].sum()


    def guess_digit(self, data):
        '''Returns the best guess from the data for the given digit'''
        probs = self.get_probabilities(data)
        return probs.tolist().index(probs.max())


    def plot_guess_overview(self, data, true_label):

        #zs, activations = self._forward_propagate(data)
        zs, activations = self._forward_propagate(data)
        probabilities = activations[-1] / activations[-1].sum()

        ###  initializing figure
        fig, (ax1, ax2) = pyplot.subplots(ncols=2, figsize=(9.5, 4.5))
        fig.subplots_adjust(left=0.08, top=0.92, right=0.96, bottom=0.1)

        ax1.set_xlabel('Digit Label', size=14)
        ax1.set_ylabel('Probability', size=14)
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='minor', bottom=False)

        ###  plotting digit image
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        s = int(data.shape[0]**0.5)
        ax2.imshow(data.reshape((s,s)), interpolation='none', cmap=pyplot.cm.Greens, vmin=data.min(), vmax=data.max())
        for i in range(28+1):
            ax2.axvline(i-0.5, color='gray', lw=1, alpha=0.3)
            ax2.axhline(i-0.5, color='gray', lw=1, alpha=0.3)

        t = ax2.text(0.02, 0.98, 'true label = %i' % true_label, transform=ax2.transAxes, ha='left', va='top', size=14)

        ###  plotting guess probabilities
        ax1.step(range(-1,11), [0]+probabilities.tolist()+[0], 
                 where='mid', color='#1f77b4', lw=2)

        ax1.xaxis.set_ticks(range(10))
        for i in numpy.arange(-0.5,10,1):
            ax1.axvline(i, color='gray', lw=1, alpha=0.5)

        for i in range(10):
            p = probabilities[i]
            ax1.text(i, p, '%i'%(100*p), ha='center', va='bottom', color='#1f77b4', fontweight='bold', family='sans-serif')

        return fig, (ax1, ax2)



if __name__ == '__main__':

    print('')
    choice = input('Run main() script? [y/n]: ')

    if choice == 'y':
        main()


