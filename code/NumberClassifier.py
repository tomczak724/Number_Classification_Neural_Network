
import pdb
import sys
import json
import utils
import numpy


def main():

    #data_train, labels_train = utils.load_MNIST_data(set_type='train')
    data_test, labels_test = utils.load_MNIST_data(set_type='test')

    n = 5000
    inds = numpy.arange(0, len(labels_test))
    numpy.random.shuffle(inds)
    inds = inds[0:n]

    nc1 = NumberClassifier(N_backprop=100)
    nc1.train_on(data_test[inds], labels_test[inds])

    nc1.save_network()
    nc1.save_training_results()



class NumberClassifier(object):

    def __init__(self, N_backprop, N_neurons=[16, 16], eta=5., verbose=True):
        '''
        Description
        -----------
            This class initializes and executes the infrastructure
            designed to train a neural network aimed at classifying
            hand-written digets from the MNIST dataabse [1].

        Parameters
        ----------
            N_backprop : int
                Number of back-propagations to perform for training
            N_neurons : array
                Number of neurons in hidden layers
            eta : float
                Learning rate

        References
        ----------
            [1] MNIST database
                http://yann.lecun.com/exdb/mnist/
        '''

        self.N_backprop = N_backprop
        self.N_neurons = N_neurons
        self.N_layers = len(N_neurons) + 2
        self.eta = eta
        self.verbose = verbose

        self.weights = None
        self.biases = None

        self.train_progress = {'iteration': [], 
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



    def train_on(self, data_train, labels_train):

        if self.verbose == True:
            utils.print_data_stats(data_train, labels_train)

        ###  reading dimensions of training data
        n_digits = data_train.shape[0]
        n_pixels = data_train.shape[1]

        ###  initializing random weights/biases if not provided
        if self.weights is None:
            self.weights = self._initialize_random_weights(n_pixels)
        if self.biases is None:
            self.biases = self._initialize_random_biases(n_pixels)


        ###  running through training iterations
        if self.verbose == True:
            print('running %i iterations...' % self.N_backprop)
        for i_backprop in range(self.N_backprop):

            ###  array for storing guesses
            guesses = numpy.zeros(n_digits)

            ###  arrays for storing gradients
            grad_weights = [numpy.zeros(w.shape) for w in self.weights]
            grad_biases = [numpy.zeros(b.shape) for b in self.biases]


            ###  looping through digits
            for i_digit in range(n_digits):

                ###  setting target activations for output layer
                target = numpy.zeros(10)
                target[labels_train[i_digit]] = 1

                ###  forward propagating data through network
                zs, activations = self._forward_propagate(data_train[i_digit])

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
                grad_biases[i_layer] /= n_digits
                grad_weights[i_layer] /= n_digits

                ###  updating weights and biases
                self.biases[i_layer] -= self.eta * grad_biases[i_layer]
                self.weights[i_layer] -= self.eta * grad_weights[i_layer]



            ###  writing summary stats to file
            n_unique = numpy.unique(guesses).shape[0]
            n_correct = (guesses - labels_train).tolist().count(0)

            self.train_progress['iteration'].append(i_backprop)
            self.train_progress['n_unique'].append(n_unique)
            self.train_progress['n_correct'].append(n_correct)
            self.train_progress['f_correct'].append(n_correct/n_digits)

            if self.verbose:
                print('%4i, %2i, %4i, %.3f' % (i_backprop+1, n_unique, n_correct, n_correct/n_digits))


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

            fname = '../output/neural_network_%s_backprop%i.json' % (str_layers[:-1], self.N_backprop)

        weights = [w.tolist() for w in self.weights]
        biases = [b.tolist() for b in self.biases]

        neurons = {'weights':weights,
                   'biases':biases}

        with open(fname, 'w') as outer_json:
            json.dump(neurons, outer_json)


    def save_training_results(self, fname=None):
        '''Save the results from training to JSON file'''

        ###  generate output filename if not provided
        if fname is None:
            str_layers = ''
            for n in self.N_neurons:
                str_layers += '%ix' % n

            fname = '../output/training_results_%s_backprop%i.json' % (str_layers[:-1], self.N_backprop)

        with open(fname, 'w') as outer_json:
            json.dump(self.train_progress, outer_json)




if __name__ == '__main__':

    print('')
    choice = input('Run main() script? [y/n]: ')

    if choice == 'y':
        main()


