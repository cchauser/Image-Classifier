import theano
import theano.tensor as T
import numpy as np
import math

#This normalizes batches and is also a trainable feature.
#I use this because otherwise as the data goes through deep neural networks the outputs go to the extremes ((-1, 1) for tanh)
#By normalizing the outputs/inputs the outputs are actually quite varied

class Batch_Normalize(object):
    def __init__(self, input_shape, mode = 0, mom = .95):

        self.input_shape = input_shape
        self.mode = mode #0 for NN, 1 for CNN
        self.train = 1
        self.mom = .95
        self.channels = self.input_shape[1]

        gamma_bounds = 1./np.sqrt(self.channels)
        
        gamma = np.random.uniform(-gamma_bounds, gamma_bounds, self.channels)
        beta = np.zeros((self.channels))
        mean = np.zeros((self.channels))
        variance = np.ones((self.channels))

        self.gamma = theano.shared(gamma.astype(theano.config.floatX))
        self.beta = theano.shared(beta.astype(theano.config.floatX))
        self.mean = theano.shared(mean.astype(theano.config.floatX))
        self.variance = theano.shared(variance.astype(theano.config.floatX))

        self.gamma_m = theano.shared(np.zeros(gamma.shape).astype(theano.config.floatX))
        self.gamma_v = theano.shared(np.zeros(gamma.shape).astype(theano.config.floatX))

        self.beta_m = theano.shared(np.zeros(beta.shape).astype(theano.config.floatX))
        self.beta_v = theano.shared(np.zeros(beta.shape).astype(theano.config.floatX))
        

        self.param = [self.gamma, self.beta]
        self.param_m = [self.gamma_m, self.beta_m]
        self.param_v = [self.gamma_v, self.beta_v]


    def normalize(self, layer_in):
        eps = 1e-4

        if self.mode == 0:
            if self.train:
                new_mu = T.mean(layer_in, axis = 0)
                new_var = T.var(layer_in, axis = 0)

                new_norm = (layer_in - new_mu) / T.sqrt(new_var + eps)

                input_norm = self.gamma * new_norm + self.beta

                self.mean = self.mom * self.mean + (1-self.mom) * new_mu
                self.variance = self.mom * self.variance + (1-self.mom) * (self.input_shape[0]/(self.input_shape[0]-1) * new_var)
            else:
                input_norm = self.gamma * (layer_in - self.mean) / T.sqrt(self.variance + eps) + self.beta

        else:
            if self.train:
                new_mu = T.mean(layer_in, axis = (0,2,3))
                new_var = T.var(layer_in, axis = (0,2,3))

                self.mean = self.mom * self.mean + (1-self.mom) * new_mu
                self.variance = self.mom * self.variance + (1-self.mom) * (self.input_shape[0]/(self.input_shape[0]-1) * new_var)
            else:
                new_mu = self.mean
                new_var = self.variance
            new_mu = self.CNN_shape(new_mu)
            new_var = self.CNN_shape(new_var)
            new_gamma = self.CNN_shape(self.gamma)
            new_beta = self.CNN_shape(self.beta)

            input_norm = new_gamma * (layer_in - new_mu) / T.sqrt(new_var + eps) + new_beta

        return input_norm

    def CNN_shape(self, mat):
        transform = T.repeat(mat, self.input_shape[2] * self.input_shape[3])
        transform = transform.reshape((self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        return transform












                
