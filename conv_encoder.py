import os
import shelve
import gzip
import six.moves.cPickle as pickle
import warnings
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import conv2d_transpose
from theano.tensor.nnet import softmax
from theano.tensor.nnet import neighbours
from theano.tensor.nnet.nnet import softsign
from theano.tensor.nnet.nnet import relu
from theano.tensor.nnet.nnet import elu
from theano.tensor.nnet.nnet import sigmoid
from theano.tensor import tanh
import pandas as pd

from Batch_Normalize import Batch_Normalize



class Hidden_Layer(object):

    def __init__(self, layer_in, shape, input_shape, norm = False):
        out_shape = (input_shape[0], shape[1])

        self.BN = Batch_Normalize(out_shape, mode = 0)

        self.layer_in = layer_in
        
        ## Variables ##
        W_bounds = np.sqrt(6. / (shape[0] + shape[1]))
        
        W = np.random.uniform(-W_bounds, W_bounds, shape) * 4
        self.W = theano.shared(value = W.astype(theano.config.floatX))
        
        b = np.zeros((shape[1],))
        self.b = theano.shared(value = b.astype(theano.config.floatX))

        mW = np.zeros(W.shape)
        self.mW = theano.shared(value = mW.astype(theano.config.floatX))

        mb = np.zeros(b.shape)
        self.mb = theano.shared(value = mb.astype(theano.config.floatX))

        vW = np.zeros(W.shape)
        self.vW = theano.shared(value = mW.astype(theano.config.floatX))

        vb = np.zeros(b.shape)
        self.vb = theano.shared(value = vb.astype(theano.config.floatX))

        if norm:
            self.param = [self.W] + self.BN.param
            self.param_m = [self.mW] + self.BN.param_m
            self.param_v = [self.vW] + self.BN.param_v
        else:
            self.param = [self.W, self.b]
            self.param_m = [self.mW, self.mb]
            self.param_v = [self.vW, self.vb]

        ##Function##
        if norm:            
            out = T.dot(self.layer_in, self.W) + self.b
            self.out = self.BN.normalize(T.switch(out<0, 0.01 * out, out))
        else:
            out = T.dot(self.layer_in, self.W) + self.b
            self.out = out
##            self.out = T.switch(out<0, 0.01 * out, out)


class Convolution_Layer(object):

    def __init__(self, layer_in, filter_shape, input_shape, pool_shape = (2,2), pad = 'valid', norm = True):
        

        in_shape = filter_shape[1] * filter_shape[2] * filter_shape[3]
        out_shape = (filter_shape[0] *filter_shape[2] * filter_shape[3]) // (pool_shape[0] * pool_shape[1])
        shape = (input_shape[2] - filter_shape[2] + 1) // pool_shape[0]
        output_shape = (input_shape[0], filter_shape[0], shape, shape)
        
        self.BN = Batch_Normalize(output_shape, mode = 1)
        self.layer_in = layer_in
        
        ## Variables ##
        W_bounds = np.sqrt(6. / (in_shape + out_shape))
        
##        W = np.random.normal(loc = .5, scale = .5, size = filter_shape)
        W = np.random.uniform(-W_bounds, W_bounds, size = filter_shape)# * 4
        self.W = theano.shared(value = W.astype(theano.config.floatX))
        
        b = np.zeros((filter_shape[0],))
        self.b = theano.shared(value = b.astype(theano.config.floatX))

        mW = np.zeros(W.shape)
        self.mW = theano.shared(value = mW.astype(theano.config.floatX))

        mb = np.zeros(b.shape)
        self.mb = theano.shared(value = mb.astype(theano.config.floatX))

        vW = np.zeros(W.shape)
        self.vW = theano.shared(value = mW.astype(theano.config.floatX))

        vb = np.zeros(b.shape)
        self.vb = theano.shared(value = vb.astype(theano.config.floatX))

        if norm:
            self.param = [self.W] + self.BN.param
            self.param_m = [self.mW] + self.BN.param_m
            self.param_v = [self.vW] + self.BN.param_v
        else:
            self.param = [self.W, self.b]
            self.param_m = [self.mW, self.mb]
            self.param_v = [self.vW, self.vb]

        ## Functions ##
        c_o = conv2d(input = self.layer_in,
                     filters = self.W,
                     filter_shape = filter_shape,
                     input_shape = input_shape,
                     border_mode = pad)

        self.p_o = pool.pool_2d(input = c_o,
                                ws = pool_shape,
                                ignore_border = False)

        if norm:
            self.out = self.BN.normalize(T.switch(self.p_o<0, 0.01 * self.p_o, self.p_o))
        else:
            out = self.p_o + self.b.dimshuffle('x', 0, 'x', 'x')
            self.out = T.switch(out<0, 0.01 * out, out)


class Deconvolution_Layer(object):

    def __init__(self, layer_in, filter_shape, input_shape, output_shape, pool_shape = (2,2), pad = 'valid'):
        n_input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2, input_shape[3] * 2)
        self.BN = Batch_Normalize(n_input_shape, mode = 1)

        self.input_shape = input_shape

        #This unpools the features
        new = np.zeros((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
        new = T.repeat(layer_in, repeats = 2, axis = 2)
        new = T.repeat(new, repeats = 2, axis = 3)

        layer_in = new
        self.layer_in = self.BN.normalize(layer_in)
        
        in_shape = filter_shape[1] * filter_shape[2] * filter_shape[3]
        out_shape = (filter_shape[0] *filter_shape[2] * filter_shape[3]) // (pool_shape[0] * pool_shape[1])

        ## Variables ##
        W_bounds = np.sqrt(6. / (in_shape + out_shape))
        
##        W = np.random.normal(loc = 0.5, scale = .1, size = filter_shape)
        W = np.random.uniform(-W_bounds, W_bounds, size = filter_shape) * 4
        self.W = theano.shared(value = W.astype(theano.config.floatX))
        
##        b = np.ones((filter_shape[1],)) * .5
##        self.b = theano.shared(value = b.astype(theano.config.floatX))

        mW = np.zeros(W.shape)
        self.mW = theano.shared(value = mW.astype(theano.config.floatX))

##        mb = np.zeros(b.shape)
##        self.mb = theano.shared(value = mb.astype(theano.config.floatX))

        vW = np.zeros(W.shape)
        self.vW = theano.shared(value = mW.astype(theano.config.floatX))

##        vb = np.zeros(b.shape)
##        self.vb = theano.shared(value = vb.astype(theano.config.floatX))

        self.param = [self.W] + self.BN.param#, self.b]
        self.param_m = [self.mW] + self.BN.param_m#, self.mb]
        self.param_v = [self.vW] + self.BN.param_v#, self.vb]
        
        ## Functions ##
        c_o = conv2d_transpose(input = self.layer_in,
                               filters = self.W,
                               output_shape = output_shape,
                               border_mode = pad)

        self.out = sigmoid(c_o)# + self.b.dimshuffle('x', 0, 'x', 'x'))

            



class Encoder(object):

    def __init__(self, data, img_dim = 224, filters = [75,50,25], filter_size = 25, color = True, batch_size = 100,
                 pool_s = 2, learn_rate = .01, num_features = 30):
        """
        A typical Convolutional Autoencoder written in Theano 1.0.1

        data -- Currently accepts it as a list of tuples (this will change later)
        [(x_train, y_train), (x_validation, y_validation)] But at the moment I'm only using the x values. Like I said I'll change this later

        img_dim -- This should be the dimension of the images you're passing in. Currently only accepts square images.

        filters -- A list of filter channels to be used in the convolution and deconvolution steps

        filter_size -- This will be the size of the filter used in the convolution and deconvolution steps

        color -- Whether the images are in color

        batch_size -- Desired batch size

        pool_s -- This is the number of dimensions to reduce the input by during the max pooling step of convolution

        learn_rate -- Exactly what you think it means

        num_features -- This will be the desired number of features to be extracted from the hidden layer after training.
        You won't always get the exact number of features but it will be close.
        """
        if color:
            self.color = 3  #3 pixel values if color is true. (R, G, B)
        else:
            self.color = 1  #1 pixel value if color is false. (Grayscale intensity)
        
        self.x_train = data[0][0] / 255
        self.x_val = data[1][0] / 255
        self.learn_rate = learn_rate
        self.img_dim = img_dim
        self.filters = filters
        self.num_conv_layers = len(filters)
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.pool_shape = (pool_s, pool_s)
        self.final_shape = self.color * img_dim ** 2

        if(self.fails_test(self.img_dim, self.filter_size, self.num_conv_layers, pool_s)):
            warnings.warn("The dimensions will be off. Recommend changing filter size or consider reducing the number of convolutional layers")


        #Define the shapes of the filters and their inputs
        self.filter_shapes = [(self.filters[0], self.color, self.filter_size, self.filter_size)]
        for i in range(1,self.num_conv_layers):
            self.filter_shapes.append((self.filters[i], self.filters[i-1], self.filter_size, self.filter_size))

        self.deconv_filter_shapes = self.filter_shapes[::-1] #Reverse order of the filter shapes

        self.input_shapes = [(self.batch_size, self.color, self.img_dim, self.img_dim)]
        for i in range(self.num_conv_layers-1):
            shape = (self.input_shapes[i][2] - self.filter_size + 1) // pool_s
            if shape < 1:# or (self.img_shapes[i][2] - self.filter_size + 1) % 2 == 1:
                print("Placeholder error: The image shape is less than 1. Consider reducing number of convolution layers")
                print(self.input_shapes[i][2] - self.filter_size + 1)
                return
            self.input_shapes.append((self.batch_size, self.filters[i], shape, shape))

        self.deconv_output_shapes = self.input_shapes[::-1]

        shape = (self.input_shapes[-1][2] - self.filter_size + 1) // pool_s
        dc_in = [(self.batch_size, self.filters[-1], shape, shape)]
        self.deconv_input_shapes = dc_in + self.input_shapes[:0:-1]


        h_in = self.filters[-1] * (shape ** 2)
        self.hidden_shapes = [(h_in, num_features), (num_features, h_in)]
##        while self.hidden_shapes[-1][1] > num_features:
##            self.hidden_shapes.append((self.hidden_shapes[-1][1], int(self.hidden_shapes[-1][1]*.66)))
##        length = len(self.hidden_shapes)
##        i = 1
##        container = []
##
##        while i <= length:
##            backwards = self.hidden_shapes[-i][::-1]
##            container.append(backwards)
##            i += 1
##        self.hidden_shapes += container

        self.hidden_input_shapes = [(self.batch_size, h_in)]
        for i in range(1, len(self.hidden_shapes)):
            self.hidden_input_shapes.append((self.batch_size, self.hidden_shapes[i][0]))
            

        self.__theano_build__()

    #Equations created by Michael Jones, Senior Mathematics - IUPUI           
    def fails_test(self, x, n, t, p):
        if ((x - n * ((2 ** t) - 1) - 1) / 2 ** (t-1)) % 4 == 2:
            return False
        elif (x - n * ((2 ** t) - 1) - 1) == 0:
            return False
        else:
            return True
    
    def display(self, index):
        img = self.x_train[index].eval()
        dim = int(math.sqrt(img.shape[0]))
        c = self.get_container(index)
        new_im = self.f_pred(c)
        if self.color == 3:
            im = new_im.reshape((dim,dim,3)) * 255
            plot = plt.imshow(im)
        else:
            im = new_im.reshape((dim,dim)) * 255
            plot = plt.imshow(im, cmap = 'gray')
        
        
        plt.show()  


    def __theano_build__(self):

        ##VARIABLE DEFINITIONS##
        if self.color == 3:
            x = T.dtensor3('x')
            y = T.dtensor3('y')
            noise = T.dtensor3('noise')
        else:
            x = T.matrix('x')
            y = T.matrix('y')
            noise = T.matrix('noise')
        ind = T.lscalar()
        learnRate = T.scalar('learnRate')
        t = T.scalar('t')
        i = 1

        L0_in = x.reshape((self.batch_size, self.color, self.img_dim, self.img_dim))
        y_o = y.reshape((self.batch_size, self.color, self.img_dim, self.img_dim))
        
        ##GENERATING LAYERS##

        #Convolution
        print("Loading convolution layers...")
        self.conv_layers = [Convolution_Layer(layer_in = L0_in,
                                              filter_shape = self.filter_shapes[0],
                                              input_shape = self.input_shapes[0],
                                              pool_shape = self.pool_shape,
                                              norm = True)]
        
        for i in range(1, self.num_conv_layers):
            self.conv_layers.append(Convolution_Layer(layer_in = self.conv_layers[i-1].out,
                                                      filter_shape = self.filter_shapes[i],
                                                      input_shape = self.input_shapes[i],
                                                      pool_shape = self.pool_shape))

        #Hidden
        self.hidden_in = self.conv_layers[-1].out.flatten(2)
        print("Loading hidden layers...")
        self.hidden_layers = [Hidden_Layer(self.hidden_in, self.hidden_shapes[0], self.hidden_input_shapes[0])]
        for i in range(1, len(self.hidden_shapes)):
            self.hidden_layers.append(Hidden_Layer(self.hidden_layers[-1].out, self.hidden_shapes[i], self.hidden_input_shapes[i]))


        #Deconvolution
        deconv_in = self.hidden_layers[-1].out.reshape(self.deconv_input_shapes[0])
        print("Loading deconvolution layers...")
        print("-Loading layer 1")
        self.deconv_layers = [Deconvolution_Layer(layer_in = deconv_in,
                                                  filter_shape = self.deconv_filter_shapes[0],
                                                  input_shape = self.deconv_input_shapes[0],
                                                  output_shape = self.deconv_output_shapes[0],
                                                  pool_shape = self.pool_shape)]

        for i in range(1, self.num_conv_layers):
            print("-Loading layer {}".format(i+1))
            self.deconv_layers.append(Deconvolution_Layer(layer_in = self.deconv_layers[-1].out,
                                                          filter_shape = self.deconv_filter_shapes[i],
                                                          input_shape = self.deconv_input_shapes[i],
                                                          output_shape = self.deconv_output_shapes[i],
                                                          pool_shape = self.pool_shape))


        ##OUTPUTS##
        output = self.deconv_layers[-1].out
        pred = output.reshape((self.batch_size,self.img_dim, self.img_dim,self.color))
        loss = T.mean((y_o - output) ** 2)
        error = T.mean(((y_o * 255) - (output*255)) ** 2)
        
        print("Loading f_pred...")
        self.f_pred = theano.function([x], pred[0]) #Just return the first. We'll passing x as the same image batched n times

        #Debug functions
        print("Loading debug functions...")
        self.debug_conv_out = theano.function([x], self.conv_layers[-1].out)            #Output of all convolution layers
        self.debug_deconv_lin = theano.function([x], self.deconv_layers[0].layer_in)    #Input of the first deconvolution layer (after undoing max pooling)
        self.debug_hidden_in = theano.function([x], self.hidden_layers[0].layer_in)     #Input of the first hidden layer
        self.debug_output = theano.function([x], output)                                #Output after deconvolution
        self.debug_y_o = theano.function([y], y_o)                                      #The expected output

        #Getting what we really want from this autoencoder: the condensed features from the middle hidden layer
        median = (len(self.hidden_shapes) // 2) - 1 #We'll always have an even number of hidden layers, I use // so that it auto casts to int for indexing
        self.features = theano.function([x], self.hidden_layers[median].out)
        

        ##GATHERING THE WEIGHTS##
        """
        I used generator functions to create lists of the weights so that the code is easier to read.
        It also has the added benefit of making the code shorter...
        """
        c_params = self.conv_layers[0].param
        c_params_m = self.conv_layers[0].param_m
        c_params_v = self.conv_layers[0].param_v
        for i in range(1, len(self.conv_layers)):
            c_params += self.conv_layers[i].param
            c_params_m += self.conv_layers[i].param_m
            c_params_v += self.conv_layers[i].param_v

        h_params = self.hidden_layers[0].param
        h_params_m = self.hidden_layers[0].param_m
        h_params_v = self.hidden_layers[0].param_v
        for i in range(1, len(self.hidden_layers)):
            h_params += self.hidden_layers[i].param
            h_params_m += self.hidden_layers[i].param_m
            h_params_v += self.hidden_layers[i].param_v

        dc_params = self.deconv_layers[0].param
        dc_params_m = self.deconv_layers[0].param_m
        dc_params_v = self.deconv_layers[0].param_v
        for i in range(1, len(self.conv_layers)):
            dc_params += self.deconv_layers[i].param
            dc_params_m += self.deconv_layers[i].param_m
            dc_params_v += self.deconv_layers[i].param_v

        params = h_params + dc_params + c_params
        params_m = h_params_m + dc_params_m + c_params_m
        params_v = h_params_v + dc_params_v + c_params_v

        ##GRADIENTS AND PARAMETER UPDATES##
        """
        I use adam as my learning algorithm.
        Once again I use generator functions to create the parameter updates.
        """
        print("Loading gradients...")
        grads = T.grad(loss, params)

        beta1 = .9
        beta2 = .999
        eps = 1e-8

        print("Loading updates...")
        update_m = [(m, beta1 * m + (1 - beta1) * grad)
                        for m, grad in zip(params_m, grads)]
        update_v = [(v, beta2 * v + (1 - beta2) * grad ** 2)
                        for v, grad in zip(params_v, grads)]
        update = [(param, param - learnRate * (m_i / (1-(beta1 ** t))) / (T.sqrt((v_i / (1-(beta2 ** t)))) + eps))
                        for param, m_i, v_i in zip(params, params_m, params_v)]

        updates = update_m + update_v + update #List addition is the same as concatenation

        print("Loading val_error...")
        self.val_err = theano.function([ind, noise],
                                       error,
                                       givens={x: self.x_val[ind * self.batch_size: (ind+ 1) * self.batch_size] + noise,
                                               y: self.x_val[ind * self.batch_size: (ind+ 1) * self.batch_size]})
        print("Loading adam_step...")
        self.adam_step = theano.function([ind, learnRate, t],
                                         error,
                                         updates = updates,
                                         givens={x: self.x_train[ind * self.batch_size: (ind+ 1) * self.batch_size],
                                                 y: self.x_train[ind * self.batch_size: (ind+ 1) * self.batch_size]})

    def toggle_BN_train(self):
        for i in range(len(self.conv_layers)):
            self.conv_layers[i].BN.train = self.conv_layers[i].BN.train * -1 + 1
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].BN.train = self.hidden_layers[i].BN.train * -1 + 1
        for i in range(len(self.deconv_layers)):
            self.deconv_layers[i].BN.train = self.deconv_layers[i].BN.train * -1 + 1

    def get_BN_train(self):
        return self.conv_layers[0].BN.train

    def train_model(self, epochs = 100, stop_at = 2500):
        print("Training model...\n")
        batch_steps = int(self.x_train.eval().shape[0] / self.batch_size) #Casting to an int automatically rounds down and cuts off any training sequences that wouldn't create a full batch step
##        val_steps = int(self.x_val.eval().shape[0] / self.batch_size) // 5

        for t in range(1,epochs+1):
            train_error = 0
            val_error = 0
            noise =  np.random.normal(loc = 0.0, scale = .1, size = (self.batch_size, (self.img_dim**2), self.color))
            if self.get_BN_train() != 1: self.toggle_BN_train()
            for b in range(batch_steps):
                train_error += self.adam_step(b, self.learn_rate, t)
                print(b, train_error/(b+1))
            if self.get_BN_train() != 0: self.toggle_BN_train()
##            for b in range(val_steps):
##                val_error += self.val_err(b, noise)
            train_error /= batch_steps
##            val_error /= val_steps
            print("Epoch {}".format(t))
            print("Train Error: {0:4f}".format(train_error))
##            print("Validation Error: {0:4f}\n".format(val_error))
            if train_error < stop_at:
                if self.get_BN_train() != 0: self.toggle_BN_train()
                print("SUCCESS")
                write_panda_csv('features_test.csv')
                write_features_csv()
                return

        if self.get_BN_train() != 0: self.toggle_BN_train()

    def predict(self, container):
        return self.features(container)
    
    def get_container(self, index, val = False):
        if not val:
            img = self.x_train[index].eval()
        else:
            img = self.x_val[index].eval()
        if self.color == 3:
            container = np.zeros((self.batch_size, img.shape[0], self.color))
        else:
            container = np.zeros((self.batch_size, img.shape[0]))
        for i in range(self.batch_size):
            container[i] = img
        return container

def load_imgs(file, size_percent = 1, limit_percent = 1, delimiter = "|||", color = True, val_per = .25):
    
    if color:
        color_s = 3
    else:
        color_s = 1
        
    with open(file, 'r') as f:
        lines = f.readlines()
        
    imgs = []
    y_ = []
    for i in range(int(len(lines) * limit_percent)):
        d = lines[i].split(delimiter)
        imgs.append(d[0])
        y_.append(d[1].strip())
        
    x_ = []
    for i in imgs:
        im = cv2.imread(i)
        im = cv2.resize(im, (0,0), fx = size_percent, fy = size_percent)
        x = im.reshape(im.shape[0] ** 2, color_s)
        x_.append(x)

    val_start = int(len(x_) * (1-val_per))
    
    x_train = theano.shared(np.asarray(x_[0:val_start]).astype(theano.config.floatX))
    y_train = theano.shared(np.asarray(y_[0:val_start]).astype(theano.config.floatX))

    x_val = theano.shared(np.asarray(x_[val_start:]).astype(theano.config.floatX))
    y_val = theano.shared(np.asarray(y_[val_start:]).astype(theano.config.floatX))

    r = [(x_train, y_train), (x_val, y_val)]
    val_size = int(len(imgs) * val_per)
    print("Data loaded\nTrain size: {}\nVal size: {}".format(val_start, val_size))
    return r, imgs

    
def load_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def split_set(d):
        x, y = d
        x_ = theano.shared(np.asarray(x).astype(theano.config.floatX))
        y_ = theano.shared(np.asarray(y).astype(T.int32))

        return x_, y_

    x_train, y_train = split_set(train_set)
    x_val, y_val = split_set(valid_set)

    r = [(x_train, y_train), (x_val, y_val)]
    return r



def write_panda_csv(file):
    print("Writing to csv")
    features = []
    val_size = e.x_val.shape[0].eval()
    v_steps = int(val_size / 200)

    for i in range(v_steps):
        container = e.x_val[i * 200:(i+1) * 200].eval()
        c = e.features(container)
        for item in c:
            features.append(item)

    for i in range(v_steps*200, val_size):
        c = e.get_container(i, val = True)
        features.append(e.predict(c)[0])
    df = pd.DataFrame(features, index = imgs)
    print(df)
    df.to_csv(file)

def write_features_csv():
    print("Writing to csv")
    r, img = load_imgs('data.txt', limit_percent = 1, size_percent = .261, val_per = 0)
    e.x_val = r[0][0] / 255
    features = []
    val_size = e.x_val.shape[0].eval()
    v_steps = int(val_size / 200)

    for i in range(v_steps):
        container = e.x_val[i * 200:(i+1) * 200].eval()
        c = e.features(container)
        for item in c:
            features.append(item)

    for i in range(v_steps*200, val_size):
        c = e.get_container(i, val = True)
        features.append(e.predict(c)[0])
    df = pd.DataFrame(features, index = img)
    print(df)
    df.to_csv('features_train.csv')

if __name__ == "__main__":
##    r = load_mnist()
##    e = Encoder(r, img_dim = 28, filter_size = 5, filters = [50, 25], batch_size = 1000, color = False)
    r, imgs = load_imgs('full.txt', limit_percent = 1, size_percent = .261, val_per = .001)
    imgs = imgs[0]
    e = Encoder(r, img_dim = int(224*.261), filter_size = 15, filters = [32, 64], batch_size = 200, num_features = 30)
    t, imgs = load_imgs('test.txt', limit_percent = 1, size_percent = .261, val_per = 0)
    e.x_val = t[0][0] / 255
    if e.get_BN_train() != 0: e.toggle_BN_train()

























        
