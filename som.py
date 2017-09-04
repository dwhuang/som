#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import warnings

np.seterr(all='warn')
warnings.simplefilter('error')
warnings.filterwarnings('ignore', 'underflow')

class Som:
    '''Self-organizing map implementation
    '''
    def __init__(self, grid, input_dim, **kwargs):
        '''Construct a SOM

           grid: a grid object representing the map layer
           input_dim: number of input dimensions
           kwargs: other parameters such as learning rate function parameters
        '''
        # hyper parameters
        self.grid = grid
        self.num_nodes = grid.size
        self.input_dim = input_dim
        self.nb_init = kwargs.get('nb_init', .9)
        self.nb_infl = kwargs.get('nb_infl', .4)
        self.nb_sigma = kwargs.get('nb_sigma', .01)
        self.lr_init = kwargs.get('lr_init', .4)
        self.lr_infl = kwargs.get('lr_infl', .4)
        self.lr_sigma = kwargs.get('lr_sigma', .001)
        self.input_ranges = np.array(kwargs.get(
            'input_ranges',
            [[0] * input_dim, [1] * input_dim],
        ), dtype=float)
        if self.input_ranges.shape[0] != 2 or self.input_ranges.shape[1] != input_dim:
            print(
                "Input argument input_ranges has invalid shape",
                self.input_ranges.shape,
            )
            raise RuntimeError()
        # states
        self.weights = self.__init_weights()
        self.activity = np.zeros(shape=(self.num_nodes,))
        self.lr = np.zeros(shape=(self.num_nodes,))
        self.nb = np.zeros(shape=(self.num_nodes,))
        self.winner = -1
        self.delta_weights = np.zeros(shape=self.weights.shape)


    def __init_weights(self, seed=5566):
        '''Random weight initialization
        '''
        np.random.seed(seed)
        return np.random.rand(self.num_nodes, self.input_dim) * (
            self.input_ranges[1] - self.input_ranges[0]
        ) + self.input_ranges[0]


    def run(self, input_vec):
        '''Run the SOM once, taking an input vector and producing an activity
           pattern as well as a winner
        '''
        in_ind = ~np.isnan(input_vec)  # input indinces that are not NaN
        np.sum(
            (self.weights[:, in_ind] - input_vec[in_ind]) ** 2,
            axis=1,
            out=self.activity,
        )
        self.winner = self.activity.argmin()
        return self.winner


    def __compute_nb(self, trn_progress):
        '''Produce a neighborhood function around the winner for training.
           The value for the winner locaiton is one. The value decreases
           exponentially as one moves away from the winner location.

           trn_progress: training progress, between 0 and 1. As training
                         progresses, the function becomes more and more
                         focused on the winner.
        '''
        gamma = self.__nonlinear_decr_func(trn_progress,
                                           self.nb_init,
                                           0,
                                           self.nb_infl,
                                           self.nb_sigma)
        # gamma must be < 1 for this to work
        np.float_power(gamma, self.grid.get_dist_map(self.winner), out=self.nb)
        self.nb[self.nb < 1e-6] = 0
        return self.nb


    def __nonlinear_decr_func(self, trn_progress, init, fin, infl, sigma):
        try:
            ret = fin + (init - fin) / (1 + np.exp((trn_progress - infl) / sigma))
        except RuntimeWarning as e:
            ret = fin
        return ret


    def learn(self, input_vec, trn_progress):
        '''Adapt the weights once based on the input vector and the current
           activity and winner
        '''
        self.run(input_vec)
        lr = self.__nonlinear_decr_func(trn_progress,
                                        self.lr_init,
                                        0,
                                        self.lr_infl,
                                        self.lr_sigma)
        # input - weights
        self.delta_weights.fill(0)
        in_ind = ~np.isnan(input_vec)  # input indices where values are not NaN
        np.subtract(
            input_vec,
            self.weights,
            out=self.delta_weights,
            where=in_ind,
        )
        # lr * nb
        self.lr.fill(lr)
        self.__compute_nb(trn_progress)
        np.multiply(self.lr, self.nb,
                    out=self.lr)
        # delta = lr * nb * (input - weights)
        np.multiply(self.delta_weights, self.lr[:, None],
                    out=self.delta_weights)
        # w = w + delta
        np.add(self.weights, self.delta_weights,
               out=self.weights)


    def train(self, inputs, num_epochs):
        '''Train the SOM

           inputs: row vectors of input vectors
           num_epochs: number of epochs
        '''
        num_inputs = inputs.shape[0]
        for epoch in range(num_epochs):
            if epoch % 100 == 0:
                print('epoch', epoch)
            trn_progress = epoch / (num_epochs - 1) if num_epochs > 1 else 1
            permu = np.random.permutation(num_inputs)
            for i in range(num_inputs):
                input_vec = inputs[permu[i]]
                self.learn(input_vec, trn_progress)


    def get_average_weight_jumps(self):
        '''For each node, calculate the average of weight differences between 
           the node and all its direct neighbor nodes. Returns an nx1 matrix.
        '''
        ret = np.zeros((self.num_nodes, 1))
        for i in range(self.num_nodes):
            nb_inds = self.grid.get_neighbors(i)
            nb_wts = self.weights[nb_inds, :]
            my_wts = self.weights[i, :]
            ret[i, 0] = np.average(((nb_wts - my_wts) ** 2).sum(axis=1) ** .5)
        return ret


    def smoothness(self):
        '''Smoothness metric: average weight changes from a node to all its
           neighbors, averaged over all nodes
        '''
        return np.average(self.get_average_weight_jumps())


    def umatrix(self, max_possible_jump=None):
        '''Returns U-matrix representation of the SOM. Each node contains a
           value between 0 and 1, where large values indicate small weight
           jumps (valleys) and small values indicate large weight jumps (hills).
           Returns a list of size n.
        '''
        ret = self.get_average_weight_jumps()
        if max_possible_jump is None:
            max_possible_jump = np.sum(
                (self.input_ranges[1, :] - self.input_ranges[0, :]) ** 2
            ) ** .5
        ret /= max_possible_jump
        ret = 1 - ret
        ret = ret.squeeze().tolist()
        return ret


    def error(self, inputs, print_details=False):
        '''Average error: error between each input vector and the winning node's
           weights, averaged over all input vectors
        '''
        err = 0
        for input_vec in inputs:
            winner = self.run(input_vec)
            in_ind = ~np.isnan(input_vec)
            dist = np.sum((self.weights[winner, in_ind] - input_vec[in_ind])
                    ** 2) ** .5
            err += dist
            if print_details:
                print(self.weights[winner, :], input_vec, dist)
        return err / len(inputs)


    def label(self, labels, inputs):
        '''Returns an array, each element represents a node. Each element is
           an array of string, which are the labels of the inputs that result
           in the current node being selected as the winner. Nodes not selected
           as a winner for any inputs contain empty arrays.
        '''
        ret = [[] for _ in range(self.num_nodes)]
        for label, input_vec in zip(labels, inputs):
            winner = self.run(input_vec)
            ret[winner].append(label)
        return ret
