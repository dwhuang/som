#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .som import Som
from .hexgrid import HexGrid

def gen_random_data(fpath, seed=7788):
    '''Generate random 3D data for testing
    '''
    np.random.seed(seed)
    d = np.random.rand(100, 3)
    with open(fpath, 'w') as fp:
        for i in range(100):
            print("d%03d\t%f\t%f\t%f" % (i, d[i, 0], d[i, 1], d[i, 2]), file=fp)


def read_data(fpath, delimiter='\t'):
    '''Read input data from a file
    '''
    with open(fpath, 'r') as fp:
        lines = fp.readlines()
        input_dim = 0
        data = []
        labels = []
        for i, line in enumerate(lines):
            toks = line.split(delimiter)
            tok_count = len(toks)
            if tok_count <= 1:
                raise RuntimeError("Not enough tokens in line:", i)
            if input_dim == 0:
                input_dim = tok_count - 1
            elif tok_count - 1 != input_dim:
                raise RuntimeError("Input dimensions do not agree:",
                                   tok_count - 1,
                                   input_dim)
            labels.append(toks[0])
            data.append([np.float(x) for x in toks[1:]])
    data = np.array(data)
    print('input dimension =', data.shape[1])
    return labels, data


def parameter_sweep(som_size, data_fname, log_fname,
        nb_init_vals, nb_infl_vals, nb_sigma_vals, 
        lr_init_vals, lr_infl_vals, lr_sigma_vals,
        **kwargs
    ):
    '''Sweep to find a good combination of parameters.
    '''
    labels, inputs = read_data(data_fname)
    with open(log_fname, 'w') as fp:
        for nb_init in nb_init_vals:
            for nb_infl in nb_infl_vals:
                for nb_sigma in nb_sigma_vals:
                    for lr_init in lr_init_vals:
                        for lr_infl in lr_infl_vals:
                            for lr_sigma in lr_sigma_vals:
                                print(
                                    "{},{},{},{},{},{}".format(
                                        nb_init, nb_infl, nb_sigma,
                                        lr_init, lr_infl, lr_sigma
                                    )
                                )
                                som = Som(grid=HexGrid(som_size),
                                    input_dim=inputs.shape[1],
                                    nb_init=nb_init,
                                    nb_infl=nb_infl,
                                    nb_sigma=nb_sigma,
                                    lr_init=lr_init,
                                    lr_infl=lr_infl,
                                    lr_sigma=lr_sigma,
                                    **kwargs
                                )
                                som.train(inputs, 1000)
                                fp.write(
                                    "{},{},{},{},{},{},{:.6},{:.6}\n".format(
                                        nb_init, nb_infl, nb_sigma,
                                        lr_init, lr_infl, lr_sigma,
                                        som.smoothness(), som.error(inputs)
                                    )
                                )
                                fp.flush()
