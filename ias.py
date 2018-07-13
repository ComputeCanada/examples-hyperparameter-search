'''
Compute Canada incremental architecture search example.

Adapted from the keras mnist example.
'''

from itertools import product
import os
import os.path as osp
from time import sleep

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from mpi4py import MPI
import numpy as np

# generic mnist stuff

BATCH_SIZE = 128
N_CLASSES = 10
N_EPOCHS = 10

assert K.image_data_format() == 'channels_first'

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
    x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def mk_conv(k1, k2, d1, d2):

    model = Sequential()

    model.add(Conv2D(k1, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(k2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(d1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(d2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def train(model, xt, yt, xv, yv):
    model.fit(
        xt, yt,
        batch_size=BATCH_SIZE, epochs=N_EPOCHS,
        verbose=0,
        validation_data=(xv, yv)
    )


def evaluate(model, xv, yv):
    return model.evaluate(xv, yv, verbose=0)

# set up MPI infrastructure

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_WORKERS = COMM.Get_size()
ROOT_RANK = 0
IS_ROOT = RANK == ROOT

# number of samples in each generation
GENERATION_WIDTH = 20


def sample_params(k1=None, k2=None, d1=None, d2=None):
    '''
    Samples random parameters, with the possibility of keeping certain ones
    fixed.
    '''

    k1 = k1 or np.randint(16, 32)
    k2 = k2 or np.randint(16, 64)
    d1 = d1 or np.randint(32, 128)
    d2 = d2 or np.randint(32, 64)

# to simplify our logic and to maximize gpu usage, insist that the number of
# workers divides the generation width
assert GENERATION_WIDTH % N_WORKERS == 0
# futhermore, assert we can use all workers
assert GENERATION_WIDTH >= N_WORKERS

SCOREFILE = osp.expanduser('~/mnist_scores.csv')


def search_ias():
    '''
    Searches for the optimal parameters using IAS.

    The selection function works something akin to gibbs sampling.
    '''

    (xt, yt), (xv, yv) = load_data()

    # order in which to search for params, parameters can repeat
    param_search_order = ['k1', 'k2', 'd1', 'd2']
    # for instance, one can use
    # param_search_order = ['k1', 'k2', 'd1', 'd2', 'd1', 'k2', 'k1']
    best_params = {}

    for gen_ix, param_name in enumerate(param_search_order):

        fixed_params = best_params.copy()
        # if doing multiple passes, make sure to pop the one being sampled
        fixed_params.pop(param_name, None)

        # sample the parameters based on the current best
        if IS_ROOT:
            all_new_params = [
                sample_params(**fixed_params)
                for i in range(GENERATION_WIDTH)
            ]
        else:
            all_new_params = None

        local_params = COMM.scatter(all_new_params, root=ROOT_RANK)
        local_scores = []

        for param in local_params:

            print(f'worker {RANK}: training model {params}')
            model = mk_conv(*params)
            train(model, xt, yt, xv, yv)
            _, score = evaluate(model, xv, yv)

            with open(SCOREFILE, 'a') as f:
                f.write(f'{",".join(params + (score,))}\n')

            local_scores.append(score)

        all_scores = sum(flatten(COMM.gather(loacal_scores)), [])

        # find the best param of the generation
        if IS_ROOT:
            best_ix = np.argmax(all_scores)
            best_param = all_new_params[best_ix]
            print(f'best param for {param_name} is {best_param}')
            best_params[param_name] = best_param

    return best_params


def main():
    print('starting search')
    best_params = search_ias()
    print(f'search done, best params are {best_params}')


if __name__ == '__main__':
    main()
