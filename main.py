'''
Compute Canada distributed hyperparamter search example.

Adapted from the keras mnist example.
'''

from itertools import product
import os
import os.path as osp

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

BATCH_SIZE = 128
N_CLASSES = 10
N_EPOCHS = 5

assert K.image_data_format() == 'channels_last'

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

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
        verbose=1,
        validation_data=(xv, yv)
    )


def evaluate(model, xv, yv):
    return model.evaluate(xv, yv, verbose=0)


PARAM_GRID = list(product(
    range(16, 68, 4),  # k1
    range(16, 68, 4),  # k2
    range(32, 136, 8),  # k3
    range(32, 136, 8),  # k4
))
SCOREFILE = osp.expanduser('./mnist_scores.csv')

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))


# performs hyperparamter search in parallel, using a slurm array job
def search_distributed():

    (xt, yt), (xv, yv) = load_data()

    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

        params = PARAM_GRID[param_ix]

        print(f'making model {params}')
        model = mk_conv(*params)

        print(f'training model {params}')
        train(model, xt, yt, xv, yv)
        _, score = evaluate(model, xv, yv)

        with open(SCOREFILE, 'a') as f:
            f.write(f'{",".join(map(str, params + (score,)))}\n')


def main():
    print(f'WORKER {this_worker} ALIVE.')
    search_distributed()
    print(f'WORKER {this_worker} DONE.')


if __name__ == '__main__':
    main()
