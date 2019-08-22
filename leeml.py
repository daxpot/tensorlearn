import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from keras import activations, losses, optimizers, metrics

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='/Users/zengkui/Desktop/tensor/tensorlearn/data/mnist.npz')
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train/255
    x_test = x_test/255
    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test)=load_data()

    model = Sequential()
    model.add(Dense(input_dim = 28*28, units = 500, activation = activations.elu))
    model.add(Dropout(0.7))
    model.add(Dense(units = 500, activation = activations.elu))
    model.add(Dropout(0.7))
    model.add(Dense(units = 500, activation = activations.relu))
    model.add(Dropout(0.7))
    model.add(Dense(units = 10, activation = activations.softmax))

    model.compile(loss = losses.binary_crossentropy, optimizer = optimizers.SGD(lr = 0.1), metrics = [metrics.binary_accuracy])

    model.fit(x_train, y_train, batch_size = 100, epochs = 10)

    result = model.evaluate(x_test, y_test)

    print('TEST ACC:', result[1])

if __name__ == '__main__':
    main()