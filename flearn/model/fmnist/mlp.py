import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform

def _construct_client_model(lr):
    
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Hidden Layer
    model.add(Dense(512, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # Output Layer
    model.add(Dense(10, 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    model.summary()
    return model

def construct_model(trainer_type, lr=0.05):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)