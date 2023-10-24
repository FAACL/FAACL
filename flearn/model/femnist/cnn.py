import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))

    # Conv1 Layer
    model.add(Conv2D(32, 5, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    
    # Conv2 Layer
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    
    
    # Dense Layer
    model.add(Dense(2048, 'relu'))
    
    # Output Layer
    model.add(Dense(62, 'softmax'))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr=0.003):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)