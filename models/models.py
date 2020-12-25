# Adversarial Learning
# Defines VGG models for classification
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (9/01/20)
# Updated (12/22/20)


# import dependencies
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


# VGG16 model for all medical classification tasks
# Employed (9/01/20)
# Debugged (MM/DD/YY)
# Dreated by Marina Joel
def medical_vgg_model(input_shape):
    # pre-trained VGG16 model
    vgg_base = VGG16(
        include_top=False,
        #weights=None,
        weights='imagenet',
        input_tensor=Input(shape=input_shape),
        input_shape=input_shape,
        pooling=None
        )

    # ouput
    x = vgg_base.output

    # Dropout Rate Hyperparameters
    dropout1 = tf.keras.layers.Dropout(0.5)
    dropout2 = tf.keras.layers.Dropout(0.5)

    # Model layers
    x = dropout1(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1',kernel_constraint=max_norm(2), trainable = True
)(x)
    x = dropout2(x)
    x = Dense(1024, activation='relu', name='fc2', kernel_constraint=max_norm(2), trainable = True
)(x)
    predictions = Dense(2, activation='softmax', name='predictions',trainable = True)(x)

    # Model inputs and outputs
    model = Model(inputs=vgg_base.input, outputs=predictions)

    # optimizer hyperparameters
    learning_rate =0.0002
    lr_decay = 1e-6
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

    # Model compile
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


# VGG16 model for mnist dataset
# Employed (9/01/20)
# Debugged (MM/DD/YY)
# Dreated by Marina Joel
def mnist_vgg_model(input_shape):
    vgg_base = VGG16(
        include_top=False,
        #weights=None,
        weights='imagenet',
        input_tensor=Input(shape=input_shape),
        input_shape=input_shape,
        pooling=None
        )
    x = vgg_base.output
    dropout1 = tf.keras.layers.Dropout(0.5)
    dropout2 = tf.keras.layers.Dropout(0.5)
    x = dropout1(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1',kernel_constraint=max_norm(2), trainable = True
)(x)
    x = dropout2(x)
    x = Dense(1024, activation='relu', name='fc2', kernel_constraint=max_norm(2), trainable = True
)(x)
    predictions = Dense(10, activation='softmax', name='predictions',trainable = True)(x)
    model = Model(inputs=vgg_base.input, outputs=predictions)
    opt = SGD(lr = 0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# VGG16 model for cifar dataset
# Employed (9/01/20)
# Debugged (MM/DD/YY)
# Dreated by Marina Joel
def cifar_vgg_model(input_shape):
    vgg_base = VGG16(
        include_top=False,
        # weights=None,
        weights='imagenet',
        input_tensor=Input(shape=input_shape),
        input_shape=input_shape,
        pooling=None
    )
    x = vgg_base.output
    dropout1 = tf.keras.layers.Dropout(0.5)
    dropout2 = tf.keras.layers.Dropout(0.5)
    x = dropout1(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_constraint=max_norm(2), trainable=True
              )(x)
    x = dropout2(x)
    x = Dense(1024, activation='relu', name='fc2', kernel_constraint=max_norm(2), trainable=True
              )(x)
    predictions = Dense(10, activation='softmax', name='predictions', trainable=True)(x)
    model = Model(inputs=vgg_base.input, outputs=predictions)
    learning_rate = 0.001
    lr_decay = 1e-6
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model
