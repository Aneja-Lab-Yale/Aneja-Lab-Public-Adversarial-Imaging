# Adversarial Learning
# Trains and saves model for desired dataset (mnist, cifar, lidc, brain mri, or cbis-ddsm)
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/16/20)
# Updated (11/28/20)

# import dependencies
import tensorflow as tf
from models.models import brain_vgg_model, mnist_vgg_model, cifar_vgg_model, lidc_vgg_model, ddsm_vgg_model
from data_loader import load_dataset
from utils import get_acc_preds, plot_tv
from configs import path
from tensorflow.keras.callbacks import EarlyStopping

# query user for dataset
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')

# load dataset
x_train, y_train, x_test, y_test = load_dataset(dataset, path)

# train model
if dataset == 'mnist':
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    input_shape = (32, 32, 3)
    model = mnist_vgg_model(input_shape)
    history = model.fit(x_train, y_train, batch_size=64, epochs=20,  callbacks=[es], validation_data=(x_test, y_test), verbose=True)


elif dataset == 'cifar':
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    input_shape = (32, 32, 3)
    model = cifar_vgg_model(input_shape)
    history = model.fit(x_train, y_train, batch_size=128, epochs=60, callbacks=[es], validation_data=(x_test, y_test), verbose=True)


elif dataset == 'brain_mri':
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    input_shape = [224, 224, 3]
    model = brain_vgg_model(input_shape)
    history = model.fit(x_train, y_train, batch_size=50, epochs=100, callbacks=[es], validation_data=(x_test, y_test), verbose=True)


elif dataset == 'ddsm':
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    input_shape = [116, 116, 3]
    model = ddsm_vgg_model(input_shape)
    history = model.fit(x_train, y_train, batch_size=50, epochs=100, callbacks=[es], validation_data=(x_test, y_test), verbose=True)


elif dataset == 'lidc':
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
    input_shape = [224, 224, 3]
    model = lidc_vgg_model(input_shape)
    history = model.fit(x_train, y_train, epochs=200, callbacks=[es], validation_data=(x_test, y_test), verbose=True)

# save model
model.save(path + 'models/' + dataset + '/' + dataset + '_vgg16_model.h5')

# Evaluate classifier on train set
acc = get_acc_preds(model, x_train, y_train)
print("\nAccuracy of original classifier on training set: %.2f%%" % (acc * 100))

# Evaluate the classifier on the test set
acc = get_acc_preds(model, x_test, y_test)
print("\nAccuracy of original classifier on test set: %.2f%%" % (acc * 100))

# Plot training and validation curves of classifier model
plot_tv(history, path + 'models/' + dataset + '/', 'VGG16', 'tv_curve', 'Accuracy')
