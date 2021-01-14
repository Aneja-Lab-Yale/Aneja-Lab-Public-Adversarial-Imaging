# Adversarial Learning
# Evaluate the adversarial susceptibility of each image
# to eliminate the most at-risk images, and measure model
# performance on remaining dataset
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (11/9/20)
# Updated (MM/DD/YY)


# import dependencies
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from tensorflow.keras.models import load_model
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, ProjectedGradientDescent
from data_loader import load_dataset
from attacks import create_attack
from models.models import medical_vgg_model, cifar_vgg_model, mnist_vgg_model
from configs import path_fig, epsilon, eps_step, max_iter, num_random_init, path
from utils import get_acc_preds
import numpy as np


# query user for dataset
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')

# load in dataset using load_dataset function
x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=False)

# verify that x_train, y_train, x_test, y_test have the correct dimensions
print('x_train shape: ', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

# load model
model = load_model(path + 'models/' + dataset + '/' + dataset + '_vgg16_model.h5')


# load input shape
if dataset == 'brain_mri':
    input_shape = [224, 224, 3]
elif dataset == 'mnist':
    input_shape = [32, 32, 3]
elif dataset == 'ddsm':
    input_shape = [116, 116, 3]
elif dataset == 'cifar':
    input_shape = [32, 32, 3]
elif dataset == 'lidc':
    input_shape = (224, 224, 3)

# convert model to KerasClassifier
classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)

# evaluate original classifier accuracy on training set
acc = get_acc_preds(classifier, x_train, y_train)
print("\nAccuracy of original classifier on training set: %.2f%%" % (acc * 100))

# Evaluate the classifier on the test set
acc = get_acc_preds(classifier, x_test, y_test)
print("\nAccuracy of original classifier on test set: %.2f%%" % (acc * 100))


# define threshold epsilon values to separate datasets into 6 subsets (ranked by adversarial sensitivity)
# use fgsm attack
if dataset == 'mnist':
    eps_range = [0.0001, 0.0005, 0.01, 0.015, 0.02, 0.025, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
elif dataset == 'cifar':
    eps_range = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1]
elif dataset == 'brain_mri':
    eps_range = [0.0001, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0013, 0.0016, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.00325, 0.0035,
                 0.00375, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012]
elif dataset == 'ddsm':
    eps_range = [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025,
                 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007, 0.008]
elif dataset == 'lidc':
    eps_range = [0.0001, 0.0003, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0023, 0.0026, 0.0028, 0.003,
                 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
                 0.019, 0.02]



# evaluate sensitivity scores of each image
test_eps_scores = [1] * x_test.shape[0]

for eps in eps_range:
    attacker = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps / 4, max_iter=max_iter,
                                        num_random_init=num_random_init)
    x_test_adv = attacker.generate(x_test)
    for i in range(x_test.shape[0]):
        img = np.expand_dims(x_test[i], axis=0)
        adv_img = np.expand_dims(x_test_adv[i], axis=0)
        pred = np.argmax(classifier.predict(img))
        pred_adv = np.argmax(classifier.predict(adv_img))
        if test_eps_scores[i] == 1:
            if pred != pred_adv:
                test_eps_scores[i] = eps
np.save(path + dataset + '/test_eps_scores.npy', test_eps_scores)


test_eps_scores = np.load(path + dataset + '/test_eps_scores.npy')

test_eps_freq = [0] * x_test.shape[0]
for eps_score in test_eps_scores:
    for i in range(len(eps_range)):
        if eps_score == eps_range[i]:
            test_eps_freq[i] = test_eps_freq[i] + 1

for i in range(len(eps_range)):
    print(eps_range[i], ": ", test_eps_freq[i])


if dataset == 'lidc':
    eps_cap = 0.0007
elif dataset == 'ddsm':
    eps_cap = 0.00025
elif dataset == 'brain_mri':
    eps_cap = 0.0006
elif dataset == 'cifar':
    eps_cap = 0.0025
elif dataset == 'mnist':
    eps_cap = 0.05
x_test_new = []
y_test_new = []

# remove most susceptible images in test set (20% of test set)
for i in range(x_test.shape[0]):
    img = x_test[i]
    if test_eps_scores[i] > eps_cap:
        x_test_new.append(img)
        y_test_new.append(y_test[i])
x_test_new = np.array(x_test_new)
y_test_new = np.array(y_test_new)

# Evaluate the classifier on the reduced test set
acc = get_acc_preds(classifier, x_test_new, y_test_new)
print("\nAccuracy of original classifier on reduced test set: %.2f%%" % (acc * 100))
print('x_test_new.shape', x_test_new.shape)
print('y_test_new.shape', y_test_new.shape)
