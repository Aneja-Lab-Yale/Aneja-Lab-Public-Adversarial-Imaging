# Adversarial attacks
# Apply adversarial attacks on 5 datasets (mnist, cifar, lidc, brain mri, ddsm)
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/16/20)
# Updated (12/22/20)

# import dependencies
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from tensorflow.keras.models import load_model
from data_loader import load_dataset
from attacks import create_attack
from configs import path_fig, path
from utils import get_acc_preds, plot_attacks_acc
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# query user for dataset name
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')

# validate that dataset type is supported
if (dataset != 'lidc') & (dataset != 'ddsm') & (dataset != 'brain_mri') & (dataset != 'mnist') & (dataset != 'cifar'):
    print('dataset name invalid\n')
    exit(0)

# load in dataset using load_dataset function
x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=False)

# query user for adversarial attack to use for generating adversarial test set
attack_type = input('Enter attack to be used (fgsm, pgd, bim)\n')
if (attack_type != 'fgsm') & (attack_type != 'pgd') & (attack_type != 'bim'):
    print('attack type not supported\n')
    exit(0)

# verify that x_train, y_train, x_test, y_test have the correct dimensions
print('x_train shape: ', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

# load model
# model = load_model(path + 'models/' + dataset + '/' + dataset + '_vgg16_model.h5')
model = load_model(path + 'models/' + dataset + '/' + dataset + '_vgg16_robust_classifier.h5')
# model = load_model(path + 'models/' + dataset + '/'  + 'best_robust_classifier.h5')

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
    input_shape = (116, 116, 3)

# convert model to KerasClassifier
classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)

# evaluate classifier accuracy on training set
acc = accuracy_score(y_train, (classifier.predict(x_train)>0.5))
print("Accuracy of original classifier on training set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_train, classifier.predict(x_train), average='macro')
# print("Auc of original classifier on training set: %.2f%%\n" % (auc * 100))

# Evaluate classifier accuracy on the test set
acc = accuracy_score(y_test, (classifier.predict(x_test)>0.5))
print("Accuracy of original classifier on test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, classifier.predict(x_test), average='macro')
# print("Auc of original classifier on test set: %.2f%%\n" % (auc * 100))

# Craft adversarial samples
attacker = create_attack(attack_type, classifier)
x_test_adv = attacker.generate(x_test)
x_train_adv = attacker.generate(x_train)

# Evaluate the classifier on the adversarial training examples
acc = accuracy_score(y_train, (classifier.predict(x_train_adv)>0.5))
print("Accuracy of original classifier on adversarial training set: %.2f%%, " % (acc * 100))

# Evaluate the classifier on the adversarial test examples
acc = accuracy_score(y_test, (classifier.predict(x_test_adv)>0.5))
print("Accuracy of original classifier on adversarial test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, classifier.predict(x_test_adv), average='macro')
# print("Auc of original classifier on adverarsial test set: %.2f%%\n" % (auc * 100))

# Plot classifier accuracy over attack strength for multiple specified attacks (in this case fgsm, bim, pgd)
# plot_attacks_acc(classifier, x_test, y_test, path_fig, dataset, 'vgg16_attacks_acc')
# plot_attacks_acc(classifier, x_test, y_test, path_fig, dataset, 'robust_vgg16_attacks_acc')
