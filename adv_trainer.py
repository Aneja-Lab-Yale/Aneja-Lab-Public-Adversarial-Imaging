# Adversarial Training
# Apply adversarial training on medical DL models
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/28/20)
# Updated (12/22/20)


# import dependencies
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD
from tensorflow.keras.models import load_model
from data_loader import load_dataset
from attacks import create_attack
from configs import path_fig, epsilon, eps_step, path
from utils import get_acc_preds, plot_attacks_acc


# query user for dataset
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')

# load in dataset
x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=True)

# query user for adversarial attack to use for generating adversarial test set
attack_type = input('Enter attack to be used (fgsm, pgd, bim)\n')
if (attack_type != 'fgsm') & (attack_type != 'pgd') & (attack_type != 'bim') & (attack_type != 'jsma'):
    print('attack type not supported\n')
    exit(0)


# verify that x_train, y_train, x_test, y_test have the correct dimensions
print('x_train shape: ', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)


# # load model
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



############### create robust classifier ###############
robust_classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)

# evaluate original classifier accuracy on training set
acc = get_acc_preds(robust_classifier, x_train, y_train)
print("\nAccuracy of original classifier on training set: %.2f%%" % (acc * 100))

# Evaluate the classifier on the test set
acc = get_acc_preds(robust_classifier, x_test, y_test)
print("\nAccuracy of original classifier on test set: %.2f%%" % (acc * 100))


# Craft adversarial samples
attacker = create_attack(attack_type, robust_classifier)
x_test_adv = attacker.generate(x_test)
x_train_adv = attacker.generate(x_train)


# Evaluate the classifier on the adversarial test examples
acc = get_acc_preds(robust_classifier, x_test_adv, y_test)
print("\nAccuracy of original classifier on adversarial test set: %.2f%%" % (acc * 100))


# ############## train robust classifier ###############
nb_epochs=30
if dataset == 'cifar':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=epsilon, eps_step=eps_step, nb_epochs=nb_epochs)
    trainer.fit(x_train, y_train)
#
elif dataset == 'ddsm':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=epsilon, eps_step=eps_step, nb_epochs=nb_epochs)
    trainer.fit(x_train, y_train)


elif dataset == 'lidc':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=epsilon, eps_step=eps_step, nb_epochs=nb_epochs)
    trainer.fit(x_train, y_train)


elif dataset == 'mnist':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=epsilon, eps_step=eps_step, nb_epochs=nb_epochs)
    trainer.fit(x_train, y_train)

elif dataset == 'brain_mri':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=epsilon, eps_step=eps_step, nb_epochs=nb_epochs)
    trainer.fit(x_train, y_train)


############## save robust classifier ###############
robust_classifier.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')


# # # # load adversarially trained robust classifier
# model = load_model(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')
# robust_classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)


# evaluate robust classifier on original training set
acc = get_acc_preds(robust_classifier, x_train, y_train)
print("\nAccuracy of robust classifier on training set: %.2f%%" % (acc * 100))


# evaluate robust classifier on original test set
acc = get_acc_preds(robust_classifier, x_test, y_test)
print("\nAccuracy of robust classifier on test set: %.2f%%" % (acc * 100))


# create adversarial test data
attacker_robust = create_attack(attack_type, robust_classifier)
x_test_adv_robust = attacker_robust.generate(x_test)


# evaluate robust classifier on adversarial test set
acc = get_acc_preds(robust_classifier, x_test_adv_robust, y_test)
print("\nAccuracy of robust classifier on adversarial test set: %.2f%%" % (acc * 100))

# Plot robust classifier accuracy over attack strength for multiple specified attacks (in this case fgsm, bim, pgd)
plot_attacks_acc(robust_classifier, x_test, y_test, path_fig, dataset, 'robust_vgg16_attacks_acc')
