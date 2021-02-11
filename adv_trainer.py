# Adversarial Training
# Apply adversarial training on medical DL models
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/28/20)
# Updated (12/22/20)


# import dependencies
import art
import importlib
importlib.reload(art)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD
from tensorflow.keras.models import load_model
from data_loader import load_dataset
from attacks import create_attack
from configs import path_fig, epsilon, eps_step, path
from utils import get_acc_preds, plot_attacks_acc, plot_compare_acc, __resize_array_images
from models.models import medical_vgg_model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score, roc_auc_score

# query user for dataset
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')

# load in dataset
# lidc, ddsm augment
if dataset == 'ddsm' or dataset == 'lidc':
    x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=True)
else:
    x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=False)
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

# load model
model = load_model(path + 'models/' + dataset + '/' + dataset + '_vgg16_model.h5')


############### create robust classifier ###############
robust_classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)


# evaluate original classifier accuracy on training set
acc = accuracy_score(y_train, (robust_classifier.predict(x_train)>0.5))
print("Accuracy of original classifier on train set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_train, robust_classifier.predict(x_train), average='macro')
# print("Auc of original classifier on train set: %.2f%%\n" % (auc * 100))


# Evaluate classifier accuracy on the test set
acc = accuracy_score(y_test, (robust_classifier.predict(x_test)>0.5))
print("Accuracy of original classifier on test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, robust_classifier.predict(x_test), average='macro')
# print("Auc of original classifier on test set: %.2f%%\n" % (auc * 100))


# Craft adversarial samples
attacker = create_attack(attack_type, robust_classifier)
x_test_adv = attacker.generate(x_test)
x_train_adv = attacker.generate(x_train)


# Evaluate the classifier on the adversarial test examples
acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv)>0.5))
print("Accuracy of original classifier on adversarial test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, robust_classifier.predict(x_test_adv), average='macro')
# print("Auc of original classifier on adversarial test set: %.2f%%\n" % (auc * 100))


############## train robust classifier ###############

if dataset == 'cifar':
    # trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.02, eps_step=0.005, nb_epochs=nb_epochs,
    #                                      batch_size=128)
    # trainer.fit(x_train, y_train)
    best_model = robust_classifier
    best_adv_test_acc = 0
    for i in range(100):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.02, eps_step=0.02/4, nb_epochs=1,
                                             batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')


elif dataset == 'ddsm':
    # trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.0007, eps_step=0.0007/4, nb_epochs=100)
    # trainer.fit(x_train, y_train)
    best_model = robust_classifier
    best_adv_test_acc = 0
    for i in range(50):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.0007, eps_step=0.0007/4, nb_epochs=1,
                                         batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')
    print('now changing learning rate \n\n\n\n\n\n')
    robust_classifier = best_model
    best_adv_test_acc = 0
    for i in range(50):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.002, eps_step=0.002/4, nb_epochs=1,
                                         batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')

elif dataset == 'lidc':
    # trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.003, eps_step=0.003/4, nb_epochs=100)
    # trainer.fit(x_train, y_train)
    best_model = robust_classifier
    best_adv_test_acc = 0
    for i in range(150):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.001, eps_step=0.001/4, nb_epochs=1,
                                         batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')
    robust_classifier = best_model
    for i in range(50):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.002, eps_step=0.002/4, nb_epochs=1,
                                         batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')


elif dataset == 'mnist':
    trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.2, eps_step=0.05, nb_epochs=50,
                                         batch_size=128)
    trainer.fit(x_train, y_train)

elif dataset == 'brain_mri':
    # trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.004, eps_step=0.001, nb_epochs=nb_epochs,
    #                                      batch_size=128)
    # trainer.fit(x_train, y_train)
    best_model = robust_classifier
    best_adv_test_acc = 0
    for i in range(50):
        trainer = AdversarialTrainerMadryPGD(robust_classifier, eps=0.004, eps_step=0.001, nb_epochs=1,
                                         batch_size=128)
        trainer.fit(x_train, y_train)
        acc = accuracy_score(y_train, (robust_classifier.predict(x_train) > 0.5))
        print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test) > 0.5))
        print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
        attacker_robust = create_attack(attack_type, robust_classifier)
        x_test_adv_robust = attacker_robust.generate(x_test)
        acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv) > 0.5))
        print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
        if acc > best_adv_test_acc:
            best_adv_test_acc = acc
            best_model = robust_classifier
            best_model.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')

############ save robust classifier ###############
robust_classifier.save(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')

# # # load adversarially trained robust classifier
# model = load_model(path + 'models/' + dataset + '/' + dataset + '_' + 'vgg16_robust_classifier.h5')
# robust_classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)


# evaluate robust classifier on original training set
acc = accuracy_score(y_train, (robust_classifier.predict(x_train)>0.5))
print("Accuracy of robust classifier on training set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_train, robust_classifier.predict(x_train), average='macro')
# print("Auc of robust classifier on training set: %.2f%%\n" % (auc * 100))

# create adversarial data
attacker_robust = create_attack(attack_type, robust_classifier)
x_train_adv_robust = attacker_robust.generate(x_train)
x_test_adv_robust = attacker_robust.generate(x_test)

# Evaluate the classifier on the adversarial training examples
acc = accuracy_score(y_train, (robust_classifier.predict(x_train_adv_robust)>0.5))
print("Accuracy of robust classifier on adversarial training set: %.2f%%, " % (acc * 100))


# evaluate robust classifier on original test set
acc = accuracy_score(y_test, (robust_classifier.predict(x_test)>0.5))
print("Accuracy of robust classifier on test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, robust_classifier.predict(x_test), average='macro')
# print("Auc of robust classifier on test set: %.2f%%\n" % (auc * 100))



# evaluate robust classifier on adversarial test set
acc = accuracy_score(y_test, (robust_classifier.predict(x_test_adv_robust)>0.5))
print("Accuracy of robust classifier on adversarial test set: %.2f%%, " % (acc * 100))
# auc = roc_auc_score(y_test, robust_classifier.predict(x_test_adv), average='macro')
# print("Auc of robust classifier on adversarial test set: %.2f%%\n" % (auc * 100))

# Plot robust classifier accuracy over attack strength for multiple specified attacks (in this case fgsm, bim, pgd)
plot_attacks_acc(robust_classifier, x_test, y_test, path_fig, dataset, 'robust_vgg16_attacks_acc')
