# Adversarial Learning
# Function tools to be utilized in adversarial experiments
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (9/01/20)
# Updated (12/22/20)


# import dependencies
import numpy as np
import cv2
import matplotlib.pyplot as plt
from configs import path_csv
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, ProjectedGradientDescent
import csv
import os


# Calculates accuracy of classifier model on labeled image dataset
def get_acc_preds(classifier, x, y):
    '''
    Description:
        This function takes in a classifier model and image dataset with corresponding labels,
        and evaluates the accuracy of model on dataset
    :param classifier: model to be evaluated
    :param x: list of images to be predicted on
    :param y: list of image labels
    :return: Model accuracy
    '''
    preds = np.argmax(classifier.predict(x), axis=1)
    acc = np.sum(preds == np.argmax(y, axis=1)) / y.shape[0]
    return acc


# Resizes array of images to new dimensions
# copied from https://github.com/Aneja-Lab-Yale/Domain_Adaptation/blob/master/Transfer_Learning/MNIST_CNN_SA.py
def __resize_array_images(array_images, size):
    '''
    Description:
        This function takes in an array of images and resizes each image to have specified dimensions (square)
    :param array_images: list of images to be resized
    :param size: desired height/width of new image
    :return: array of newly resized image
    '''
    new_array = []
    for i in range(len(array_images)):
        img = cv2.resize(array_images[i], (size, size), interpolation=cv2.INTER_CUBIC)
        new_array.append(img)
    return np.array(new_array)


# creates plot showing accuracy of classifier against different attacks
# Employed (10/01/20)
# Debugged (12/01/20)
# Created by Marina Joel
def plot_attacks_acc(classifier, x, y, path_fig, dataset, title):
    '''
    Description:
        This function takes in a classifier model and a list of images with labels and creates
        a plot showing how the accuracy of model on the dataset decreases as attack strength (perturbation size)
        increases for 3 different attacks (FGSM, PGD, BIM).
    :param classifier: model to be evaluated
    :param x: list of images to be predicted on
    :param y: labels of images
    :param path_fig: path to save the plot figure
    :param dataset: name of dataset (e.g. mnist, cifar, ddsm, brain_mri, lidc)
    :param title: title to define plot figure
    :return: Figure will be saved with title
    '''
    if dataset == 'ddsm':
        eps_range = [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007, 0.008]
        step_size = 0.001
    elif dataset == 'brain_mri':
        eps_range = [0.0001, 0.0005, 0.001, 0.0013, 0.0016, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012]
        step_size = 0.001
    elif dataset == 'mnist':
        eps_range = [0.0001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        step_size = 0.05
    elif dataset == 'cifar':
        eps_range = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
        step_size = 0.005
    elif dataset == 'lidc':
        eps_range = [0.0001, 0.0003, 0.0006, 0.0008, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0023, 0.0026, 0.0028, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
        step_size = 0.001
    nb_correct_fgsm = []
    nb_correct_pgd = []
    nb_correct_bim = []
    for eps in eps_range:
        attacker_fgsm = FastGradientMethod(classifier, eps=eps)
        attacker_pgd = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps/4, max_iter=10,
                                            num_random_init=5)
        attacker_bim = BasicIterativeMethod(classifier, eps=eps, eps_step=eps/10, max_iter=10)
        x_fgsm = attacker_fgsm.generate(x)
        x_pgd = attacker_pgd.generate(x)
        x_bim = attacker_bim.generate(x)
        x_pred_fgsm = np.argmax(classifier.predict(x_fgsm), axis=1)
        nb_correct_fgsm += [np.sum(x_pred_fgsm == np.argmax(y, axis=1))]
        x_pred_pgd = np.argmax(classifier.predict(x_pgd), axis=1)
        nb_correct_pgd += [np.sum(x_pred_pgd == np.argmax(y, axis=1))]
        x_pred_bim = np.argmax(classifier.predict(x_bim), axis=1)
        nb_correct_bim += [np.sum(x_pred_bim == np.argmax(y, axis=1))]

    fig, ax = plt.subplots()
    ax.plot(np.array(eps_range) / step_size, 100 * np.array(nb_correct_fgsm) / y.shape[0], 'b--', label='FGSM')
    ax.plot(np.array(eps_range) / step_size, 100 * np.array(nb_correct_pgd) / y.shape[0], 'r--', label='PGD')
    ax.plot(np.array(eps_range) / step_size, 100 * np.array(nb_correct_bim) / y.shape[0], 'g--', label='BIM')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.xlabel('Perturbation (x $10^{-3}$)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(path_fig + dataset + '/' + title + '.png')
    plt.clf()

    data = [np.array(eps_range), np.array(nb_correct_fgsm) / y.shape[0], np.array(nb_correct_pgd) / y.shape[0], np.array(nb_correct_bim) / y.shape[0]]
    out = csv.writer(open(path_csv + dataset + '/' + title + '.csv', "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    out.writerows(zip(*data))
    return 0



# Plot Training and Validation
# Employed (11-7-19)
# Debugged (8-3-20)
# Created By (Sanjay Aneja, MD)
# Plots training accuracy and validation
def plot_tv(
        fit_model,
        path,
        model_title,
        filename,
        metrics
):
    """
    Description: This function plots training and validation curves for your DL model
    Param:
        fit_model = variable storing fit DL model from Keras
        path = folder you would like to save pictures
        model_title = keyword (string) to define model
        metrics = metrics list from keras model
    Return:
        Training and validation curves (loss and accuracy) for your DL model
        Figure will be saved with title [Timestamp_keyword]
    """
    # Plot the results
    for i in metrics:
        if i =="Accuracy":
            train_acc = fit_model.history['accuracy']
            val_acc = fit_model.history['val_accuracy']
            fig1,(fig_acc) = plt.pyplot.subplots(nrows=1, ncols=1, figsize=(8, 5))
            xa = np.arange(len(val_acc))
            fig_acc.plot(xa, train_acc)
            fig_acc.plot(xa, val_acc)
            fig_acc.set(xlabel='Epochs')
            fig_acc.set(ylabel='Accuracy')
            fig_acc.set(title='Training Accuracy vs Validation Accuracy')
            fig1.legend(['Train', 'Validation'], loc=1, borderaxespad=1)
            #fig.grid('True')
            fig1.savefig((os.path.join(path, (filename + '_' + i + '_' + model_title + '.png'))))
            continue
        elif i =="AUC":
            train_acc = fit_model.history['AUC']
            val_acc = fit_model.history['val_AUC']
            fig2,(fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
            xa = np.arange(len(val_acc))
            fig_acc.plot(xa, train_acc)
            fig_acc.plot(xa, val_acc)
            fig_acc.set(xlabel='Epochs')
            fig_acc.set(ylabel='AUC')
            fig_acc.set(title='Training AUC vs Validation AUC')
            fig2.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
            #fig.grid('True')
            fig2.savefig((os.path.join(path, (filename + '_' + i + '_' + model_title + '.png'))))
            continue
        elif i == "Binary_Accuracy":
            train_acc = fit_model.history['Binary_Accuracy']
            val_acc = fit_model.history['val_Binary_Accuracy']
            fig3, (fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
            xa = np.arange(len(val_acc))
            fig_acc.plot(xa, train_acc)
            fig_acc.plot(xa, val_acc)
            fig_acc.set(xlabel='Epochs')
            fig_acc.set(ylabel='Accuracy')
            fig_acc.set(title='Training Accuracy vs Validation Accuracy')
            fig3.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
            #fig.grid('True')
            fig3.savefig((os.path.join(path, (filename + '_' + i + '_' + model_title + '.png'))))
            continue
        elif i == "sparse_categorical_accuracy":
            train_acc = fit_model.history['sparse_categorical_accuracy']
            val_acc = fit_model.history['val_sparse_categorical_accuracy']
            fig3, (fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
            xa = np.arange(len(val_acc))
            fig_acc.plot(xa, train_acc)
            fig_acc.plot(xa, val_acc)
            fig_acc.set(xlabel='Epochs')
            fig_acc.set(ylabel='Accuracy')
            fig_acc.set(title='Training Accuracy vs Validation Accuracy')
            fig3.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
            #fig.grid('True')
            fig3.savefig((os.path.join(path, (filename + '_' + i + '_' + model_title + '.png'))))
            continue
        elif i =="True_Pos":
            try:
                train_acc = fit_model.history['True_Pos']
                val_acc = fit_model.history['val_True_Pos']
                train_tneg=fit_model.history['True_Neg']
                val_tneg=fit_model.history['val_True_Neg']
                train_fp = fit_model.history['False_Pos']
                val_fp = fit_model.history['val_False_Pos']
                train_fn = fit_model.history['False_Neg']
                val_fn = fit_model.history['val_False_Neg']
                fig4, (fig_acc,fig_neg ) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                xa = np.arange(len(val_acc))
                fig_acc.plot(xa, train_acc,label='Train TP')
                fig_acc.plot(xa, val_acc, label='Val TP')
                fig_acc.plot(xa, train_fp, label='Train FP')
                fig_acc.plot(xa, val_fp, label = 'Val FP')
                fig_acc.set(xlabel='Epochs')
                fig_acc.set(ylabel='Positives')
                fig_acc.set(title='Training Positives vs Validation Positives')
                fig_neg.plot(xa,train_tneg, label='Train TN')
                fig_neg.plot(xa,val_tneg, label='Val TN')
                fig_neg.plot(xa,train_fn, label = 'Train FN')
                fig_neg.plot(xa,val_fn, label = 'Val FN')
                fig_neg.set(xlabel='Epochs')
                fig_neg.set(ylabel='Negatives')
                fig_neg.set(title='Training Negatives vs Validation Negatives')
                fig4.legend(loc=1, borderaxespad=2)
                #fig.grid('True')
                fig4.savefig((os.path.join(path, (filename + '_Positive_Negatives_' + model_title + '.png'))))
                continue
            except:
                print('ERROR: For TP/TN/FP/FN analysis all metrics must be recorded')
                continue
        elif i =='Precision_Specificity':
            try:
                train_acc = fit_model.history['Precision_Specificity']
                val_acc = fit_model.history['val_Precision_Specificity']
                train_tneg = fit_model.history['Recall_Sensitivity']
                val_tneg = fit_model.history['val_Recall_Sensitivity']
                fig5, (fig_acc, fig_neg) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                xa = np.arange(len(val_acc))
                fig_acc.plot(xa, train_acc)
                fig_acc.plot(xa, val_acc)
                fig_neg.plot(xa, train_tneg)
                fig_neg.plot(xa, val_tneg)
                fig_acc.set(xlabel='Epochs')
                fig_acc.set(ylabel='Precision_Specificity')
                fig_acc.set(title='Precision_Specificity')
                fig_neg.set(xlabel='Epochs')
                fig_neg.set(ylabel='Recall_Sensitivity')
                fig_neg.set(title='Recall_Sensitivity')
                fig5.legend(['Train', 'Validation'], loc=1, borderaxespad=1)
                #fig.grid('True')
                fig5.savefig((os.path.join(path, (filename + '_Precision_Recall_' + model_title + '.png'))))
            except:
                print('ERROR: For Precision/Recall Analysis, all metrics must be recorded')
                continue
        else: print("ERROR: NOT GRAPHED-", i)
    train_loss = fit_model.history['loss']
    val_loss = fit_model.history['val_loss']
    fig6, (fig_loss) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    xc = np.arange(len(train_loss))
    fig_loss.plot(xc, train_loss)
    fig_loss.plot(xc, val_loss)
    fig_loss.set(xlabel='Epochs')
    fig_loss.set(ylabel='Loss')
    fig_loss.set(title='Training Loss vs Validation Loss')
    fig6.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
    #fig.grid('True')
    fig6.savefig((os.path.join(path, (filename + '_loss_' + model_title + '.png'))))
