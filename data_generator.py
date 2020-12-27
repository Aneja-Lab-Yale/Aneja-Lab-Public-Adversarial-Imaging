# Adversarial Learning
# Augments dataset using simple fips and rotations
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/16/20)
# Updated (12/22/20)


from data_loader import load_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from configs import path
import numpy as np


# query user for dataset name
dataset = input('Enter dataset to be used (brain_mri, ddsm, lidc)\n')

# validate that dataset type is supported (medical datasets only)
if (dataset != 'lidc') & (dataset != 'ddsm') & (dataset != 'brain_mri'):
    print('dataset name invalid\n')
    exit(0)

# load dataset
x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=False)

datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, rotation_range=20, fill_mode="nearest")
datagen.fit(x_train)

x_train_aug = []
y_train_aug = []

batches = 0
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=1):
    x_train_aug.append(x_batch)
    y_train_aug.append(y_batch)
    batches += 1
    if dataset == 'brain_mri':
        if batches >= 4 * len(x_train):
            break
    else:
        if batches >= 10 * len(x_train):
            break

save_path = path + dataset
# if dataset == 'brain_mri':
#     save_path = '/mys3bucket/' + dataset
np.save(save_path + '/' + 'x_train_aug.npy', x_train_aug)
np.save(save_path + '/' + 'y_train_aug.npy', y_train_aug)
