# Adversarial Attacks on Medical Deep Learning Models

In this project, we will investigate the robustness of existing medical DL models by testing their performance on adversarial examples from various medical imaging datasets. We will use adversarial training to strengthen them against adversarial attacks. Some questions we will explore are 1) whether some imaging modalities are more susceptible to adversarial attacks than others; 2) use PGD-based adversarial training to increase model robustness
# Code Flow
```
- configs.py # Designates parameters for model, data, and attack
- main.py # Run to apply attack on desired dataset (LIDC, DDSM, brain MRI, MNIST, CIFAR) and use adversarial training to generate robust classifiers
- data_loader.py # Loads in desired dataset
- attacks.py # Creates adversarial attacks
- models.py # Defines models (vgg)
- train.py # Trains models on desired dataset, preprocesses data, splits data into train and test sets
- utils.py # Defines useful functions (plot accuracies/images/predictions, resize images, generate indices of misclassified images, etc.)
 ```

# How to Run
1. Clone Repository
3. Edit ```config.py``` to customize parameters 
4. Run ``` train.py ``` to train DNN models for each classification task
5. Run ``` main.py ``` for adversarial attacking experiments
6. Run ```adv_trainer.py``` to apply PGD-based adversarial training to models
