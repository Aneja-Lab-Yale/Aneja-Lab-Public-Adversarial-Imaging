# Adversarial Attacks on Medical Deep Learning Models

Code accompanying the paper: Using Adversarial Images to Assess the Robustness of Deep Learning Models Trained on Diagnostic Images in Oncology. JCO Clin Cancer Inform. 2022 Feb;6:e2100170. https://ascopubs.org/doi/10.1200/CCI.21.00170

In this project, we will investigate the robustness of existing medical DL models by testing their performance on adversarial examples from various medical imaging datasets. We will use adversarial training to strengthen them against adversarial attacks. Some questions we will explore are 1) whether some imaging modalities are more susceptible to adversarial attacks than others; 2) how to use PGD-based adversarial training to increase model robustness; 3) whether image level adversarial susceptibility is a useful metric to improve model performance.

# Code Flow
```
- configs.py # Designates parameters for adversarial attack and training experiments
- utils.py # Defines useful functions used in experiments
- data_loader.py # Loads in desired training and test set, preprocesses data
- data_generator.py # Performs data augmentations (flips, rotations) on medical datasets
- models.py # Defines models (VGG-16 model)
- train.py # Trains model on data
- attacks.py # Creates adversarial attacks
- main.py # Applies attack on desired dataset and evaluates model accuracy on adversarial examples
- adv_trainer.py # Applies PGD-based adversarial training to models
- adv_sensitivity.py # Evaluates the adversarial susceptibility of each image to eliminate most vulnerable images, measures model accuracy on remaining dataset
 ```

# How to Run
1. Clone Repository from Github
2. Install necessary dependencies ```python pip3 install -r Requirements.txt```
3. Edit ```configs.py``` to customize parameters 
4. Run ```data_generator.py``` to augment medical datasets
5. Run ``` train.py ``` to train DNN models
6. Run ``` main.py ``` for adversarial attacking experiments
7. Run ```adv_trainer.py``` for adversarial training experiments
8. Run ```adv_sensitivity.py``` for adversarial sensitivity experiments

# Examples
We show example images along with their adversarial counterparts (generated by FGSM, PGD, BIM attacks) from each our datasets: lung CT, mammography, brain MRI, MNIST, and CIFAR-10. 

![image](https://user-images.githubusercontent.com/64549018/105315197-f6da9000-5b8c-11eb-989b-f5ab1359114f.png)
