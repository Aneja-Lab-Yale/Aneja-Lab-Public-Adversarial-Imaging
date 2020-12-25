# Adversarial Attacks on Medical Deep Learning Models

In this project, we will investigate the robustness of existing medical DL models by testing their performance on adversarial examples from various medical imaging datasets. We will use adversarial training to strengthen them against adversarial attacks. Some questions we will explore are 1) whether some imaging modalities are more susceptible to adversarial attacks than others; 2) use PGD-based adversarial training to increase model robustness
# Code Flow
```
- configs.py # Designates parameters for adversarial attack and training experiments
- utils.py # Defines useful functions used in experiments
- data_loader.py # Loads in desired training and test set, preprocesses data
- models.py # Defines models (VGG-16 model)
- train.py # Trains model on data
- attacks.py # Creates adversarial attacks
- main.py # Applies attack on desired dataset and evaluates model accuracy on adversarial examples
- adv_trainer.py # Applies PGD-based adversarial training to models


 ```

# How to Run
1. Clone Repository
3. Edit ```config.py``` to customize parameters 
4. Run ``` train.py ``` to train DNN models
5. Run ``` main.py ``` for adversarial attacking experiments
6. Run ```adv_trainer.py``` for adversarial training experiments
