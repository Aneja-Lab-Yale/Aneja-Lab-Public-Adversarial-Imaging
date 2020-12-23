# Adversarial Learning
# Generates adversarial attacks
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/16/20)
# Updated (12/22/20)

# import dependencies
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, ProjectedGradientDescent
from configs import epsilon, eps_step, max_iter, num_random_init


def create_attack(attack_type, classifier):
    if attack_type == 'fgsm':
        # Create a Fast Gradient Sign Method instance, specifying the classifier model, eps : attack step size
        attacker = FastGradientMethod(classifier, eps=epsilon)
    elif attack_type == 'pgd':
        # Create a Projected Gradient Descent instance, specifying the classifier model, eps : Maximum perturbation that
        # attacker can introduce, eps_step : Attack step size/input variation at each iteration,
        # max_iter : maximum number of iterations, num_random_init : number of random initializations
        attacker = ProjectedGradientDescent(classifier, eps=epsilon, eps_step=eps_step, max_iter=max_iter,
                                            num_random_init=num_random_init)
    elif attack_type == 'bim':
        # Create a Basic Iterative Method instance, specifying the classifier model, eps : Maximum perturbation,
        # eps_step : attack step size, max_iter : maximum number of iterations
        attacker = BasicIterativeMethod(classifier, eps=epsilon, eps_step=epsilon/max_iter, max_iter=max_iter)
    else:
        print('No supported attack specified')
        exit(0)
    return attacker
