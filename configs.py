# Adversarial Learning
# Designate model/data/attack parameters
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (10/16/20)
# Updated (11/29/20)


############### Designate attack parameters ###############
# epsilon : Maximum perturbation that the attacker can introduce
epsilon = 0.006
# eps_step : Attack step size/input variation at each iteration (for pgd, bim)
eps_step = epsilon/4
# max_iter : maximum number of iterations
max_iter = 10
#  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
num_random_init = 20


############### Designate attack parameters for adversarial training#######
# # # # ##### for mnist ####
# # epsilon : Maximum perturbation that the attacker can introduce
# epsilon = 0.2
# # eps_step : Attack step size/input variation at each iteration (for pgd, bim)
# eps_step = epsilon/4
# # max_iter : maximum number of iterations
# max_iter = 10
# #  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
# num_random_init = 20


# # ##### for cifar10 ####
# # epsilon : Maximum perturbation that the attacker can introduce
# epsilon = 0.02
# # eps_step : Attack step size/input variation at each iteration (for pgd, bim)
# eps_step = epsilon/4
# # max_iter : maximum number of iterations
# max_iter = 10
# #  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
# num_random_init = 20


# # # # ##### for ddsm ####
# # epsilon : Maximum perturbation that the attacker can introduce
# epsilon = 0.0007
# # eps_step : Attack step size/input variation at each iteration (for pgd, bim)
# eps_step = epsilon/4
# # max_iter : maximum number of iterations
# max_iter = 10
# #  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
# num_random_init = 5


## for lidc ####
# # epsilon : Maximum perturbation that the attacker can introduce
# epsilon = 0.003
# # eps_step : Attack step size/input variation at each iteration (for pgd, bim)
# eps_step = epsilon/4
# # max_iter : maximum number of iterations
# max_iter = 10
# #  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
# num_random_init = 20


# # ##### for brain_mets ####
# # epsilon : Maximum perturbation that the attacker can introduce
# epsilon = 0.004
# # eps_step : Attack step size/input variation at each iteration (for pgd, bim)
# eps_step = epsilon/4
# # max_iter : maximum number of iterations
# max_iter = 10
# #  Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.
# num_random_init = 20



############### Directories ###############
path = '/home/joelma/'
path_fig = path + 'figures/'
path_csv = path + 'csv/'
