import torch
import os
from pathlib import Path

import torch
import os
from pathlib import Path

torch.manual_seed(3407)

def sgd_momentum(w, dw, config=None):

  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)

  v = config.get('velocity', torch.zeros_like(w))

  next_w = None

  v = config['momentum']*v - config['learning_rate'] * dw
  next_w = w + v

  config['velocity'] = v

  return next_w, config

def new_weight_update(Inputs=[], gInputs=[], epochs = 0):
	weight, bias, gamma, beta = Inputs
	gweight, gbias, ggamma, gbeta = gInputs
	learning_rate = 0.005
	for i in range(8):
		weight[i] = weight[i].cuda()
		gamma[i] = gamma[i].cuda()
		beta[i] = beta[i].cuda()
		gweight[i] = gweight[i].cuda()
		ggamma[i] = ggamma[i].cuda()
		gbeta[i] = gbeta[i].cuda()
	weight[8] = weight[8].cuda()
	gweight[8] = gweight[8].cuda()
	bias = bias.cuda()
	gbias = gbias.cuda()

	with torch.no_grad():
		for i in range(8):
			# pytorch_model.params[f'W{i}'] -= learning_rate * grads[f'W{i}']
			weight[i] -= learning_rate * gweight[i]
			gamma[i] -= learning_rate * ggamma[i].reshape(-1)
			beta[i] -= learning_rate * gbeta[i].reshape(-1)
		weight[8] -= learning_rate * gweight[8]
		bias -= learning_rate * gbias
	return weight, bias, gamma, beta


def new_weight_update_two(Inputs=[], gInputs=[], epochs = 0):
	weight, bias, gamma, beta = Inputs
	gweight, gbias, ggamma, gbeta = gInputs
	# learning_rate = 0.001
	if epochs <= 10:
		learning_rate = 5e-3
	elif epochs <= 20:
		learning_rate = 5e-4
	elif epochs <= 30:
		learning_rate = 5e-5
	else:
		learning_rate = 5e-6
	for i in range(8):
		weight[i] = weight[i].cuda()
		gamma[i] = gamma[i].cuda()
		beta[i] = beta[i].cuda()
		gweight[i] = gweight[i].cuda()
		ggamma[i] = ggamma[i].cuda()
		gbeta[i] = gbeta[i].cuda()
	weight[8] = weight[8].cuda()
	gweight[8] = gweight[8].cuda()
	bias = bias.cuda()
	gbias = gbias.cuda()

	with torch.no_grad():
		for i in range(8):
			# pytorch_model.params[f'W{i}'] -= learning_rate * grads[f'W{i}']
			weight[i] -= learning_rate * gweight[i]
			gamma[i] -= learning_rate * ggamma[i].reshape(-1)
			beta[i] -= learning_rate * gbeta[i].reshape(-1)
		weight[8] -= learning_rate * gweight[8]
		bias -= learning_rate * gbias
	return weight, bias, gamma, beta


def sgd_momentum_update(Inputs=[], gInputs=[], epochs = 0, optimizer_config = None):
	weight, bias, gamma, beta = Inputs
	gweight, gbias, ggamma, gbeta = gInputs

	for i in range(8):
		gweight[i] = torch.clamp(gweight[i], -1, 1)
		gbias[i] = torch.clamp(gbias[i], -1, 1)
		ggamma[i] = torch.clamp(ggamma[i], -1, 1)
		gbeta[i] = torch.clamp(gbeta[i], -1, 1)

	gweight[8] = torch.clamp(gweight[8], -1, 1)
	gbias = torch.clamp(gbias, -1, 1)

	#  Learning Rate
	# Initial LR = 0.01 gives NaN for LN
	# Initial LR = 0.001 --- gives best result when training from scratch
	# Initial LR = 0.0001 --- should be better for pre-trained results.
	# Initial LR = 0.00001 is probably too slow.

	# initial_lr = 0.0001 # initial learning rate
	# warmup_epochs = 10
	# plateau_epochs = 30
	# decay_rate = 0.1

	initial_lr = 0.001  # Initial learning rate
	warmup_epochs = 5   # Number of epochs for warmup
	plateau_epochs = 30 # Number of epochs for plateau phase
	decay_rate = 0.98    # Decay rate

	if epochs < warmup_epochs:
		learning_rate = initial_lr * (epochs + 1) / warmup_epochs
	elif epochs < plateau_epochs:
		learning_rate = initial_lr
	else:
		learning_rate = initial_lr * (decay_rate ** (epochs - plateau_epochs))

	i = 0

	for i in range(8):
		weight[i] = weight[i].cuda()
		gamma[i] = gamma[i].cuda()
		beta[i] = beta[i].cuda()
		gweight[i] = gweight[i].cuda()
		ggamma[i] = ggamma[i].cuda()
		gbeta[i] = gbeta[i].cuda()
	weight[8] = weight[8].cuda()
	gweight[8] = gweight[8].cuda()
	bias = bias.cuda()
	gbias = gbias.cuda()

	config = {'learning_rate': learning_rate, 'momentum': 0.7}
	
	with torch.no_grad():
		for i in range(8):
			config = optimizer_config['W{}'.format(i)]
			config['learning_rate'] = learning_rate
			weight[i], next_config = sgd_momentum(weight[i], gweight[i], config)
			optimizer_config['W{}'.format(i)] = next_config

		# for i in range(8):
			config = optimizer_config['gamma{}'.format(i)]
			config['learning_rate'] = learning_rate
			gamma[i], next_config = sgd_momentum(gamma[i], ggamma[i].reshape(-1), config)
			optimizer_config['gamma{}'.format(i)] = next_config
		
		# for i in range(8):
			config = optimizer_config['beta{}'.format(i)]
			config['learning_rate'] = learning_rate
			beta[i], next_config = sgd_momentum(beta[i], gbeta[i].reshape(-1), config)
			optimizer_config['beta{}'.format(i)] = next_config

		config = optimizer_config['W8']
		config['learning_rate'] = learning_rate
		weight[8], next_config = sgd_momentum(weight[8], gweight[8], config)
		optimizer_config['W8'] = next_config

		config = optimizer_config['b8']
		config['learning_rate'] = learning_rate
		bias, next_config = sgd_momentum(bias, gbias, config)
		optimizer_config['b8'] = next_config
	
	return (weight, bias, gamma, beta), optimizer_config

	# with torch.no_grad():
	#     for i in range(8):
	#         # pytorch_model.params[f'W{i}'] -= learning_rate * grads[f'W{i}']
	#         weight[i] -= learning_rate * gweight[i]
	#         gamma[i] -= learning_rate * ggamma[i].reshape(-1)
	#         beta[i] -= learning_rate * gbeta[i].reshape(-1)
	#     weight[8] -= learning_rate * gweight[8]
	#     bias -= learning_rate * gbias
	# return weight, bias, gamma, beta