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
	learning_rate = 0.001

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

	config = {'learning_rate': learning_rate, 'momentum': 0.9}
	
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