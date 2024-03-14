import torch
import os
from pathlib import Path

# def new_weight_update(Inputs=[], gInputs=[]):
#     weight, bias, gamma, beta = Inputs
#     gweight, gbias, ggamma, gbeta = gInputs
#     learning_rate = 0.01

def new_weight_update(Inputs, gInputs):
    weight, bias, gamma, beta = Inputs
    gweight, gbias, ggamma, gbeta = gInputs
    learning_rate = 0.001
    with torch.no_grad():
        for i in range(8):
            # pytorch_model.params[f'W{i}'] -= learning_rate * grads[f'W{i}']
            weight[i] -= learning_rate * gweight[i]
            gamma[i] -= learning_rate * ggamma[i].reshape(-1)
            beta[i] -= learning_rate * gbeta[i].reshape(-1)
        weight[8] -= learning_rate * gweight[8]
        bias -= learning_rate * gbias
    return weight, bias, gamma, beta