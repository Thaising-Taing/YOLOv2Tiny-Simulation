import torch
import os
from pathlib import Path

# def new_weight_update(Inputs=[], gInputs=[]):
#     weight, bias, gamma, beta = Inputs
#     gweight, gbias, ggamma, gbeta = gInputs
#     learning_rate = 0.01


def new_weight_update(Inputs=[], gInputs=[], epochs = 0):
    weight, bias, gamma, beta = Inputs
    gweight, gbias, ggamma, gbeta = gInputs
    # learning_rate = 0.001
    if epochs <= 7:
        learning_rate = 1e-2
    elif epochs >= 8:
        learning_rate = 1e-3
    elif epochs >= 20:
        learning_rate = 1e-4
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