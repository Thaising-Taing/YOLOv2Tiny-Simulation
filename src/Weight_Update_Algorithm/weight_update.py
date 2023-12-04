import torch

    
def get_weights(model):
    """
    Get the weights of a model.

    Args:
        model (nn.Module): The model to extract weights from.

    Returns:
        dict: A dictionary containing the weights of each layer in the model.
    """

    # Initialize an empty dictionary to store the weights
    weights = {}

    # Iterate over the named parameters of the model
    for layer_name, layer_params in model.named_parameters():
        # Check if the parameter is a weight parameter
        if 'weight' in layer_name:
            # Detach the parameter from the computation graph, move it to the CPU, and convert it to a numpy array
            weight_array = layer_params.detach().cpu().numpy()
            # Store the weight array in the dictionary using the layer name as the key
            weights[layer_name.split('.')[0]] = weight_array

    # Return the dictionary of weights
    return weights

def get_bias(model):
    """
    Get the bias values from the given model.

    Args:
        model (nn.Module): The model from which to retrieve the bias values.

    Returns:
        dict: A dictionary containing the bias values for each layer.

    """
    b = {}
    for name, param in model.named_parameters():
        if 'bias' in name:
            b[name.split('.')[0]] = param.detach().cpu().numpy()
    return b
    
def get_weight_grads(model):
    """
    Retrieves the weight gradients from the model.

    Args:
        model (nn.Module): The model to retrieve the weight gradients from.

    Returns:
        dict: A dictionary containing the weight gradients for each layer and batch normalization parameter.
    """
    # Initialize the dictionary to store the weight gradients
    gw = {}

    # Retrieve the weight gradients for the convolutional layers
    for i in range(1, 9):
        conv_name = f"conv{i}"
        gw[conv_name] = model._modules[conv_name].weight.grad.detach().cpu().numpy()

    # Retrieve the weight gradient for the last convolutional layer
    # gw["conv9.0"] = model._modules["conv9.0"].weight.grad.detach().cpu().numpy()
    gw["conv9"] = model.conv9[0].weight.grad.detach().cpu().numpy()

    # Retrieve the weight gradients for the batch normalization layers
    for i in range(1, 9):
        bn_name = f"bn{i}"
        gw[bn_name] = model._modules[bn_name].weight.grad.detach().cpu().numpy()

    return gw

def get_bias_grads(model):
    gb = {}
    
    # Get the gradient of the bias for each batch normalization layer
    for i in range(1, 9):
        bn_name = f'bn{i}'
        bn_bias_grad = getattr(model, bn_name).bias.grad.detach().cpu().numpy()
        gb[bn_name] = bn_bias_grad
    
    # Get the gradient of the bias for the convolutional layer
    conv_bias_grad = getattr(model.conv9[0], 'bias').grad.detach().cpu().numpy()
    gb['conv9'] = conv_bias_grad

    return gb
   
   

def get_w_and_b(model):
    w  = get_weights(model)
    gw = get_weight_grads(model)
    b  = get_bias(model)
    gb = get_bias_grads(model)
    return w, gw, b, gb

def update_weight_values(custom_model, w, gw, b, gb):
    for name, param in custom_model.named_parameters():
        if name == "conv1.weight": 
            param.data = torch.from_numpy( w[ 'conv1']  )
            param.grad = torch.from_numpy( gw['conv1']  )

        if name == "conv2.weight": 
            param.data = torch.from_numpy( w[ 'conv2']  )
            param.grad = torch.from_numpy( gw['conv2']  )

        if name == "conv3.weight": 
            param.data = torch.from_numpy( w[ 'conv3']  )
            param.grad = torch.from_numpy( gw['conv3']  )

        if name == "conv4.weight": 
            param.data = torch.from_numpy( w[ 'conv4']  )
            param.grad = torch.from_numpy( gw['conv4']  )

        if name == "conv5.weight": 
            param.data = torch.from_numpy( w[ 'conv5']  )
            param.grad = torch.from_numpy( gw['conv5']  )

        if name == "conv6.weight": 
            param.data = torch.from_numpy( w[ 'conv6']  )
            param.grad = torch.from_numpy( gw['conv6']  )

        if name == "conv7.weight": 
            param.data = torch.from_numpy( w[ 'conv7']  )
            param.grad = torch.from_numpy( gw['conv7']  )

        if name == "conv8.weight": 
            param.data = torch.from_numpy( w[ 'conv8']  )
            param.grad = torch.from_numpy( gw['conv8']  )

        if name == "conv9.0.weight": 
            param.data = torch.from_numpy( w[ 'conv9']  )
            param.grad = torch.from_numpy( gw['conv9']  )

        if name == "bn1.weight": 
            param.data = torch.from_numpy( w[ 'bn1']    )
            param.grad = torch.from_numpy( gw['bn1']    )

        if name == "bn2.weight": 
            param.data = torch.from_numpy( w[ 'bn2']    )
            param.grad = torch.from_numpy( gw['bn2']    )

        if name == "bn3.weight": 
            param.data = torch.from_numpy( w[ 'bn3']    )
            param.grad = torch.from_numpy( gw['bn3']    )

        if name == "bn4.weight": 
            param.data = torch.from_numpy( w[ 'bn4']    )
            param.grad = torch.from_numpy( gw['bn4']    )

        if name == "bn5.weight": 
            param.data = torch.from_numpy( w[ 'bn5']    )
            param.grad = torch.from_numpy( gw['bn5']    )

        if name == "bn6.weight": 
            param.data = torch.from_numpy( w[ 'bn6']    )
            param.grad = torch.from_numpy( gw['bn6']    )

        if name == "bn7.weight": 
            param.data = torch.from_numpy( w[ 'bn7']    )
            param.grad = torch.from_numpy( gw['bn7']    )

        if name == "bn8.weight": 
            param.data = torch.from_numpy( w[ 'bn8']    )
            param.grad = torch.from_numpy( gw['bn8']    )

        if name == "bn1.bias": 
            param.data = torch.from_numpy( b[ 'bn1']    )
            param.grad = torch.from_numpy( gb['bn1']    )

        if name == "bn2.bias": 
            param.data = torch.from_numpy( b[ 'bn2']    )
            param.grad = torch.from_numpy( gb['bn2']    )

        if name == "bn3.bias": 
            param.data = torch.from_numpy( b[ 'bn3']    )
            param.grad = torch.from_numpy( gb['bn3']    )

        if name == "bn4.bias": 
            param.data = torch.from_numpy( b[ 'bn4']    )
            param.grad = torch.from_numpy( gb['bn4']    )

        if name == "bn5.bias": 
            param.data = torch.from_numpy( b[ 'bn5']    )
            param.grad = torch.from_numpy( gb['bn5']    )

        if name == "bn6.bias": 
            param.data = torch.from_numpy( b[ 'bn6']    )
            param.grad = torch.from_numpy( gb['bn6']    )

        if name == "bn7.bias": 
            param.data = torch.from_numpy( b[ 'bn7']    )
            param.grad = torch.from_numpy( gb['bn7']    )

        if name == "bn8.bias": 
            param.data = torch.from_numpy( b[ 'bn8']    )
            param.grad = torch.from_numpy( gb['bn8']    )

        if name == "conv9.0.bias": 
            param.data = torch.from_numpy( b[ 'conv9']  )
            param.grad = torch.from_numpy( gb['conv9']  )
            
            
    return custom_model


def set_weight_values_FPGA(custom_model, Inputs, gInputs):
    """
    Updates the weight values of a custom model with the given inputs.
    Args:
        custom_model (torch.nn.Module): The custom model to update.
        Inputs (Tuple): A tuple containing the weight, bias, gamma_weight_bn, and beta_bn values.
        gInputs (Tuple): A tuple containing the gradient of weight, bias, gamma_weight_bn, and beta_bn values.
    Returns:
        custom_model (torch.nn.Module): The updated custom model.
    """
    
    Weight,   Bias,  Gamma_WeightBN,  BetaBN = Inputs    # Gamma is weight for BN, Beta is Bias for BN
    gWeight, gBias, gGamma_WeightBN, gBetaBN = gInputs 
    
    for name, param in custom_model.named_parameters():
        if name == "conv1.weight": 
            param.data = Weight[0]
            param.grad = gWeight[0]

        if name == "conv2.weight": 
            param.data = Weight[1]
            param.grad = gWeight[1]

        if name == "conv3.weight": 
            param.data = Weight[2]
            param.grad = gWeight[2]

        if name == "conv4.weight": 
            param.data = Weight[3]
            param.grad = gWeight[3]

        if name == "conv5.weight": 
            param.data = Weight[4]
            param.grad = gWeight[4]

        if name == "conv6.weight": 
            param.data = Weight[5]
            param.grad = gWeight[5]

        if name == "conv7.weight": 
            param.data = Weight[6]
            param.grad = gWeight[6]

        if name == "conv8.weight": 
            param.data = Weight[7]
            param.grad = gWeight[7]

        if name == "conv9.0.weight": 
            param.data = Weight[8]
            param.grad = gWeight[8]

        if name == "bn1.weight": 
            param.data = Gamma_WeightBN[0]
            param.grad = gGamma_WeightBN[0]

        if name == "bn2.weight": 
            param.data = Gamma_WeightBN[1]
            param.grad = gGamma_WeightBN[1]

        if name == "bn3.weight": 
            param.data = Gamma_WeightBN[2]
            param.grad = gGamma_WeightBN[2]

        if name == "bn4.weight": 
            param.data = Gamma_WeightBN[3]
            param.grad = gGamma_WeightBN[3]

        if name == "bn5.weight": 
            param.data = Gamma_WeightBN[4]
            param.grad = gGamma_WeightBN[4]

        if name == "bn6.weight": 
            param.data = Gamma_WeightBN[5]
            param.grad = gGamma_WeightBN[5]

        if name == "bn7.weight": 
            param.data = Gamma_WeightBN[6]
            param.grad = gGamma_WeightBN[6]

        if name == "bn8.weight": 
            param.data = Gamma_WeightBN[7]
            param.grad = gGamma_WeightBN[7]

        if name == "bn1.bias": 
            param.data = BetaBN[0]
            param.grad = gBetaBN[0]

        if name == "bn2.bias": 
            param.data = BetaBN[1]
            param.grad = gBetaBN[1]

        if name == "bn3.bias": 
            param.data = BetaBN[2]
            param.grad = gBetaBN[2]

        if name == "bn4.bias": 
            param.data = BetaBN[3]
            param.grad = gBetaBN[3]

        if name == "bn5.bias": 
            param.data = BetaBN[4]
            param.grad = gBetaBN[4]

        if name == "bn6.bias": 
            param.data = BetaBN[5]
            param.grad = gBetaBN[5]

        if name == "bn7.bias": 
            param.data = BetaBN[6]
            param.grad = gBetaBN[6]

        if name == "bn8.bias": 
            param.data = BetaBN[7]
            param.grad = gBetaBN[7]

        if name == "conv9.0.bias": 
            param.data = Bias
            param.grad = gBias
            
    return custom_model

def get_weight_values_FPGA(custom_model, Inputs, gInputs):
    """
    Updates the weight values of a custom model with the given inputs.
    Args:
        custom_model (torch.nn.Module): The custom model to update.
        Inputs (Tuple): A tuple containing the weight, bias, gamma_weight_bn, and beta_bn values.
        gInputs (Tuple): A tuple containing the gradient of weight, bias, gamma_weight_bn, and beta_bn values.
    Returns:
        custom_model (torch.nn.Module): The updated custom model.
    """
    
    Weight,   Bias,  Gamma_WeightBN,  BetaBN = Inputs    # Gamma is weight for BN, Beta is Bias for BN
    
    for name, param in custom_model.named_parameters():
        if name == "conv1.weight": 
            Weight[0] = param.data

        if name == "conv2.weight": 
            Weight[1] = param.data

        if name == "conv3.weight": 
            Weight[2] = param.data

        if name == "conv4.weight": 
            Weight[3] = param.data

        if name == "conv5.weight": 
            Weight[4] = param.data

        if name == "conv6.weight": 
            Weight[5] = param.data

        if name == "conv7.weight": 
            Weight[6] = param.data

        if name == "conv8.weight": 
            Weight[7] = param.data

        if name == "conv9.0.weight": 
            Weight[8] = param.data

        if name == "bn1.weight": 
            Gamma_WeightBN[0] = param.data

        if name == "bn2.weight": 
            Gamma_WeightBN[1] = param.data

        if name == "bn3.weight": 
            Gamma_WeightBN[2] = param.data

        if name == "bn4.weight": 
            Gamma_WeightBN[3] = param.data

        if name == "bn5.weight": 
            Gamma_WeightBN[4] = param.data

        if name == "bn6.weight": 
            Gamma_WeightBN[5] = param.data

        if name == "bn7.weight": 
            Gamma_WeightBN[6] = param.data

        if name == "bn8.weight": 
            Gamma_WeightBN[7] = param.data

        if name == "bn1.bias": 
            BetaBN[0] = param.data

        if name == "bn2.bias": 
            BetaBN[1] = param.data

        if name == "bn3.bias": 
            BetaBN[2] = param.data

        if name == "bn4.bias": 
            BetaBN[3] = param.data

        if name == "bn5.bias": 
            BetaBN[4] = param.data

        if name == "bn6.bias": 
            BetaBN[5] = param.data

        if name == "bn7.bias": 
            BetaBN[6] = param.data

        if name == "bn8.bias": 
            BetaBN[7] = param.data

        if name == "conv9.0.bias": 
            Bias = param.data
            
    Outputs = Weight, Bias, Gamma_WeightBN, BetaBN
    return Outputs


def update_weights(w, gw, b, gb, custom_model, custom_optimizer):
    """
    Update the weights of a custom model using the provided gradients and optimizer.

    Args:
        w  (Tensor): The weights of the model.
        gw (Tensor): The gradients of the weights.
        b  (Tensor): The biases of the model.
        gb (Tensor): The gradients of the biases.
        custom_model (nn.Module): The custom model to update.
        custom_optimizer (Optimizer): The optimizer to use for updating the weights.

    Returns:
        Dict[str, Tensor]: The updated state dictionary of the custom model.
    """
    custom_optimizer.zero_grad()
    
    # Update weight values
    custom_model = update_weight_values(custom_model, w, gw, b, gb)
    
    # Move model to GPU
    custom_model.to('cuda')

    # Perform optimization step
    custom_optimizer.step()

    # Return the updated state dictionary of the custom model
    return custom_model.state_dict()

def update_weights_FPGA(Inputs, gInputs, custom_model, custom_optimizer):
    """
    Update the weights of a custom model using the provided gradients and optimizer.

    Args:
        w  (Tensor): The weights of the model.
        gw (Tensor): The gradients of the weights.
        b  (Tensor): The biases of the model.
        gb (Tensor): The gradients of the biases.
        custom_model (nn.Module): The custom model to update.
        custom_optimizer (Optimizer): The optimizer to use for updating the weights.

    Returns:
        Dict[str, Tensor]: The updated state dictionary of the custom model.
    """
    custom_optimizer.zero_grad()
    
    # Update weight values
    custom_model = set_weight_values_FPGA(custom_model, Inputs, gInputs)

    # Perform optimization step
    custom_optimizer.step()

    # Get the updated state dictionary of the custom model
    Outputs = get_weight_values_FPGA(custom_model, Inputs, gInputs)
    
    return Outputs




def get_dataset_names(name):
    if name == 'voc07train':
        imdb_name = 'voc_2007_train'
        imdbval_name = 'voc_2007_train'
    elif name == 'voc07trainval':
        imdb_name = 'voc_2007_trainval'
        imdbval_name = 'voc_2007_trainval'
    elif name == 'voc0712trainval':
        imdb_name = 'voc_2007_trainval+voc_2012_trainval'
        imdbval_name ='voc_2007_test'
    else:
        raise NotImplementedError   
    return imdb_name, imdbval_name
