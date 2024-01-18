import os
from Weight_Update_Algorithm.Test_with_train import *
import torch
    
class Shoaib_Code(object):
    
    def __init__(
        self,
        Weight_Dec      = []     ,
        Bias_Dec        = []     ,
        Beta_Dec        = []     ,
        Gamma_Dec       = []     ,
        Running_Mean_Dec= []     ,
        Running_Var_Dec = []     ,
        args            = []     ,
        pth_weights_path= ''     ,
        lr              = 0.0001  ,
        momentum        = 0.9  , 
        weight_decay    = 0.0005  ,
        model           = []     ,
        optim           = []     ,
        parent          = []
        ):
        
        self.Weight_Dec         = Weight_Dec      
        self.Bias_Dec           = Bias_Dec        
        self.Beta_Dec           = Beta_Dec        
        self.Gamma_Dec          = Gamma_Dec       
        self.Running_Mean_Dec   = Running_Mean_Dec
        self.Running_Var_Dec    = Running_Var_Dec 
        self.args               = args
        self.weight_path        = pth_weights_path
        self.model              = model
        self.custom_model       = model()
        self.init_model         = model()
        self.custom_optimizer   = optim.SGD(
            self.custom_model.parameters(), 
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            )
        self.Inputs = [ 
            Weight_Dec,
            Bias_Dec,
            Beta_Dec,
            Gamma_Dec,
            Running_Mean_Dec, 
            Running_Var_Dec
            ]
        self.parent = parent
        
    def get_weights(self, model):
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

    def get_bias(self, model):
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
        
    def get_weight_grads(self, model):
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

    def get_bias_grads(self, model):
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

    def get_w_and_b(self, model):
        w  = self.get_weights(model)
        gw = self.get_weight_grads(model)
        b  = self.get_bias(model)
        gb = self.get_bias_grads(model)
        return w, gw, b, gb

    def update_weight_values(self,  w, gw, b, gb):
        for name, param in self.custom_model.named_parameters():
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

    def set_weight_values_FPGA(self, Inputs, gInputs):
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
        
        for name, param in self.custom_model.named_parameters():
            if name == "conv1.weight": 
                param.data = Weight[0]
                param.grad = gWeight[0].cpu()

            elif name == "conv2.weight": 
                param.data = Weight[1]
                param.grad = gWeight[1].cpu()

            elif name == "conv3.weight": 
                param.data = Weight[2]
                param.grad = gWeight[2].cpu()

            elif name == "conv4.weight": 
                param.data = Weight[3]
                param.grad = gWeight[3].cpu()

            elif name == "conv5.weight": 
                param.data = Weight[4]
                param.grad = gWeight[4].cpu()

            elif name == "conv6.weight": 
                param.data = Weight[5]
                param.grad = gWeight[5].cpu()

            elif name == "conv7.weight": 
                param.data = Weight[6]
                param.grad = gWeight[6].cpu()

            elif name == "conv8.weight": 
                param.data = Weight[7]
                param.grad = gWeight[7].cpu()

            elif name == "conv9.0.weight": 
                param.data = Weight[8]
                param.grad = gWeight[8].cpu()

            elif name == "bn1.weight": 
                param.data = Gamma_WeightBN[0]
                try: param.grad = gGamma_WeightBN[0]
                except: param.grad = gGamma_WeightBN[0].view(-1).cpu()

            elif name == "bn2.weight": 
                param.data = Gamma_WeightBN[1]
                try: param.grad = gGamma_WeightBN[1]
                except: param.grad = gGamma_WeightBN[1].view(-1).cpu()

            elif name == "bn3.weight": 
                param.data = Gamma_WeightBN[2]
                try: param.grad = gGamma_WeightBN[2]
                except: param.grad = gGamma_WeightBN[2].view(-1).cpu()

            elif name == "bn4.weight": 
                param.data = Gamma_WeightBN[3]
                try: param.grad = gGamma_WeightBN[3]
                except: param.grad = gGamma_WeightBN[3].view(-1).cpu()

            elif name == "bn5.weight": 
                param.data = Gamma_WeightBN[4]
                try: param.grad = gGamma_WeightBN[4]
                except: param.grad = gGamma_WeightBN[4].view(-1).cpu()

            elif name == "bn6.weight": 
                param.data = Gamma_WeightBN[5]
                try: param.grad = gGamma_WeightBN[5]
                except: param.grad = gGamma_WeightBN[5].view(-1).cpu()

            elif name == "bn7.weight": 
                param.data = Gamma_WeightBN[6]
                try: param.grad = gGamma_WeightBN[6]
                except: param.grad = gGamma_WeightBN[6].view(-1).cpu()

            elif name == "bn8.weight": 
                param.data = Gamma_WeightBN[7]
                try: param.grad = gGamma_WeightBN[7]
                except: param.grad = gGamma_WeightBN[7].view(-1).cpu()

            elif name == "bn1.bias": 
                param.data = BetaBN[0]
                try: param.grad = gBetaBN[0]
                except: param.grad = gBetaBN[0].view(-1).cpu()

            elif name == "bn2.bias": 
                param.data = BetaBN[1]
                try: param.grad = gBetaBN[1]
                except: param.grad = gBetaBN[1].view(-1).cpu()

            elif name == "bn3.bias": 
                param.data = BetaBN[2]
                try: param.grad = gBetaBN[2]
                except: param.grad = gBetaBN[2].view(-1).cpu()

            elif name == "bn4.bias": 
                param.data = BetaBN[3]
                try: param.grad = gBetaBN[3]
                except: param.grad = gBetaBN[3].view(-1).cpu()

            elif name == "bn5.bias": 
                param.data = BetaBN[4]
                try: param.grad = gBetaBN[4]
                except: param.grad = gBetaBN[4].view(-1).cpu()

            elif name == "bn6.bias": 
                param.data = BetaBN[5]
                try: param.grad = gBetaBN[5]
                except: param.grad = gBetaBN[5].view(-1).cpu()

            elif name == "bn7.bias": 
                param.data = BetaBN[6]
                try: param.grad = gBetaBN[6]
                except: param.grad = gBetaBN[6].view(-1).cpu()

            elif name == "bn8.bias": 
                param.data = BetaBN[7]
                try: param.grad = gBetaBN[7]
                except: param.grad = gBetaBN[7].view(-1).cpu()

            elif name == "conv9.0.bias": 
                param.data = Bias
                param.grad = gBias.cpu()
            
            elif name == "bn1.running_mean": pass        
            elif name == "bn1.running_var": pass        
            elif name == "bn2.running_mean": pass        
            elif name == "bn2.running_var": pass        
            elif name == "bn3.running_mean": pass        
            elif name == "bn3.running_var": pass        
            elif name == "bn4.running_mean": pass        
            elif name == "bn4.running_var": pass
            elif name == "bn5.running_mean": pass        
            elif name == "bn5.running_var": pass
            elif name == "bn6.running_mean": pass        
            elif name == "bn6.running_var": pass
            elif name == "bn7.running_mean": pass        
            elif name == "bn7.running_var": pass
            elif name == "bn8.running_mean": pass        
            elif name == "bn8.running_var": pass
            else: print(name)
            
                
        return self.custom_model

    def get_weight_values_FPGA(self, Inputs):
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
                
        for name, param in self.custom_model.named_parameters():
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

    def update_weights(self, w, gw, b, gb):
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
        self.custom_optimizer.zero_grad()
        
        # Update weight values
        self.custom_model = self.update_weight_values(w, gw, b, gb)
        
        # Move model to GPU
        self.custom_model.to('cuda')

        # Perform optimization step
        self.custom_optimizer.step()

        # Return the updated state dictionary of the custom model
        return self.custom_model.state_dict()

    def update_weights_FPGA(self, Inputs=[], gInputs=[]):
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
        
        self.custom_optimizer.zero_grad()
        
        # Update weight values
        self.custom_model = self.set_weight_values_FPGA(Inputs, gInputs)

        # Perform optimization step
        self.custom_optimizer.step()

        # Get the updated state dictionary of the custom model
        Outputs = self.get_weight_values_FPGA(Inputs)
        return Outputs, self.custom_model

    def update_weights_Pytorch_Python(self, Inputs=[], gInputs=[]):
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
        self.custom_optimizer.zero_grad()
        
        # Update weight values
        self.custom_model = self.set_weight_values_FPGA(Inputs, gInputs)

        # Perform optimization step
        self.custom_optimizer.step()

        # Get the updated state dictionary of the custom model
        Outputs = self.get_weight_values_FPGA(Inputs)
        return Outputs, self.custom_model

    def load_weights(self):
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

        if self.weight_path:
            if self.parent==[]: print(f'--> Loading weights from:\n{self.weight_path}\n')
            else: self.parent.Show_Text(f'--> Loading weights from:\n{self.weight_path}\n')
            
            self.pretrained_checkpoint = torch.load(self.weight_path,map_location='cpu')
            loaded_model = torch.load(self.weight_path,map_location='cpu')['model']
            self.custom_model.load_state_dict(loaded_model)
            
        else:
            if self.parent==[]: print(f'--> Starting training from scratch.')
            else: self.parent.Show_Text(f'--> Starting training from scratch.')
            
        # self.pretrained_checkpoint = torch.load(self.weight_path,map_location='cpu')
        # loaded_model = torch.load(self.weight_path,map_location='cpu')['model']
        # self.custom_model.load_state_dict(loaded_model)
        Weight,   Bias,  Gamma_WeightBN,  BetaBN, Running_Mean_Dec, Running_Var_Dec = self.Inputs    # Gamma is weight for BN, Beta is Bias for BN
        _model_state_dict = self.custom_model.state_dict()        
        
        _delete_after_copy=False
        for name in _model_state_dict:
            if name == "conv1.weight": 
                Weight[0] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv2.weight": 
                Weight[1] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv3.weight": 
                Weight[2] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv4.weight": 
                Weight[3] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv5.weight": 
                Weight[4] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv6.weight": 
                Weight[5] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv7.weight": 
                Weight[6] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv8.weight": 
                Weight[7] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv9.0.weight": 
                Weight[8] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn1.weight": 
                Gamma_WeightBN[0] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn2.weight": 
                Gamma_WeightBN[1] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn3.weight": 
                Gamma_WeightBN[2] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn4.weight": 
                Gamma_WeightBN[3] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn5.weight": 
                Gamma_WeightBN[4] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn6.weight": 
                Gamma_WeightBN[5] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn7.weight": 
                Gamma_WeightBN[6] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn8.weight": 
                Gamma_WeightBN[7] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn1.bias": 
                BetaBN[0] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn2.bias": 
                BetaBN[1] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn3.bias": 
                BetaBN[2] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn4.bias": 
                BetaBN[3] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn5.bias": 
                BetaBN[4] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn6.bias": 
                BetaBN[5] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn7.bias": 
                BetaBN[6] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn8.bias": 
                BetaBN[7] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "conv9.0.bias": 
                Bias = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn1.running_mean": 
                Running_Mean_Dec[0] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn1.running_var": 
                Running_Var_Dec[0] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn2.running_mean": 
                Running_Mean_Dec[1] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn2.running_var": 
                Running_Var_Dec[1] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn3.running_mean": 
                Running_Mean_Dec[2] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn3.running_var": 
                Running_Var_Dec[2] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn4.running_mean": 
                Running_Mean_Dec[3] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn4.running_var": 
                Running_Var_Dec[3] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn5.running_mean": 
                Running_Mean_Dec[4] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn5.running_var": 
                Running_Var_Dec[4] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn6.running_mean": 
                Running_Mean_Dec[5] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn6.running_var": 
                Running_Var_Dec[5] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn7.running_mean": 
                Running_Mean_Dec[6] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn7.running_var": 
                Running_Var_Dec[6] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

            if name == "bn8.running_mean": 
                Running_Mean_Dec[7] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]
        
            if name == "bn8.running_var": 
                Running_Var_Dec[7] = _model_state_dict[name]
                if _delete_after_copy: del _model_state_dict[name]

        
        Outputs = Weight, Bias, Gamma_WeightBN, BetaBN, Running_Mean_Dec, Running_Var_Dec
        return Outputs

    def set_weights(self, Inputs_with_running=[]):
        Weight,   Bias,  Gamma_WeightBN,  BetaBN, RunningMean, RunningVar = Inputs_with_running    # Gamma is weight for BN, Beta is Bias for BN
        
        for name, param in self.custom_model.named_parameters():
            # Conv - Weights
            if   name == "conv1.weight"     : self.custom_model.conv1.weight.data      = Weight[0]
            elif name == "conv2.weight"     : self.custom_model.conv2.weight.data      = Weight[1]
            elif name == "conv3.weight"     : self.custom_model.conv3.weight.data      = Weight[2]
            elif name == "conv4.weight"     : self.custom_model.conv4.weight.data      = Weight[3]
            elif name == "conv5.weight"     : self.custom_model.conv5.weight.data      = Weight[4]
            elif name == "conv6.weight"     : self.custom_model.conv6.weight.data      = Weight[5]
            elif name == "conv7.weight"     : self.custom_model.conv7.weight.data      = Weight[6]
            elif name == "conv8.weight"     : self.custom_model.conv8.weight.data      = Weight[7]
            elif name == "conv9.0.weight"   : self.custom_model.conv9[0].weight.data   = Weight[8]
            elif name == "conv9.0.bias"     : self.custom_model.conv9[0].bias.data     = Bias      
            elif name == "bn1.weight"       : self.custom_model.bn1.weight.data        = Gamma_WeightBN[0]
            elif name == "bn2.weight"       : self.custom_model.bn2.weight.data        = Gamma_WeightBN[1]
            elif name == "bn3.weight"       : self.custom_model.bn3.weight.data        = Gamma_WeightBN[2]
            elif name == "bn4.weight"       : self.custom_model.bn4.weight.data        = Gamma_WeightBN[3]
            elif name == "bn5.weight"       : self.custom_model.bn5.weight.data        = Gamma_WeightBN[4]
            elif name == "bn6.weight"       : self.custom_model.bn6.weight.data        = Gamma_WeightBN[5]
            elif name == "bn7.weight"       : self.custom_model.bn7.weight.data        = Gamma_WeightBN[6]
            elif name == "bn8.weight"       : self.custom_model.bn8.weight.data        = Gamma_WeightBN[7]
            elif name == "bn1.bias"         : self.custom_model.bn1.bias.data          = BetaBN[0]
            elif name == "bn2.bias"         : self.custom_model.bn2.bias.data          = BetaBN[1]
            elif name == "bn3.bias"         : self.custom_model.bn3.bias.data          = BetaBN[2]
            elif name == "bn4.bias"         : self.custom_model.bn4.bias.data          = BetaBN[3]
            elif name == "bn5.bias"         : self.custom_model.bn5.bias.data          = BetaBN[4]
            elif name == "bn6.bias"         : self.custom_model.bn6.bias.data          = BetaBN[5]
            elif name == "bn7.bias"         : self.custom_model.bn7.bias.data          = BetaBN[6]
            elif name == "bn8.bias"         : self.custom_model.bn8.bias.data          = BetaBN[7]      
            elif name == "bn1.running_mean" : self.custom_model.bn1.running_mean.data  = RunningMean[0]
            elif name == "bn2.running_mean" : self.custom_model.bn2.running_mean.data  = RunningMean[1]
            elif name == "bn3.running_mean" : self.custom_model.bn3.running_mean.data  = RunningMean[2]
            elif name == "bn4.running_mean" : self.custom_model.bn4.running_mean.data  = RunningMean[3]
            elif name == "bn5.running_mean" : self.custom_model.bn5.running_mean.data  = RunningMean[4]
            elif name == "bn6.running_mean" : self.custom_model.bn6.running_mean.data  = RunningMean[5]
            elif name == "bn7.running_mean" : self.custom_model.bn7.running_mean.data  = RunningMean[6]
            elif name == "bn8.running_mean" : self.custom_model.bn8.running_mean.data  = RunningMean[7]
            elif name == "bn1.running_var"  : self.custom_model.bn1.running_var.data   = RunningVar[0]              
            elif name == "bn2.running_var"  : self.custom_model.bn2.running_var.data   = RunningVar[1]      
            elif name == "bn3.running_var"  : self.custom_model.bn3.running_var.data   = RunningVar[2]      
            elif name == "bn4.running_var"  : self.custom_model.bn4.running_var.data   = RunningVar[3]      
            elif name == "bn5.running_var"  : self.custom_model.bn5.running_var.data   = RunningVar[4]      
            elif name == "bn6.running_var"  : self.custom_model.bn6.running_var.data   = RunningVar[5]      
            elif name == "bn7.running_var"  : self.custom_model.bn7.running_var.data   = RunningVar[6]       
            elif name == "bn7.running_var"  : self.custom_model.bn8.running_var.data   = RunningVar[7]      
            else: print(name)
        return self.custom_model
    
    def cal_mAP(self, Inputs_with_running=[]):
        self.set_weights(Inputs_with_running=Inputs_with_running)
        save_name_temp = self.args.output_dir
        # save_name_temp = os.path.join(self.args.output_dir, 'temp')
        map = test_for_train(save_name_temp, self.custom_model, self.args)
        return map
    
    def get_dataset_names(self, name):
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
    