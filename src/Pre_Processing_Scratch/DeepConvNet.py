import torch


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu', dtype=torch.float64):

    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###########################################################################

        # The weight scale is sqrt(gain / fan_in),                                #
        # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
        # and fan_in = num_in_channels (= Din).                                   #
        # The output should be a tensor in the designated size, dtype, and device.#
        ###########################################################################
        weight_scale = gain / (Din)
        weight = torch.zeros(Din, Dout, dtype=dtype, device=device)
        weight += weight_scale * torch.randn(Din, Dout, dtype=dtype, device=device)

    else:
        ###########################################################################
        # The weight scale is sqrt(gain / fan_in),                                #
        # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
        # and fan_in = num_in_channels (= Din) * K * K                            #
        # The output should be a tensor in the designated size, dtype, and device.#
        ###########################################################################
        weight_scale = gain / (Din * K * K)
        weight = torch.zeros(Din, Dout, K, K, dtype=dtype, device=device)
        weight += weight_scale * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)

    return weight


class DeepConvNet(object):

    def __init__(self, input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 slowpool=True,
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float32, device='cpu'):

        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype
        self.slowpool = slowpool
        self.num_filters = num_filters
        self.save_pickle = False
        self.save_output = False

        if device == 'cuda':
            device = 'cuda:0'

        ############################################################################
        # TO DO: Initialize the parameters for the DeepConvNet. All weights,        #
        # biases, and batchnorm scale and shift parameters should be stored in the #
        # dictionary self.params.                                                  #
        #                                                                          #
        # Weights for conv and fully-connected layers should be initialized        #
        # according to weight_scale. Biases should be initialized to zero.         #
        # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
        # to ones and zeros respectively.                                          #
        ############################################################################
        # Replace "pass" statement with your code
        filter_size = 3
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pred_filters, H_out, W_out = input_dims
        HH = filter_size
        WW = filter_size
        # print('num_filters:', num_filters)
        for i, num_filter in enumerate(num_filters):
            H_out = int(1 + (H_out + 2 * conv_param['pad'] - HH) / conv_param['stride'])
            W_out = int(1 + (W_out + 2 * conv_param['pad'] - WW) / conv_param['stride'])
            if self.batchnorm:
                self.params['running_mean{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['running_var{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['gamma{}'.format(i)] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
                self.params['beta{}'.format(i)] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
            if i in max_pools:
                H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
                W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
            if weight_scale == 'kaiming':
                self.params['W{}'.format(i)] = kaiming_initializer(num_filter, pred_filters, K=filter_size, relu=True,
                                                                   device=device, dtype=dtype)
            else:
                self.params['W{}'.format(i)] = torch.zeros(num_filter, pred_filters, filter_size, filter_size,
                                                           dtype=dtype, device=device)
                self.params['W{}'.format(i)] += weight_scale * torch.randn(num_filter, pred_filters, filter_size,
                                                                           filter_size, dtype=dtype, device=device)
            pred_filters = num_filter

        i += 1

        if weight_scale == 'kaiming':
            self.params['W{}'.format(i)] = kaiming_initializer(125, 1024, K=1, relu=False, device=device, dtype=dtype)
        # else:
        #     self.params['W{}'.format(i)] = torch.zeros(num_filter*H_out*W_out, num_classes, dtype=dtype,device = device)
        #     self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter*H_out*W_out, num_classes, dtype=dtype,device= device)
        self.params['b{}'.format(i)] = torch.zeros(125, dtype=dtype, device=device)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_params object to each batch
        # normalization layer. You should pass self.bn_params[0] to the Forward pass
        # of the first batch normalization layer, self.bn_params[1] to the Forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
            for i, num_filter in enumerate(num_filters):
                self.bn_params[i]['running_mean'] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.bn_params[i]['running_var'] = torch.zeros(num_filter, dtype=dtype, device=device)

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 3  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        # assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def Training_Parameters(self):
        # print("Weight0: ",self.params['W0'])
        Weight = [self.params['W0'], self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4'],
                  self.params['W5'], self.params['W6'], self.params['W7'], self.params['W8']]
        Bias = self.params['b8']
        Gamma = [self.params['gamma0'], self.params['gamma1'], self.params['gamma2'], self.params['gamma3'],
                 self.params['gamma4'],
                 self.params['gamma5'], self.params['gamma6'], self.params['gamma7']]
        Beta = [self.params['beta0'], self.params['beta1'], self.params['beta2'], self.params['beta3'],
                self.params['beta4'], self.params['beta5'],
                self.params['beta6'], self.params['beta7']]
        Running_Mean = [self.bn_params[0]['running_mean'], self.bn_params[1]['running_mean'],
                        self.bn_params[2]['running_mean'],
                        self.bn_params[3]['running_mean'], self.bn_params[4]['running_mean'],
                        self.bn_params[5]['running_mean'],
                        self.bn_params[6]['running_mean'], self.bn_params[7]['running_mean']]
        Running_Var = [self.bn_params[0]['running_var'], self.bn_params[1]['running_var'],
                       self.bn_params[2]['running_var'],
                       self.bn_params[3]['running_var'], self.bn_params[4]['running_var'],
                       self.bn_params[5]['running_var'],
                       self.bn_params[6]['running_var'], self.bn_params[7]['running_var']]

        return Weight, Bias, Beta, Gamma, Running_Mean, Running_Var
