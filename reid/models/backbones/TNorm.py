import torch
import torch.nn as nn
import numpy as np

#Transform Normalization
class TNorm(nn.Module):
    # domain_number is 6 for Market1501
    # num_features is the number of feature maps
    def __init__(self, num_features, domain_number, using_moving_average=True, eps=1e-5, momentum=0.9):
        super(TNorm, self).__init__()
        self.num_features = num_features#the number of features in the input tensor
        self.eps = eps# a small value added to the denominator for numerical stability
        self.momentum = momentum# the momentum factor used for moving average updates
        self.domain_number = domain_number#the number of domains in the dataset
        self.using_moving_average = using_moving_average#a boolean indicating whether to use moving averages to compute running mean and variance
        self.register_buffer('running_mean', torch.zeros((domain_number, num_features)))
        self.register_buffer('running_var', torch.ones((domain_number, num_features)))
        self.register_buffer('num_batches_tracked', torch.zeros(domain_number, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):#: resets the running mean and variance, and the number of batches tracked to zero
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def _check_input_dim(self, input):#checks the input tensor's dimensions to ensure it is 4-dimensional.
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, domain_index=None, convert=False, selected_domain=None):
        #selected_domain is a parameter is an optional argument that specifies the domain to which the input tensor should be converted.
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        temp = var_in + mean_in ** 2

        if self.training:
            
            if convert:#if convert is True, the module first normalizes x using the mean and variance of domain_index and then converts it to the mean and variance of selected_domain
                assert selected_domain is not None
                if domain_index is not None and type(domain_index) != int:
                    mean_bn = self.running_mean[domain_index, :].view(N, C, 1)
                    var_bn = self.running_var[domain_index, :].view(N, C, 1)
                    x_after_in = (x - mean_bn) / (var_bn + self.eps).sqrt()
                else:
                    mean_bn = mean_in.mean(0, keepdim=True)
                    var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                    sig = (var_bn + self.eps).sqrt()
                    mean_bn, sig = mean_bn.detach(), sig.detach()
                    x_after_in = (x - mean_bn) / sig

                convert_mean = self.running_mean[selected_domain, :].view(N, C, 1)  # N * C
                convert_var = self.running_var[selected_domain, :].view(N, C, 1)  # N * C
                x = x_after_in * (convert_var + self.eps).sqrt() + convert_mean

                if domain_index is not None and type(domain_index) == int:
                    if self.using_moving_average:
                        self.running_mean[domain_index].mul_(self.momentum)
                        self.running_mean[domain_index].add_((1 - self.momentum) * mean_bn.squeeze().data)
                        self.running_var[domain_index].mul_(self.momentum)
                        self.running_var[domain_index].add_((1 - self.momentum) * var_bn.squeeze().data)
                    else:
                        self.num_batches_tracked[domain_index] += 1
                        exponential_average_factor = 1 - 1.0 / self.num_batches_tracked[domain_index]
                        self.running_mean[domain_index].mul_(exponential_average_factor)
                        self.running_mean[domain_index].add_(
                            (1 - exponential_average_factor) * mean_bn.squeeze().data)
                        self.running_var[domain_index].mul_(exponential_average_factor)
                        self.running_var[domain_index].add_(
                            (1 - exponential_average_factor) * var_bn.squeeze().data)
            else:##If convert is False, the module updates the running mean and variance for domain_index and normalizes x using the updated mean and variance.
                if domain_index is not None:
                    mean_bn = mean_in.mean(0, keepdim=True)
                    var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                    if self.using_moving_average:
                        self.running_mean[domain_index].mul_(self.momentum)
                        self.running_mean[domain_index].add_((1 - self.momentum) * mean_bn.squeeze().data)
                        self.running_var[domain_index].mul_(self.momentum)
                        self.running_var[domain_index].add_((1 - self.momentum) * var_bn.squeeze().data)
                    else:
                        self.num_batches_tracked[domain_index] += 1
                        exponential_average_factor = 1 - 1.0 / self.num_batches_tracked[domain_index]
                        self.running_mean[domain_index].mul_(exponential_average_factor)
                        self.running_mean[domain_index].add_(
                            (1 - exponential_average_factor) * mean_bn.squeeze().data)
                        self.running_var[domain_index].mul_(exponential_average_factor)
                        self.running_var[domain_index].add_(
                            (1 - exponential_average_factor) * var_bn.squeeze().data)
        else:#durinng inference
            domain_mean_bn = torch.autograd.Variable(self.running_mean)
            domain_var_bn = torch.autograd.Variable(self.running_var)
            if convert:# if convert is True, the module normalizes x using the running mean and variance of domain_index and then converts it to the mean and variance of selected_domain.
                assert domain_index is not None

                x_after_in = (x - domain_mean_bn[domain_index[0], :].view(N, C, 1)) / (domain_var_bn[domain_index[0], :].view(N, C, 1) + self.eps).sqrt()
                
                convert_mean = domain_mean_bn[domain_index[1], :].view(1, C, 1)  # N * C
                convert_var = domain_var_bn[domain_index[1], :].view(1, C, 1)  # N * C
                x = x_after_in * (convert_var + self.eps).sqrt() + convert_mean
            else:#If convert is False, the module normalizes x using the running mean and variance of domain_index.
                pass
        x = x.view(N, C, H, W)

        return x
