import torch
import torch.nn as nn
import torch.nn.functional as F

#my finding that, in the case of noisy labels, when we use soft target instead of hard target
#in soft cross entropy, the true probabilities are not just 1 or 0, it is a vlaue between 0 and 1, it uses soft targets instead of one-hot encoded targets.
class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
#The "forward" function takes in inputs and targets and calculates the soft cross-entropy 
# loss between the predicted outputs of the model (inputs) and the soft targets. The soft targets are
# obtained by applying the softmax function to the true targets and detaching them from the computation 
# graph to ignore it in backpropagation step. The log softmax probabilities are multiplied
# element-wise/product with the soft targets and then averaged across the batch dimension (dimension 0) and summed across 
# the class dimension (dimension 1) to obtain a single scalar value for the loss.
    def forward(self, inputs, targets):
        #inputs is predicted labels and targets is true labels,here pseudo labels
        log_probs = self.logsoftmax(inputs)
        loss = (-F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        return loss

#consider we have a batch of 3 inputs for 4 classes
# inputs=[[0.1,0.2,0.3,0.4],[0.4,0.1,0.2,0.3],[0.1,0.4,0.2,0.3]]
# targets=[]
# se=SoftEntropy()
# se.forward(inputs,)

class SoftLabelLoss(nn.Module):
    def __init__(self, alpha=1., T=20):
        super(SoftLabelLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p_logit, softlabel):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        p_logit = p_logit.view(p_logit.size(0), -1)
        log_probs = self.logsoftmax(p_logit / self.T)

        return self.T * self.alpha * self.kl_div(log_probs, softlabel)
