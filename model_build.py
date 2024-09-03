import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# NN class
class NN(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        super(NN, self).__init__()
        self.policy0 = nn.Sequential(
            nn.Linear(input_size, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, output_size)
        )
        self.policy1 = nn.Sequential(
            nn.Linear(input_size, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, output_size)
        )
        self.value = nn.Sequential(
            nn.Linear(input_size, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, 1)
        )

    def forward(self, x1, x2, xs):
        logit0 = self.policy0(x1)
        logit1 = self.policy1(x2)
        a_dist0 = F.softmax(logit0, dim=-1)
        a_dist1 = F.softmax(logit1, dim=-1)
        value = self.value(xs)
        return a_dist0, a_dist1, value

    def log_a_dists(self, x1, x2):
        logit0 = self.policy0(x1)
        logit1 = self.policy1(x2)
        log_a_dist0 = F.log_softmax(logit0, dim=-1)
        log_a_dist1 = F.log_softmax(logit1, dim=-1)
        return log_a_dist0, log_a_dist1

# Shared optimizer class
# reference: https://github.com/MorvanZhou/pytorch-A3C/blob/master/shared_adam.py
class SharedAdam(optim.Adam):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        super(SharedAdam, self).__init__(params, lr, betas, eps)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()