import torch
from torch.nn import ModuleList, Linear, Tanh, Sequential, BatchNorm1d, ReLU, Dropout

'''
    Dynamic Module:
    This module is for setting up the ODE dynamic. 
'''

torch.set_default_dtype(torch.double)

class odeModule(torch.nn.Module):
    def __init__(self, fOde, theta):
        super().__init__()
        self.f = fOde
        if (not torch.is_tensor(theta)):
            theta = torch.tensor(theta).double()
        self.theta = torch.nn.Parameter(theta)

    def forward(self, x):
        x = x.squeeze()
        if (x.ndimension()==1):
            x = x.reshape(1,-1)
        return (self.f(self.theta, x))

class nnModule(torch.nn.Module):
    # Neural Network with Constraint on Output
    def __init__(self, nodes, dp=0):
        super().__init__()
        # define network intermediate layers
        self.layers = ModuleList()
        for i in range(1,len(nodes)-1):
            self.layers.append(
                Sequential(
                    Linear(nodes[i-1], nodes[i], bias=True),
                    # BatchNorm1d(nodes[i]),
                    ReLU(),
                    # Tanh(),
                    Dropout(p=dp)
                ).double()
            )
        # define network last layers
        self.layers.append(
            Sequential(
                Linear(nodes[-2], nodes[-1], bias=True)
            ).double()
        )
        # define output standardization
        self.output_dim = nodes[-1]
        self.output_means = torch.zeros(nodes[-1])
        self.output_stds = torch.ones(nodes[-1])

    def update_output_layer(self, means, stds):
        self.output_means = means.detach()
        self.output_stds = stds.detach()

    def forward(self, x):
        if (x.ndimension()==1):
            x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        for i in range(self.output_dim):
            x[:,i] = self.output_means[i] + self.output_stds[i] * x[:,i]
        return (x)

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()
