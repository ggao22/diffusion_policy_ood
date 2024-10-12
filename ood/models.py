import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from math import pi

import sys
sys.path.append('ood/')


class EquivariantMap(nn.Module):
    def __init__(self, input_size=(180,256), input_ch=3, output_size=3):
        super(EquivariantMap,self).__init__()
        self.input_size = input_size
        self.input_ch = input_ch
        self.output_size = output_size

        #encoder
        self.c1 = nn.Conv2d(input_ch,64,kernel_size=4,padding=1,stride=2)
        self.c2 = nn.Conv2d(64,64,kernel_size=4,padding=1,stride=2)
        self.c3 = nn.Conv2d(64,64,kernel_size=4,padding=1,stride=2)
        self.c4 = nn.Conv2d(64,64,kernel_size=4,padding=1,stride=2)
        self.c5 = nn.Conv2d(64,8,kernel_size=4,padding=1,stride=2)
        self.fc1 = nn.Linear(np.prod([8,3,3]), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,self.output_size)

    def forward(self,input):
        input = input.view(-1,self.input_ch,self.input_size[0],self.input_size[1])
        x = F.relu(self.c1(input))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))

        # print(x.shape)
        x = x.view([x.size()[0], -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def action_loss(self, z, z_prime, action):
        return torch.sum(torch.norm(z_prime - z - action, dim=1)**2)

    def position_loss(self, z, pos):
        return torch.sum(torch.norm(z - pos, dim=1)**2)


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)




from torch import tensor, log, exp, flatten
from torch.distributions import MultivariateNormal as MVN
from utils import normalize_data, unnormalize_data, unnormalize_gradient


class RecoveryPolicy(nn.Module):
    def __init__(self, Encoder, GMM_PARAMS, latent_stats, eps=-0.5, tau=0.5, eta=1.0):
        super(RecoveryPolicy,self).__init__()
        self.Encoder = Encoder
        self.GMM_PARAMS = GMM_PARAMS
        self.latent_stats = latent_stats

        self.eps = eps
        self.tau = tau
        self.eta = eta

    def pdfs(self, x):
        densities = []
        x.retain_grad()
        for i in range(len(self.GMM_PARAMS)):
            gmm_params = self.GMM_PARAMS[str(i)][()]
            MVNs = [MVN(mu, sigma) for (mu, sigma) in zip(tensor(gmm_params["means_"]), tensor(gmm_params["covariances_"]))]
            pdf = sum([pi * exp(MVN.log_prob(x[2*i:2*(i+1)])) for (pi, MVN) in zip(tensor(gmm_params['weights_']), MVNs)])
            pdf.backward()
            pdf = float(pdf.detach().cpu())
            densities.append([pdf]*2)
        return np.array(densities).flatten()
    
    def _encoder(self,obs):
        with torch.no_grad():
            z = self.Encoder(obs)
            z = normalize_data(flatten(z).cpu().numpy(), self.latent_stats)
        return z

    def forward(self,obs):
        z = self._encoder(obs)
        densities, grad = self._get_grad(z)
        return densities, self.eta * grad
    
    def _get_grad(self, z):
        z = tensor(z, requires_grad=True)
        densities = self.pdfs(z)
        densities_norm = 1/(1+np.exp(-(densities+self.eps)/self.tau)) # parameterized sigmoid

        grad = unnormalize_gradient(z.grad.detach().cpu().numpy(), self.latent_stats)
        negexp_grad = np.copy(grad)
        for i in range(len(self.GMM_PARAMS)):
            grad_kp = grad[2*i:2*(i+1)]
            grad_kp_norm = grad_kp/np.linalg.norm(grad_kp)
            mag = np.exp((-np.linalg.norm(grad_kp)+5500)/1100)
            negexp_grad[2*i:2*(i+1)] = grad_kp_norm*mag

        return densities_norm, negexp_grad



class GMMGradient(nn.Module):
    def __init__(self, GMM_PARAMS):
        super(GMMGradient,self).__init__()
        self.GMM_PARAMS = GMM_PARAMS

        # set parameters for negative exponential
        self.eps = 5500
        self.lmb = 1100
    
    def param_negative_exponential(self, x):
        return np.exp((-x+self.eps)/self.lmb)

    def pdfs(self, x):
        densities = []
        x.retain_grad()
        for i in range(len(self.GMM_PARAMS)):
            gmm_params = self.GMM_PARAMS[str(i)][()]
            MVNs = [MVN(mu, sigma) for (mu, sigma) in zip(tensor(gmm_params["means_"]), tensor(gmm_params["covariances_"]))]
            pdf = sum([pi * exp(MVN.log_prob(x[i])) for (pi, MVN) in zip(tensor(gmm_params['weights_']), MVNs)])
            pdf.backward()
            pdf = float(pdf.detach().cpu())
            densities.append([pdf]*2)
        return np.array(densities).flatten()
    
    def forward(self,kps):
        densities = []
        grad_vectors = []
        for i in range(kps.shape[0]):
            dens, grad = self._get_grad(kps[i])
            densities.append(dens)
            grad_vectors.append(grad)
        grad_vectors = np.array(grad_vectors)
        densities = np.array(densities)
        return densities, grad_vectors
    
    def _get_grad(self, z):
        z = tensor(z, requires_grad=True)
        densities = self.pdfs(z)

        grad = z.grad.detach().cpu().numpy()
        negexp_grad = np.copy(grad)
        for i in range(len(self.GMM_PARAMS)):
            grad_kp = grad[i]
            grad_kp_norm = grad_kp/np.linalg.norm(grad_kp)
            mag = self.param_negative_exponential(np.linalg.norm(grad_kp))
            negexp_grad[i] = grad_kp_norm*mag

        return densities, negexp_grad.flatten()
    

