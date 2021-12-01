import math

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from gym import spaces

from common.distributions import *
from common.util import Flatten

#hyper-parameters

def classic_control(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, spaces.Box):
        dist = DiagGaussian
        policy_dim = env.action_space.shape[0] * 2
    elif isinstance(env.action_space, spaces.Discrete):
        dist = Categorical
        policy_dim = env.action_space.n
    else:
        raise ValueError
    network = MLP(in_dim, policy_dim)
    optimizer = Adam(network.parameters(), 3e-4, eps=1e-5)
    params = dict(
        dist=dist,
        network=network,
        optimizer=optimizer,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=90,
        ent_coef=0,
        vf_coef=0.5,
        gae_lam=0.95,
        nminibatches=90,
        opt_iter=4,
        cliprange=0.2,
        ob_scale=1
    )
    params.update(kwargs)
    return params


#neural network
class MLP(nn.Module):
    def __init__(self, in_dim, policy_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()


        )
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        self.policy = nn.Linear(1024, policy_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)


        self.value = nn.Linear(1024, 1)
        nn.init.orthogonal_(self.value.weight, 1)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)

        # return self.policy(F.relu(self.policy2(F.relu(self.policy1(latent))))), self.value(F.relu(self.value2(F.relu(self.value1(latent)))))
        return self.policy(latent),self.value(latent)
