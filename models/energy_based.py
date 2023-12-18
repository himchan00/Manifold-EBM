import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyBasedModel(nn.Module):
    def __init__(self, net, bound = 'spherical', gamma=1.0,  temperature = 1.0, step_size=0.1, sample_step=50,
                 langevin_clip_grad=None, noise_scale = None, buffer_size=10000, replay_ratio=0.95):
        super().__init__()
        self.net = net
        self.bound = bound
        self.gamma = gamma
        self.temperature = temperature
        self.step_size = step_size
        self.sample_step = sample_step
        self.langevin_clip_grad = langevin_clip_grad
        if noise_scale is None:
            self.noise_scale = np.sqrt(step_size * 2)
        else:
            self.noise_scale = noise_scale
        if step_size is None:
            self.stepsize = (noise_scale ** 2) / 2
        else:
            self.stepsize = step_size
        self.buffer_size = buffer_size
        self.buffer = SampleBuffer(max_samples= buffer_size, replay_ratio=replay_ratio, bound=bound)
        # self.buffer = SampleBuffer_HMC(replay_ratio= replay_ratio, bound=bound)
        self.replay_ratio = replay_ratio
        self.replay = True if self.replay_ratio > 0 else False

    def forward(self, x):
        return self.net(x).view(-1)

    def predict(self, x):
        return self(x)

    def validation_step(self, x, y=None):
        with torch.no_grad():
            pos_e = self(x)

        return {'loss': pos_e.mean(),
                'predict': pos_e,
                }

    def energy_loss(self, x):
        neg_x = self.sample(shape=x.shape, sample_step= self.sample_step, device=x.device, replay=self.replay)
        pos_e = self(x)
        neg_e = self(neg_x)

        ebm_loss = pos_e.mean() - neg_e.mean()
        reg_loss = (pos_e ** 2).mean() + (neg_e ** 2).mean()

        loss = (ebm_loss + self.gamma * reg_loss).mean()/self.temperature
        return loss, ebm_loss, pos_e, neg_e, reg_loss, neg_x


    def sample(self, shape, sample_step, device, replay=True):
        # initialize
        x0 = self.buffer.sample(shape, device, replay = replay) # for HMC
        # x0 = self.buffer.sample(shape, device, replay=replay) # for langevin
        # run langevin
        sample_x = sample_langevin(x = x0, model = self, stepsize = self.step_size, n_steps = sample_step,
                                    temperature= self.temperature, noise_scale=self.noise_scale, 
                                    spherical=True, clip_grad=self.langevin_clip_grad)
        
        # run hmc
        # sample_x = sample_geodesic_hmc(x = x0, model = self, stepsize = self.step_size, n_steps = sample_step,
        #                                  temperature= self.temperature, intermediate_samples=False)
        # push samples
        if replay:
            self.buffer.push(sample_x)
        return sample_x

    def sample_mcmc_trajectory(self, shape, sample_step, device, replay=True):
        # initialize
        x0 = self.buffer.sample(shape, device, replay = replay) # for HMC
        # x0 = self.buffer.sample(shape, device, replay=replay) # for langevin
        # # run langevin
        l_samples, l_dynamics, l_drift, l_diffusion = sample_langevin(x = x0, model = self, stepsize = self.step_size, n_steps = sample_step,
                                                                        temperature= self.temperature, noise_scale=self.noise_scale, 
                                                                        spherical=True, clip_grad=self.langevin_clip_grad, intermediate_samples=True)
        # # run hmc
        # l_samples, accept = sample_geodesic_hmc(x = x0, model = self, stepsize = self.step_size, n_steps = sample_step,
        #                                             temperature= self.temperature, intermediate_samples=True)
        return l_samples, l_dynamics, l_drift, l_diffusion
        # return l_samples, accept

# class SampleBuffer_HMC:
#     def __init__(self, replay_ratio, bound=None):
#         self.previous_samples = None
#         self.replay_ratio = replay_ratio
#         self.bound = bound
    
#     def push(self, samples):
#         self.previous_samples = samples
    
#     def sample(self, shape, device, replay):
#         if self.previous_samples == None:
#             return self.random(shape, device)
#         n_random = (np.random.rand(shape[0]) > self.replay_ratio).sum()
#         previous_samples = self.previous_samples.detach().clone().to(device)
#         n_replay = shape[0] - n_random
#         if n_replay > previous_samples.shape[0]:
#             n_random += n_replay - previous_samples.shape[0]
#             n_replay = previous_samples.shape[0]
        
#         if n_random > 0:
#             random_sample = self.random((n_random,) + shape[1:], device)

#             return torch.cat([previous_samples, random_sample])
#         else:
#             indices = torch.randperm(previous_samples.shape[0])[:shape[0]]
#             return previous_samples[indices].to(device)


    # def random(self, shape, device):
    #     if self.bound is None:
    #         r = torch.randn(*shape, dtype=torch.float).to(device)

    #     elif self.bound == 'spherical':
    #         r = torch.randn(*shape, dtype=torch.float).to(device)
    #         norm = r.view(len(r), -1).norm(dim=-1)
    #         if len(shape) == 4:
    #             r = r / norm[:, None, None, None]
    #         elif len(shape) == 2:
    #             r = r / norm[:, None]
    #         else:
    #             raise NotImplementedError

    #     elif len(self.bound) == 2:
    #         r = torch.rand(*shape, dtype=torch.float).to(device)
    #         r = r * (self.bound[1] - self.bound[0]) + self.bound[0]
    #     return r

class SampleBuffer:
    def __init__(self, max_samples=10000, replay_ratio=0.9, bound=None):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio
        self.bound = bound

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

    def sample(self, shape, device, replay=False, init_samples=None):
        if len(self.buffer) < 1 or not replay:  # empty buffer
            return self.random(shape, device, init_samples=init_samples)

        n_replay = (np.random.rand(shape[0]) < self.replay_ratio).sum()

        replay_sample = self.get(n_replay).to(device)
        n_random = shape[0] - n_replay
        if n_random > 0:
            random_sample = self.random((n_random,) + shape[1:], device, init_samples=init_samples)
            return torch.cat([replay_sample, random_sample])
        else:
            return replay_sample

    def random(self, shape, device, init_samples=None):
        if init_samples is not None:
            if init_samples.shape[0] >= shape[0]:
                indices = torch.randperm(init_samples.shape[0])[:shape[0]]
                r = init_samples[indices].to(device)
            else:
                NotImplementedError("Less init samples than required")
        else:
            if self.bound is None:
                r = torch.randn(*shape, dtype=torch.float).to(device)

            elif self.bound == 'spherical':
                r = torch.randn(*shape, dtype=torch.float).to(device)
                norm = r.view(len(r), -1).norm(dim=-1)
                if len(shape) == 4:
                    r = r / norm[:, None, None, None]
                elif len(shape) == 2:
                    r = r / norm[:, None]
                else:
                    raise NotImplementedError

            elif len(self.bound) == 2:
                r = torch.rand(*shape, dtype=torch.float).to(device)
                r = r * (self.bound[1] - self.bound[0]) + self.bound[0]
        return r

import torch.autograd as autograd

def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x

def sample_geodesic_hmc(x, model, stepsize, n_steps, temperature, intermediate_samples=False):
    """Draw samples using HMC"""
    bs, n = x.shape[0], x.shape[1]

    l_samples = []
    x.requires_grad = True
    x_ = x.detach()
    X = x_.unsqueeze(2).to(x)
    X_t = X.permute(0, 2, 1).to(x)
    P = torch.eye(n).repeat(bs, 1, 1).to(x) - torch.bmm(X, X_t).to(x)
    v = torch.randn_like(x)
    # project v to tangent space
    v = torch.bmm(P, v.unsqueeze(2)).squeeze(2) # (bs, n)
    # initial hamiltonian
    H = model(x).squeeze()/temperature + 0.5 * (v ** 2).sum(dim=1) # (bs,)
    # x_init = x.detach().clone()
    for i_step in range(n_steps):
        l_samples.append(x.detach().to('cpu'))
        # leapfrog
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = - stepsize * grad / (2*temperature)
        v = v + dynamics
        # project v to tangent space
        x_ = x.detach()
        X = x_.unsqueeze(2).to(x)
        X_t = X.permute(0, 2, 1).to(x)
        P = torch.eye(n).repeat(bs, 1, 1).to(x) - torch.bmm(X, X_t).to(x)
        v = torch.bmm(P, v.unsqueeze(2)).squeeze(2)

        v_norm = v.norm(dim=1, keepdim=True)
        x = x * torch.cos(v_norm*stepsize) + v/v_norm * torch.sin(v_norm*stepsize)
        v = v * torch.cos(v_norm*stepsize) - x * v_norm * torch.sin(v_norm*stepsize)

        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = - stepsize * grad / (2*temperature)
        v = v + dynamics
        # project v to tangent space
        x_ = x.detach()
        X = x_.unsqueeze(2).to(x)
        X_t = X.permute(0, 2, 1).to(x)
        P = torch.eye(n).repeat(bs, 1, 1).to(x) - torch.bmm(X, X_t).to(x)
        v = torch.bmm(P, v.unsqueeze(2)).squeeze(2)

    # final hamiltonian
    H_new = model(x).squeeze()/temperature + 0.5 * (v ** 2).sum(dim=1) # (bs,)
    # accept-reject
    accept = (torch.rand_like(H) < torch.exp(H - H_new)).to(torch.float)
    x_random = torch.randn_like(x)
    x_random = x_random / x_random.norm(dim=1, keepdim=True)
    x_new = x * accept[:, None] + x_random * (1 - accept[:, None])
    x = x_new.detach().clone()
    if intermediate_samples:
        return l_samples, accept
    else:
        return x.detach()

def sample_langevin(x, model, stepsize, n_steps, temperature, noise_scale=None, intermediate_samples=False,
                    clip_x=None, clip_grad=None, reject_boundary=False, noise_anneal=None,
                    spherical=False):
    """Draw samples using Langevin dynamics
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    clip_x : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not ((stepsize is None) and (noise_scale is None)), 'stepsize and noise_scale cannot be None at the same time'
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale ** 2) / 2
    noise_scale_ = noise_scale

    l_samples = []
    l_dynamics = []; l_drift = []; l_diffusion = []
    x.requires_grad = True
    bs, n = x.shape[0], x.shape[1]
    for i_step in range(n_steps):
        l_samples.append(x.detach().to('cpu'))
        noise = torch.randn_like(x) * noise_scale_
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        if clip_grad is not None:
            grad = clip_vector_norm(grad, max_norm=clip_grad)
        if spherical:
            x_ = x.detach()
            X = x_.unsqueeze(2).to(x)
            X_t = X.permute(0, 2, 1).to(x)
            P = torch.eye(n).repeat(bs, 1, 1).to(x) - torch.bmm(X, X_t).to(x)
            dynamics = - stepsize * grad / temperature + noise # negative!
            v = torch.bmm(P, dynamics.unsqueeze(2)).squeeze(2)
            v_norm = v.norm(dim=1, keepdim=True)
            xnew = x * torch.cos(v_norm) + v/v_norm * torch.sin(v_norm)
        else:
            dynamics = - stepsize * grad / temperature + noise # negative!
            xnew = x + dynamics
        if clip_x is not None:
            if reject_boundary:
                accept = ((xnew >= clip_x[0]) & (xnew <= clip_x[1])).view(len(x), -1).all(dim=1)
                reject = ~ accept
                xnew[reject] = x[reject]
                x = xnew
            else:
                x = torch.clamp(xnew, clip_x[0], clip_x[1])
        else:
            x = xnew

        # if spherical:
        #     if len(x.shape) == 4:
        #         x = x / x.view(len(x), -1).norm(dim=1)[:, None, None ,None]
        #     else:
        #         x = x / x.norm(dim=1, keepdim=True)

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)

        l_dynamics.append(dynamics.detach().to('cpu'))
        l_drift.append((- stepsize * grad).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
    l_samples.append(x.detach().to('cpu'))

    if intermediate_samples:
        return l_samples, l_dynamics, l_drift, l_diffusion
    else:
        return x.detach()