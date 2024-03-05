import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import cm
from functorch import hessian, vmap, jacrev
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse
from geometry import (
    relaxed_volume_preserving_measure,
    get_pullbacked_Riemannian_metric,
    get_flattening_scores,
    get_log_det_jacobian,
    jacobian_of_f,
    relaxed_distortion_measure,
    get_log_det_jacobian_new,
    get_projection_coord_rep,
    conformal_distortion_measure,
    curvature_reg,
)
import scipy

class neg_log_approx(torch.autograd.Function):
    """
    f(x) = -log(1 - x) when x < 0, x + x**2/2 + x**3/3 + x**4/4 when x >= 0
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        index = input < 0
        output = torch.zeros_like(input)
        output[index] = -torch.log(1 - input[index])
        output[~index] = input[~index] + input[~index]**2/2 + input[~index]**3/3 + input[~index]**4/4
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        index = input < 0
        grad_input[index] = 1/(1 - input[index])
        grad_input[~index] = 1 + input[~index] + input[~index]**2 + input[~index]**3
        return grad_output * grad_input
    
class erfi(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_npy = input.detach().cpu().numpy()
        output_npy = scipy.special.erfi(input_npy)
        return torch.as_tensor(output_npy, dtype=input.dtype).to(input.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * torch.exp(input ** 2) / (np.pi ** 0.5)
        return grad_output * grad_input
    
class dawson(torch.autograd.Function):
    """
    dawson(x) = exp(-x^2) erfi(x) * sqrt(pi) / 2
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_npy = input.detach().cpu().numpy()
        output_npy = scipy.special.dawsn(input_npy)
        return torch.as_tensor(output_npy, dtype=input.dtype).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_npy = input.detach().cpu().numpy()
        grad_input_npy = -2 * input_npy * scipy.special.dawsn(input_npy) + 1
        grad_input = torch.as_tensor(grad_input_npy, dtype=input.dtype).to(input.device)
        return grad_output * grad_input
    
class erfcx(torch.autograd.Function):
    """
    erfcx(x) = exp(x^2) erfc(x)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_npy = input.detach().cpu().numpy()
        output_npy = scipy.special.erfcx(input_npy)
        return torch.as_tensor(output_npy, dtype=input.dtype).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_npy = input.detach().cpu().numpy()
        grad_input_npy = 2 * input_npy * scipy.special.erfcx(input_npy) - 2 / np.pi ** 0.5
        grad_input = torch.as_tensor(grad_input_npy, dtype=input.dtype).to(input.device)
        return grad_output * grad_input


def jacobian(output, input_tensor):
    """
    Calculate the Jacobian matrix.

    Args:
        output (torch.Tensor): Output tensor of shape (bs, n).
        input_tensor (torch.Tensor): Input tensor of shape (bs, m).

    Returns:
        torch.Tensor: Jacobian matrix of shape (bs, n, m).
    """
    bs, n = output.shape
    _, m = input_tensor.shape
    jac = torch.zeros(bs, n, m).to(output)

    for i in range(n):
        grad_output_i = torch.zeros_like(output)
        grad_output_i[:, i] = 1.0
        jac[:, i, :] = torch.autograd.grad(output, input_tensor, grad_outputs=grad_output_i, retain_graph=True)[0]

    return jac

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        z = self.encoder(x)
        if len(z.size()) == 4:
            z = z.squeeze(2).squeeze(2)
        half_chan = int(z.shape[1] / 2)
        return z[:, :half_chan]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

class EnergyAE(AE):
    def __init__(
        self, encoder, decoder, ebm, sigma, minimizer, sigma_sq=1e-4, 
        n_eval = None, curvature_reg = None, train_sigma = True
    ):
        super(EnergyAE, self).__init__(encoder, decoder)
        self.ebm = ebm
        self.n_eval = n_eval
        self.curvature_reg = curvature_reg
        self.train_sigma = train_sigma
        self.sigma = sigma
        self.decoder_target = copy.deepcopy(decoder)
        self.minimizer = minimizer
        self.register_parameter("log_sigma_sq", nn.Parameter(torch.log(sigma_sq * torch.ones(1))))
        self.register_parameter("constant_term", nn.Parameter(torch.tensor(0.0)))

    def sample(self, shape,  device,apply_noise = True):
        # sample from latent space
        # z = self.ebm.sample(shape=shape, sample_step = sample_step, device=device, replay=replay)
        z = torch.randn(shape, device=device)
        # decode
        with torch.no_grad():
            x = self.decode(z)
            if apply_noise:
                    sigma = self.decoder.sigma(z)
                    # sigma = torch.exp(self.log_sigma_sq)
                    x = x + torch.randn_like(x) * sigma.unsqueeze(1).unsqueeze(2)
        
        return x
    
    def conditional_energy_function(self, x, z):
        x_star = self.decoder(z).view(-1) # (D)
        D = x_star.shape[0]
        sigma = self.decoder.sigma(z) 
        energy = (x - x_star).pow(2).sum() / (2 * (sigma ** 2))
        energy = energy + D * torch.log(sigma)
        return energy.squeeze()
    
    def half_riemannian_metric(self, z, create_graph=True):
        J = jacobian_of_f(self.decoder, z, create_graph=create_graph)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z.shape[1]).to(z)
        return G/2


    def new_pretrain_step(self, x, optimizer_pre, is_neg = False, **kwargs):
        optimizer_pre.zero_grad()
        z_star = self.minimizer(x)

        x = x.view(len(x), -1) # (B, D)
        sigma = self.decoder.sigma(z_star) # (B, )

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]

        # compute recon loss, which is conditional energy E(x|x_bar)
        compute_recon = vmap(self.conditional_energy_function, in_dims = (0, 0))
        recon_loss = compute_recon(x, z_star) # (B, )

        # compute gradient
        compute_grad = vmap(jacrev(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)

        # compute hessian
        compute_batch_hessian = vmap(hessian(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        
        # compute log det and latent energy
        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(n).to(z_star)
        log_det = torch.logdet(G) / 2
        log_det[torch.isnan(log_det)] = 0
        log_det[torch.isinf(log_det)] = 0
        latent_energy = (z_star ** 2).sum(dim = 1) / 2
        invariant_energy = log_det + latent_energy

        # sigma loss
        # sigma_loss = (D) * torch.log(sigma) # (B, )

        # second order loss
        s = 1.0
        sigma_detached = sigma #.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        second_order_term = (Trace_grad + Trace_hess) * (radius ** 2) / (2*n+4)
        second_order_loss = neg_log_approx.apply(second_order_term)

        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)
        
        # geometric regularization to make G numerically stable (condition number = (max_eigenvalue/min_eigenvalue))
        geometric_loss = relaxed_distortion_measure(self.decoder, z_star, create_graph=True)

        loss = (recon_loss + invariant_energy + constant_term+ second_order_loss+ 1 * geometric_loss)/D #  
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.minimizer.parameters(), 0.1)

        optimizer_pre.step()

        if is_neg:
            return {"loss_neg": loss.item(),
                    "pre_neg/recon_loss_neg_": recon_loss.mean().item(),
                    "pre_neg/invariant_energy_neg_": invariant_energy.mean().item(),
                    "pre_neg/log_det_neg_": log_det.mean().item(),
                   "pre_neg/latent_energy_neg_": latent_energy.mean().item(),
                    "pre_neg/second_order_loss_neg_": second_order_loss.mean().item(),
                    "pre_neg/geometric_loss_neg_": geometric_loss.mean().item(),
                    "pre_neg/sigma_neg_": sigma.mean().item()}
        else:
            return {"loss": loss.item(),
                    "pre/recon_loss_": recon_loss.mean().item(),
                    "pre/invariant_energy_": invariant_energy.mean().item(),
                    "pre/log_det_": log_det.mean().item(),
                    "pre/latent_energy_": latent_energy.mean().item(),
                    "pre/second_order_loss_": second_order_loss.mean().item(),
                    "pre/geometric_loss_": geometric_loss.mean().item(),
                    "pre/sigma_": sigma.mean().item()}
    
    def pretrain_step(self, x, optimizer_pre, is_neg = False, **kwargs):

        optimizer_pre.zero_grad()
        z_star = self.minimizer(x) # (B, n)

        x = x.view(len(x), -1) # (B, D)
        sigma = torch.exp(self.log_sigma_sq)
        
        D = x.shape[1]
        n = z_star.shape[1]

        # compute energy
        compute_energy = vmap(self.energy_function, in_dims = (0, 0))
        energy = compute_energy(x, z_star) # (B, )

        # gradient of energy
        compute_grad = vmap(jacrev(self.energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)

        # hessian of energy
        compute_batch_hessian = vmap(hessian(self.energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        

        G = self.half_riemannian_metric(z_star) * 2  # (B, n, n)
        sqrt_G = torch.linalg.cholesky(G, upper=True)

        n_hat = torch.randn_like(z_star)
        n_hat = n_hat / n_hat.norm(dim=1, keepdim=True)
        v = torch.linalg.solve(sqrt_G, n_hat.unsqueeze(2)).squeeze() # (B, n)
        _, C = torch.autograd.functional.jvp(self.half_riemannian_metric, z_star, v, create_graph=True)
        c = (v.unsqueeze(1) @ C @ v.unsqueeze(2)).squeeze() # (B,)
        c_sq = c ** 2

        # energy_loss
        energy_loss = energy # (B, )

        # log det loss
        log_det_loss = torch.logdet(G) / 2 # (B, )
        log_det_loss[torch.isnan(log_det_loss)] = 0
        log_det_loss[torch.isinf(log_det_loss)] = 0

        # sigma loss
        sigma_loss = (D) * torch.log(sigma) # (B, )

        # second order loss
        # intrgrand part
        s = 2.0
        sigma_detached = sigma.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        grad_term = Trace_grad / (2*n+4)
        hess_term = Trace_hess / (2*n+4)

       # integration area part
        volume_term = - c_sq * (5*n)/ (8)
        moment_term = - (grad.unsqueeze(1) @ v.unsqueeze(2)).squeeze() * n /(2)
        
        second_order_term = grad_term + hess_term + volume_term + moment_term
        ratio = 1 - second_order_term * (radius ** 2)
        ratio = ratio.clamp(min=1e-8, max = 2)
        second_order_loss = - torch.log(ratio)
        
        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)

        loss  = (energy_loss + log_det_loss + sigma_loss + second_order_loss + constant_term)/D 
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.minimizer.parameters(), 0.1)
        # torch.nn.utils.clip_grad_norm_(self.sigma.net.parameters(), 0.1)

        optimizer_pre.step()
        if is_neg:
            return {"loss_neg": loss.item(),
                    "pre_neg/log_det_loss_neg_": log_det_loss.mean().item(),
                    "pre_neg/energy_loss_neg_": energy_loss.mean().item(),
                    "pre_neg/grad_term_neg_": grad_term.mean().item(),
                    "pre_neg/hess_term_neg_": hess_term.mean().item(),
                    "pre_neg/volume_term_neg_": volume_term.mean().item(),
                    "pre_neg/moment_term_neg_": moment_term.mean().item(),
                    'pre_neg/second_order_loss_neg_': second_order_loss.mean().item(),
                    "pre_neg/sigma_neg_": sigma.mean().item(),
                    "pre_neg/c_neg_": c.mean().item()}
        else:
            return {"loss": loss.item(), 
                    "pre/log_det_loss_": log_det_loss.mean().item(),
                    "pre/energy_loss_": energy_loss.mean().item(),
                    "pre/grad_term_": grad_term.mean().item(),
                    "pre/hess_term_": hess_term.mean().item(),
                    "pre/volume_term_": volume_term.mean().item(),
                    "pre/moment_term_": moment_term.mean().item(),
                    'pre/second_order_loss_': second_order_loss.mean().item(),
                    "pre/sigma_": sigma.mean().item(),
                    "pre/c_": c.mean().item()}
    

    def new_train_step(self, x, neg_x, optimizer, **kwargs):
        optimizer.zero_grad()
        z_star = self.minimizer(x).detach() # (B, n)
        z_star_neg = self.minimizer(neg_x).detach() # (B, n)

        x = x.view(len(x), -1) # (B, D)
        neg_x = neg_x.view(len(neg_x), -1) # (B, D)
        sigma = self.minimizer.sigma(x).detach() # (B, )
        signa_neg = self.minimizer.sigma(neg_x).detach() # (B, )

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]

        # compute recon loss, which is conditional energy E(x|x_bar)
        compute_recon = vmap(self.conditional_energy_function, in_dims = (0, 0))
        recon_loss = compute_recon(x, z_star) # (B, )
        recon_loss_neg = compute_recon(neg_x, z_star_neg)

        # compute gradient
        compute_grad = vmap(jacrev(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)
        grad_neg = compute_grad(neg_x, z_star_neg)

        # compute hessian
        compute_batch_hessian = vmap(hessian(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        hess_neg = compute_batch_hessian(neg_x, z_star_neg)
        
        # compute log det and latent energy
        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        J_neg = jacobian_of_f(self.decoder, z_star_neg, create_graph=True)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z_star.shape[1]).to(z_star)
        G_neg = J_neg.permute(0, 2, 1)@J_neg + 1e-3* torch.eye(z_star_neg.shape[1]).to(z_star_neg)

        log_det = torch.logdet(G) / 2
        log_det_neg = torch.logdet(G_neg) / 2
        log_det[torch.isnan(log_det)] = 0
        log_det[torch.isinf(log_det)] = 0
        log_det_neg[torch.isnan(log_det_neg)] = 0
        log_det_neg[torch.isinf(log_det_neg)] = 0

        latent_energy = (z_star ** 2).sum(dim = 1) / 2
        latent_energy_neg = (z_star_neg ** 2).sum(dim = 1) / 2
        invariant_energy = latent_energy + log_det
        invariant_energy_neg = latent_energy_neg + log_det_neg

        # sigma loss
        sigma_loss = (D) * torch.log(sigma) # (B, )
        sigma_loss_neg = (D) * torch.log(signa_neg) # (B, ) 

        # second order loss
        s = 1.0
        sigma_detached = sigma #.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        second_order_term = (Trace_grad + Trace_hess) * (radius ** 2) / (2*n+4)
        second_order_loss = neg_log_approx.apply(second_order_term)

        sigma_detached_neg = signa_neg #.detach()
        radius_neg = s * sigma_detached_neg
        Trace_grad_neg = (- grad_neg.unsqueeze(1) @ torch.linalg.solve(G_neg, grad_neg.unsqueeze(2))).squeeze() # (B, )
        Trace_hess_neg = vmap(torch.trace)(torch.linalg.solve(G_neg, hess_neg)) # (B, )

        second_order_term_neg = (Trace_grad_neg + Trace_hess_neg) * (radius_neg ** 2) / (2*n+4)
        second_order_loss_neg = neg_log_approx.apply(second_order_term_neg)

        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)
        
        pos_e = (recon_loss + invariant_energy + sigma_loss + second_order_loss + constant_term)/D
        neg_e = (recon_loss_neg + invariant_energy_neg + sigma_loss_neg + second_order_loss_neg + constant_term)/D
        loss = pos_e.mean() - neg_e.mean()
        reg_loss = pos_e.pow(2).mean() + neg_e.pow(2).mean()
        loss = loss + 0.1 * reg_loss
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.minimizer.parameters(), 0.1)

        optimizer.step()

        return {"loss": loss.item(),
                'AE/pos_e_': pos_e.mean().item(),
                "AE/log_det_loss_": log_det.mean().item(),
                "AE/lantent_energy_": latent_energy.mean().item(),
                "AE/recon_loss_": recon_loss.mean().item(),
                "AE/second_order_loss_": second_order_loss.mean().item(),
                "AE/sigma_": sigma.mean().item(),
                'AE_neg/neg_e_': neg_e.mean().item(),
                "AE_neg/neg_log_det_loss_": log_det_neg.mean().item(),
                "AE_neg/neg_lantent_energy_": latent_energy_neg.mean().item(),
                "AE_neg/neg_recon_loss_": recon_loss_neg.mean().item(),
                "AE_neg/neg_second_order_loss_": second_order_loss_neg.mean().item(),
                "AE_neg/neg_sigma_": signa_neg.mean().item()}
    
    def train_step(self, x, neg_x, optimizer, **kwargs):
        optimizer.zero_grad()
        z_star = self.minimizer(x).detach() # (B, n)
        neg_z_star = self.minimizer(neg_x).detach() # (B, n)

        x = x.view(len(x), -1) # (B, D)
        neg_x = neg_x.view(len(neg_x), -1) # (B, D)
        sigma = torch.exp(self.log_sigma_sq)
        
        D = x.shape[1]
        n = z_star.shape[1]

        # compute energy
        compute_energy = vmap(self.energy_function, in_dims = (0, 0))
        energy = compute_energy(x, z_star) # (B, )
        neg_energy = compute_energy(neg_x, neg_z_star) # (B, )

        # gradient of energy
        compute_grad = vmap(jacrev(self.energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)
        neg_grad = compute_grad(neg_x, neg_z_star) # (B, n)

        # hessian of energy
        compute_batch_hessian = vmap(hessian(self.energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        neg_hess = compute_batch_hessian(neg_x, neg_z_star) # (B, n, n)

        G = self.half_riemannian_metric(z_star) * 2  # (B, n, n)
        neg_G = self.half_riemannian_metric(neg_z_star) * 2  # (B, n, n)
        sqrt_G = torch.linalg.cholesky(G, upper=True)
        neg_sqrt_G = torch.linalg.cholesky(neg_G, upper=True)

        n_hat = torch.randn_like(z_star)
        n_hat = n_hat / n_hat.norm(dim=1, keepdim=True)
        v = torch.linalg.solve(sqrt_G, n_hat.unsqueeze(2)).squeeze() # (B, n)
        _, C = torch.autograd.functional.jvp(self.half_riemannian_metric, z_star, v, create_graph=True)
        c = (v.unsqueeze(1) @ C @ v.unsqueeze(2)).squeeze() # (B,)
        c_sq = c ** 2

        neg_n_hat = torch.randn_like(z_star)
        neg_n_hat = neg_n_hat / neg_n_hat.norm(dim=1, keepdim=True)
        neg_v = torch.linalg.solve(neg_sqrt_G, neg_n_hat.unsqueeze(2)).squeeze() # (B, n)
        _, neg_C = torch.autograd.functional.jvp(self.half_riemannian_metric, neg_z_star, neg_v, create_graph=True)
        neg_c = (neg_v.unsqueeze(1) @ neg_C @ neg_v.unsqueeze(2)).squeeze() # (B,)
        neg_c_sq = neg_c ** 2

        # energy_loss
        energy_loss = energy # (B, )
        neg_energy_loss = neg_energy # (B, )

        # log det loss
        log_det_loss = torch.logdet(G) / 2 # (B, )
        log_det_loss[torch.isnan(log_det_loss)] = 0
        log_det_loss[torch.isinf(log_det_loss)] = 0
        neg_log_det_loss = torch.logdet(neg_G) / 2 # (B, )
        neg_log_det_loss[torch.isnan(neg_log_det_loss)] = 0
        neg_log_det_loss[torch.isinf(neg_log_det_loss)] = 0

        # sigma loss
        sigma_loss = (D) * torch.log(sigma) # (B, )
        neg_sigma_loss = (D) * torch.log(sigma) # (B, )

        # second order loss
        # intrgrand part
        s = 1.0
        sigma_detached = sigma.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )
        neg_Trace_grad = (- neg_grad.unsqueeze(1) @ torch.linalg.solve(neg_G, neg_grad.unsqueeze(2))).squeeze()
        neg_Trace_hess = vmap(torch.trace)(torch.linalg.solve(neg_G, neg_hess))

        grad_loss = Trace_grad * (radius ** 2) / (2*n+4)
        hess_loss = Trace_hess * (radius ** 2) / (2*n+4)
        neg_grad_loss = neg_Trace_grad * (radius ** 2) / (2*n+4)
        neg_hess_loss = neg_Trace_hess * (radius ** 2) / (2*n+4)

       # integration area part
        volume_loss = - c_sq * (radius ** 2) * (5*n)/ (8)
        moment_loss = - (grad.unsqueeze(1) @ v.unsqueeze(2)).squeeze() * n * (radius ** 2) /(2)
        neg_volume_loss = - neg_c_sq * (radius ** 2) * (5*n)/ (8)
        neg_moment_loss = - (neg_grad.unsqueeze(1) @ neg_v.unsqueeze(2)).squeeze() * n * (radius ** 2) /(2)
        
        second_order_loss = grad_loss + hess_loss + volume_loss + moment_loss
        neg_second_order_loss = neg_grad_loss + neg_hess_loss + neg_volume_loss + neg_moment_loss
        
        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)


        pos_e = (energy_loss + log_det_loss + sigma_loss + second_order_loss + constant_term)/D
        neg_e = (neg_energy_loss + neg_log_det_loss + neg_sigma_loss + neg_second_order_loss + constant_term)/D
        loss = pos_e.mean() - neg_e.mean()
        reg_loss = pos_e.pow(2).mean() + neg_e.pow(2).mean()
        loss = loss + 0.1 * reg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.minimizer.parameters(), 0.1)

        optimizer.step()
        return {"loss": loss.item(),
                'AE/pos_e_': pos_e.mean().item(),
                "AE/log_det_loss_": log_det_loss.mean().item(),
                "AE/energy_loss_": energy_loss.mean().item(),
                "AE/grad_loss_": grad_loss.mean().item(),
                "AE/hess_loss_": hess_loss.mean().item(),
                "AE/volume_loss_": volume_loss.mean().item(),
                "AE/moment_loss_": moment_loss.mean().item(),
                'AE/second_order_loss_': second_order_loss.mean().item(),
                "AE/sigma_": sigma.mean().item(),
                "AE/c_": c.mean().item(),
                'AE/neg_e_': neg_e.mean().item(),
                "AE/neg_log_det_loss_": neg_log_det_loss.mean().item(),
                "AE/neg_energy_loss_": neg_energy_loss.mean().item(),
                "AE/neg_grad_loss_": neg_grad_loss.mean().item(),
                "AE/neg_hess_loss_": neg_hess_loss.mean().item(),
                "AE/neg_volume_loss_": neg_volume_loss.mean().item(),
                "AE/neg_moment_loss_": neg_moment_loss.mean().item(),
                'AE/neg_second_order_loss_': neg_second_order_loss.mean().item(),
                "AE/neg_sigma_": sigma.mean().item(),
                "AE/neg_c_": c.mean().item()}

    
    def neg_log_prob(self, x, pretrain = False):    
        z_star = self.minimizer(x)

        x = x.view(len(x), -1) # (B, D)
        sigma = torch.exp(self.log_sigma_sq)

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]

        # compute energy
        compute_energy = vmap(self.energy_function, in_dims = (0, 0))
        energy = compute_energy(x, z_star) # (B, )

        # gradient of energy
        compute_grad = vmap(jacrev(self.energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)

        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z_star.shape[1]).to(z_star)
        U = torch.linalg.cholesky(G, upper=True) # U^T U = G
        
        # hessian of energy
        compute_batch_hessian = vmap(hessian(self.energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        A = torch.linalg.solve(U.permute(0, 2, 1), (torch.linalg.solve(U.permute(0, 2, 1), hess)).permute(0, 2, 1)) # (U^{-1})^THU^{-1} (B, n, n)
        eigvals_h, Q = torch.linalg.eigh(A) # A = Q \Lambda Q^T
        eigvals_sh = torch.linalg.eigvalsh(A) 
        error = torch.norm(eigvals_h - eigvals_sh)
        # energy_loss
        energy_loss = energy # (B, )

        # log det loss
        log_det_loss = torch.logdet(G) / 2 # (B, )
        log_det_loss[torch.isnan(log_det_loss)] = 0
        log_det_loss[torch.isinf(log_det_loss)] = 0


        # curvature loss
        s = 1.5
        radius = s * sigma.detach()
        a = eigvals_sh.view(-1) # (B*n, )
        b =  (Q.permute(0, 2, 1) @ torch.linalg.solve(U.permute(0, 2, 1), grad.unsqueeze(2))).squeeze().view(-1) # (B*n, )
        b = torch.abs(b)
        mask = a >= 0
        pos_a = a[mask]
        pos_b = b[mask]
        neg_a = a[~mask]
        neg_b = b[~mask]
        if len(pos_a) > 0:
            sqrt_pos_a = torch.sqrt(pos_a)
            z_start = (pos_b - 2*pos_a*radius)/(2*sqrt_pos_a)
            z_end = (pos_b + 2*pos_a*radius)/(2*sqrt_pos_a)
            mean = pos_b/(2*sqrt_pos_a)
            mask = z_start < 0
            z_start_neg = z_start[mask]
            z_start_pos = z_start[~mask]
            pos_a_neg = pos_a[mask]
            pos_a_pos = pos_a[~mask]
            mean_neg = mean[mask]
            mean_pos = mean[~mask]

            z_end_neg = z_end[mask]
            z_end_pos = z_end[~mask]
            if len(z_start_neg) > 0:
                pos_log_scale_neg =  -torch.log(2*torch.exp(mean_neg**2) - torch.exp(mean_neg**2-z_end_neg ** 2) * erfcx.apply(z_end_neg) - torch.exp(mean_neg**2-z_start_neg ** 2) * erfcx.apply(-z_start_neg)) + torch.log(pos_a_neg)/2
                pos_log_scale_neg[torch.isnan(pos_log_scale_neg)] = 0
                pos_log_scale_neg[torch.isinf(pos_log_scale_neg)] = 0
            else:
                pos_log_scale_neg = torch.tensor(0.0)
            if len(z_start_pos) > 0:
                pos_log_scale_pos = - torch.log(torch.exp(mean_pos**2 - z_start_pos**2)* erfcx.apply(z_start_pos) - torch.exp(mean_pos**2 - z_end_pos ** 2) * erfcx.apply(z_end_pos)) + torch.log(pos_a_pos)/2
                pos_log_scale_pos[torch.isnan(pos_log_scale_pos)] = 0
                pos_log_scale_pos[torch.isinf(pos_log_scale_pos)] = 0
            else:
                pos_log_scale_pos = torch.tensor(0.0)
            pos_log_scale = torch.cat([pos_log_scale_neg, pos_log_scale_pos], dim = 0)
        else:
            pos_log_scale = torch.tensor(0.0)
        if len(neg_a) > 0:
            sqrt_neg_a = torch.sqrt(-neg_a)
            z_start = (neg_b + 2*neg_a*radius)/(2*sqrt_neg_a)
            z_end = (neg_b - 2*neg_a*radius)/(2*sqrt_neg_a)
            mean = neg_b/(2*sqrt_neg_a)
            neg_log_scale = - torch.log(torch.exp(z_end ** 2 - mean ** 2) * dawson.apply(z_end) - torch.exp(z_start**2 - mean**2) * dawson.apply(z_start)) \
                + torch.log(-neg_a)/2 - torch.log(torch.tensor(2/np.pi**0.5))
            neg_log_scale[torch.isnan(neg_log_scale)] = 0
            neg_log_scale[torch.isinf(neg_log_scale)] = 0
        else:
            neg_log_scale = torch.tensor(0.0)
        if len(pos_a) > 0 and len(neg_a) > 0:
            scale = torch.cat([pos_log_scale, neg_log_scale], dim = 0)
        elif len(pos_a) > 0:
            scale = pos_log_scale
        else:
            scale = neg_log_scale
        curvature_loss = scale.sum()/bs

        # sigma loss
        sigma_loss = (D) * torch.log(sigma) # (B, )

        neg_log_prob  = (energy_loss + log_det_loss + sigma_loss + curvature_loss)/D

        return {"neg_log_prob": neg_log_prob,
                'log_det_loss': log_det_loss,
                'energy_loss': energy_loss,
                'sigma_loss': sigma_loss,
                #'curvature_loss': curvature_loss
        }

    def new_neg_log_prob(self, x, pretrain = False):
        z_star = self.minimizer(x)

        x = x.view(len(x), -1) # (B, D)
        sigma = self.decoder.sigma(z_star) # (B, )

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]

        # compute recon loss, which is conditional energy E(x|x_bar)
        compute_recon = vmap(self.conditional_energy_function, in_dims = (0, 0))
        recon_loss = compute_recon(x, z_star) # (B, )

        # compute gradient
        compute_grad = vmap(jacrev(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star) # (B, n)

        # compute hessian
        compute_batch_hessian = vmap(hessian(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star) # (B, n, n)
        
        # compute log det and latent energy
        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(n).to(z_star)
        log_det = torch.logdet(G) / 2
        log_det[torch.isnan(log_det)] = 0
        log_det[torch.isinf(log_det)] = 0
        latent_energy = (z_star ** 2).sum(dim = 1) / 2
        invariant_energy = log_det + latent_energy

        # sigma loss
        # sigma_loss = (D) * torch.log(sigma) # (B, )

        # second order loss
        s = 1.0
        sigma_detached = sigma #.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        second_order_term = (Trace_grad + Trace_hess) * (radius ** 2) / (2*n+4)
        second_order_loss = neg_log_approx.apply(second_order_term)

        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)
        
        neg_log_prob = (recon_loss + invariant_energy + constant_term+ second_order_loss)/D

        return {"neg_log_prob": neg_log_prob,
                'recon_loss': recon_loss,
                'invariant_energy': invariant_energy,
                'second_order_loss': second_order_loss,
                'sigma': sigma}
    

    def visualization_step(self, dl, procedure, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        z_star = self.minimizer(x.to(device))

        recon = self.decode(z_star)
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        if procedure == 'train_energy' or procedure == "train":
            sampled_x = self.sample(shape = z_star.shape,  device=device, apply_noise=True)
            sampled_img = make_grid(sampled_x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        
        if z_star.shape[1] == 3:
            # 2d graph (latent sapce)
            num_points_for_each_class = 200
            num_points_neg_samples = 200
            label_unique = torch.unique(dl.dataset.targets)
            z_ = []
            label_ = []
            for label in label_unique:
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class]
                temp_z = self.encoder(temp_data.to(device))

                z_.append(temp_z)
                label_.append(label.repeat(temp_z.size(0)))


            z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
            label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
            color_ = label_to_color(label_)
            if procedure == 'train_energy' or procedure == "train":
                sample_step = 100
                l_samples, accept = self.ebm.sample_mcmc_trajectory(shape=(num_points_neg_samples, z_.shape[1]), sample_step = sample_step, 
                                        device=device, replay=self.ebm.replay)
                print("accept rate : ", accept.sum()/len(accept))
                neg_z = l_samples[-1]

            f = plt.figure()
            ax = f.add_subplot(projection='3d')
            ax.axes.set_xlim3d(left=-1, right=1) 
            ax.axes.set_ylim3d(bottom=-1, top=1) 
            ax.axes.set_zlim3d(bottom=-1, top=1) 
            plt.title('Encoded points via encoder')
            for label in label_unique:
                label = label.item()
                ax.scatter(z_[label_==label,0], z_[label_==label,1], z_[label_==label, 2], s = 5, c=color_[label_==label]/255, label=label)
            if procedure == 'train_energy' or procedure == "train":    
                ax.scatter(neg_z[:,0], neg_z[:,1], neg_z[:, 2], c='black', marker='x')
            plt.legend()
            plt.close()
            f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]

            if procedure == 'train_energy' or procedure == "train":  
                f = plt.figure()
                ax = f.add_subplot(projection='3d')
                ax.axes.set_xlim3d(left=-1, right=1)
                ax.axes.set_ylim3d(bottom=-1, top=1)  
                ax.axes.set_zlim3d(bottom=-1, top=1)
                plt.title('MCMC trajectory')
                trajectory = torch.stack(l_samples, dim=1).detach().cpu().numpy()[0]
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='black')
                plt.close()
                f_np_mcmc = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]

                f = plt.figure()
                ax = f.add_subplot(projection='3d')
                ax.axes.set_xlim3d(left=-1, right=1) 
                ax.axes.set_ylim3d(bottom=-1, top=1) 
                ax.axes.set_zlim3d(bottom=-1, top=1) 
                plt.title('Latent space energy')
                u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
                x = np.cos(u)*np.sin(v)
                y = np.sin(u)*np.sin(v)
                z = np.cos(v)
                Energy = self.ebm(torch.tensor(np.stack([x, y, z], axis=2).reshape(-1, 3), dtype=torch.float32).to(device)).detach().cpu().numpy().reshape(40, 20)
                ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', facecolors=cm.jet(Energy))
                m = cm.ScalarMappable(cmap=cm.jet)
                m.set_array(Energy)
                plt.colorbar(m)
                plt.close()
                f_np_energy = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]


                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon@': torch.clip(recon_img, min=0, max=1),
                    'sampled@': torch.clip(sampled_img, min=0, max=1),
                    'latent_space@': f_np,
                    'latent_space_energy@': f_np_energy,
                    'latent_space_mcmc@': f_np_mcmc
                }
            else:
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon@': torch.clip(recon_img, min=0, max=1),
                }
        else:
            if procedure == 'train_energy' or procedure == "train":  
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon@': torch.clip(recon_img, min=0, max=1),
                    'sampled@': torch.clip(sampled_img, min=0, max=1),
                }
            else:
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon@': torch.clip(recon_img, min=0, max=1),
                }   

class VAE(AE):
    def __init__(
        self, encoder, decoder
    ):
        super(VAE, self).__init__(encoder, decoder)

    def encode(self, x):
        z = self.encoder(x)
        if len(z.size()) == 4:
            z = z.squeeze(2).squeeze(2)
        half_chan = int(z.shape[1] / 2)
        return z[:, :half_chan]

    def decode(self, z):
        return self.decoder(z)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        
        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)

        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            # "nll_": nll.item(),
            # "kl_loss_": kl_loss.mean(),
            # "sigma_": self.decoder.sigma.item(),
        }

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]
        score = []
        for x, _ in dl:
            z = self.encode(x.to(device))
            G = get_pullbacked_Riemannian_metric(self.decode, z)
            score.append(get_flattening_scores(G, mode='condition_number'))
        mean_condition_number = torch.cat(score).mean()
        return {
            "MCN_": mean_condition_number.item()
        }

    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        recon = self.decode(self.encode(x.to(device)))
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)

        # 2d graph (latent sapce)
        num_points_for_each_class = 200
        num_G_plots_for_each_class = 20
        label_unique = torch.unique(dl.dataset.targets)
        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class]
            temp_z = self.encode(temp_data.to(device))
            z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
            G = get_pullbacked_Riemannian_metric(self.decode, z_sampled)

            z_.append(temp_z)
            label_.append(label.repeat(temp_z.size(0)))
            z_sampled_.append(z_sampled)
            label_sampled_.append(label.repeat(z_sampled.size(0)))
            G_.append(G)


        z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
        label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
        color_ = label_to_color(label_)
        G_ = torch.cat(G_, dim=0).detach().cpu()
        z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
        label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
        color_sampled_ = label_to_color(label_sampled_)

        f = plt.figure()
        plt.title('Latent space embeddings with equidistant ellipses')
        z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
        eig_mean = torch.svd(G_).S.mean().item()
        scale = 0.1 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3
        for idx in range(len(z_sampled_)):
            e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
            plt.gca().add_artist(e)
        for label in label_unique:
            label = label.item()
            plt.scatter(z_[label_==label,0], z_[label_==label,1], c=color_[label_==label]/255, label=label)
        plt.legend()
        plt.axis('equal')
        plt.close()
        f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]

        return {
            'input@': torch.clip(x_img, min=0, max=1),
            'recon@': torch.clip(recon_img, min=0, max=1),
            'latent_space@': f_np
        }

class IRVAE(VAE):
    def __init__(
        self, encoder, decoder, iso_reg=1.0, metric='identity', 
    ):
        super(IRVAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric
    
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        
        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = relaxed_distortion_measure(self.decode, z_sample, eta=0.2, metric=self.metric)
          
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}
        