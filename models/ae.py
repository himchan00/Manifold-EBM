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
    f(x) = -log(1 - x) when x < 0, x + x**2/2 when x >= 0
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        index = input < 0
        output = torch.zeros_like(input)
        output[index] = -torch.log(1 - input[index])
        output[~index] = input[~index] + input[~index]**2/2
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        index = input < 0
        grad_input[index] = 1/(1 - input[index])
        grad_input[~index] = 1 + input[~index]
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

class EnergyAE(nn.Module):
    def __init__(
        self, encoder, decoder, sigma_sq=1e-4,
        radius = 1.0, train_sigma = True, epsilon = 1e-3
    ):
        super(EnergyAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = self.encoder.z_dim
        self.x_dim = self.encoder.x_dim
        self.radius = radius
        self.train_sigma = train_sigma
        self.epsilon = epsilon
        self.register_parameter("log_sigma_sq", nn.Parameter(torch.log(sigma_sq * torch.ones(1))))
        self.register_parameter("constant_term", nn.Parameter(torch.tensor(0.0)))

    def sample(self, batch_size, device, apply_noise = True):

        z = torch.randn((batch_size, self.z_dim), device=device)
        # decode
        with torch.no_grad():
            x = self.decoder(z)
            if apply_noise:
                    sigma = self.get_sigma(z)
                    x = x + torch.randn_like(x) * sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        return x
    
    def get_riemannian_metric(self, z_star, create_graph = True):
        J = jacobian_of_f(self.decoder, z_star, create_graph=create_graph)
        G = J.permute(0, 2, 1)@J + self.epsilon * torch.eye(z_star.shape[1]).to(z_star)
        log_det = torch.logdet(G) / 2
        if torch.isnan(log_det).sum() > 0:
            print("nan occured in log_det")
            log_det[torch.isnan(log_det)] = 0
        if torch.isinf(log_det).sum() > 0:
            print("inf occured in log_det")
            log_det[torch.isinf(log_det)] = 0
        return G,log_det
    
    def get_sigma(self, z):
        """
        returns sigma of shape (B,)
        """
        if self.train_sigma:
            return self.decoder.sigma(z).squeeze()
        else:
            bs = z.shape[0]
            return torch.exp(self.log_sigma_sq).repeat(bs)
        
    def conditional_energy_function(self, x, z):
        x_star = self.decoder(z).view(-1) # (D)
        D = x_star.shape[0]
        sigma = self.decoder.sigma(z) 
        energy = (x - x_star).pow(2).sum() / (2 * (sigma ** 2))
        energy = energy + D * torch.log(sigma)
        return energy.squeeze()


    def pretrain_step(self, x, optimizer_pre, is_neg = False, **kwargs):
        optimizer_pre.zero_grad()
        d_train = self.neg_log_prob(x, eval = False)
        loss = d_train["neg_log_prob"]
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)

        optimizer_pre.step()
        
        d_logging = {"loss" : loss.item()}

        for key, values in d_train.items():
            if is_neg:
                d_logging["pretrain/neg_" + key + "_"] = values.mean().item()
            else:
                d_logging["pretrain/" + key + "_"] = values.mean().item()
        return d_logging
    

    def train_step(self, x, neg_x, optimizer, **kwargs):
        optimizer.zero_grad()
        z_star = self.minimizer(x).detach() # (B, n)
        z_star_neg = self.minimizer(neg_x).detach() # (B, n)

        x = x.view(len(x), -1) # (B, D)
        neg_x = neg_x.view(len(neg_x), -1) # (B, D)
        sigma = self.decoder.sigma(z_star)
        signa_neg = self.decoder.sigma(z_star_neg)

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
        # sigma_loss = (D) * torch.log(sigma) # (B, )
        # sigma_loss_neg = (D) * torch.log(signa_neg) # (B, ) 

        # second order loss
        s = 1.0
        sigma_detached = sigma.detach()
        radius = s * sigma_detached
        # Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        second_order_term = (Trace_hess) * (radius ** 2) / (2*n+4)
        second_order_loss = neg_log_approx.apply(second_order_term)

        sigma_detached_neg = signa_neg.detach()
        radius_neg = s * sigma_detached_neg
        # Trace_grad_neg = (- grad_neg.unsqueeze(1) @ torch.linalg.solve(G_neg, grad_neg.unsqueeze(2))).squeeze() # (B, )
        Trace_hess_neg = vmap(torch.trace)(torch.linalg.solve(G_neg, hess_neg)) # (B, )

        second_order_term_neg = (Trace_hess_neg) * (radius_neg ** 2) / (2*n+4)
        second_order_loss_neg = neg_log_approx.apply(second_order_term_neg)

        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius) + self.constant_term
        
        pos_e = (recon_loss + invariant_energy + second_order_loss + constant_term)/D
        neg_e = (recon_loss_neg + invariant_energy_neg + second_order_loss_neg + constant_term)/D
        loss = pos_e.mean() - neg_e.mean()
        reg_loss = neg_e.pow(2).mean() + pos_e.pow(2).mean()
        loss = loss + 1.0 * reg_loss
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
    
    def neg_log_prob(self, x, eval = False):
        z_star = self.encoder(x)
        z_star_detached = z_star.detach()
        x = x.view(len(x), -1) # (B, D)
        sigma = self.get_sigma(z_star) # (B,)

        bs = x.shape[0]
        D = self.x_dim
        n = self.z_dim

        # compute gradient
        compute_grad = vmap(jacrev(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        grad = compute_grad(x, z_star_detached) # (B, n)

        # compute hessian
        compute_batch_hessian = vmap(hessian(self.conditional_energy_function, argnums = 1), in_dims = (0, 0))
        hess = compute_batch_hessian(x, z_star_detached) # (B, n, n)
        
        # compute recon loss
        recon = self.decoder(z_star).view(len(x), -1) # (B, D)
        recon_loss = ((recon - x) ** 2).sum(dim = 1) / (2 * (sigma ** 2)) # (B,)
        sigma_loss = D * torch.log(sigma) # (B,)

        # compute log det and latent energy
        G, log_det = self.get_riemannian_metric(z_star, create_graph=True)
        latent_energy = (z_star ** 2).sum(dim = 1) / 2
        invariant_energy = log_det + latent_energy


        # second order loss
        if not eval:
            J = jacobian_of_f(self.decoder, z_star_detached, create_graph=True)
            G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z_star_detached.shape[1]).to(z_star_detached)
        s = self.radius * sigma.detach() # (B, )

        # Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        second_order_term = Trace_hess * (s ** 2) / (2*n+4) 
        second_order_loss = neg_log_approx.apply(second_order_term)

        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(s) + self.constant_term
        
        neg_log_prob = (recon_loss + sigma_loss + invariant_energy + second_order_loss + constant_term)/D

        return {"neg_log_prob": neg_log_prob,
                'recon_loss': recon_loss,
                'latent_energy': latent_energy,
                'log_det': log_det,
                'second_order_loss': second_order_loss,
                'gradient': grad,
                'sigma': sigma}
    

    def visualization_step(self, dl, procedure, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        z_star = self.encoder(x.to(device))

        recon = self.decoder(z_star)
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        if procedure == 'train_energy' or procedure == "train":
            sampled_x = self.sample(batch_size=x.shape[0],  device=device, apply_noise=True)
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
        