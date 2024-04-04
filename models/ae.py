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
    spherical_jacobian_of_f,
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

class EnergyAE(nn.Module):
    def __init__(
        self, encoder, decoder, z_dim, max_latent_variance = 0.1
    ):
        super(EnergyAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim
        self.max_latent_variance = torch.tensor(max_latent_variance)


    def sample(self, batch_size, device):

        z = torch.randn((batch_size, self.z_dim), device=device)
        # decode
        with torch.no_grad():
            x = self.decoder(z)
        
        return x
    

    def function_for_hessian(self, z, x, sigma):
        """
        caculate d.detach().T @ d
        x : (D)
        z : (n)
        sigma : (1)
        returns (,)
        """
        x = x.unsqueeze(0) # (1, D)
        z = z.unsqueeze(0) # (1, n)
        x_star = self.decoder(z).view(1, -1) # (1, D)
        d = (x - x_star) / sigma.unsqueeze(1) # (1, D)
        d_detached = d.detach() 
        return (d_detached @ d.T).squeeze()
    
        
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        d_train = self.neg_log_prob(x)
        loss = d_train["neg_log_prob"]
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)

        optimizer.step()
        
        d_logging = d_train
        d_logging["loss"] = loss.item()
        return d_logging

    def neg_log_prob(self, x, n_eval = 1):
        z_star, sigma = self.encoder.forward_with_sigma(x)
        x = x.view(len(x), -1) # (B, D)

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]
        
        # compute jacobian
        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        # compute hessian
        compute_batch_hessian = vmap(hessian(self.function_for_hessian, argnums = 0), in_dims = (0, 0, 0))
        hess = compute_batch_hessian(z_star, x, sigma.unsqueeze(1)) # (B, n, n)
        # compute precision matrix
        Precision = J.permute(0, 2, 1)@J/((sigma**2).unsqueeze(1).unsqueeze(2)) + torch.eye(n).to(z_star) + hess
        # Find minimum eigenvalue
        eigvals = torch.linalg.eigvalsh(Precision)
        eigvals_min = torch.min(eigvals, dim = 1).values # (B,)
        
        # Insure that eigenvalues of precision matrix are greater than 1/max_latent_variance
        delta = 1/self.max_latent_variance - eigvals_min # (B,)
        Precision = Precision + torch.eye(n).to(z_star).repeat(bs, 1, 1) * delta.unsqueeze(1).unsqueeze(2)
        eigvals = eigvals + delta.unsqueeze(1)

        # posterior sampling
        L = torch.linalg.cholesky(Precision, upper = True)
        eps = torch.randn((n_eval, bs, n)).to(z_star)
        z_sample = z_star.unsqueeze(0) + torch.linalg.solve(L.unsqueeze(0), eps.unsqueeze(-1)).squeeze(-1) # (n_eval, B, n)

        recon = self.decoder(z_sample.view(-1, n)).view(n_eval, bs, D)

        # compute recon loss
        recon_loss = ((recon - x.unsqueeze(0)) ** 2).sum(dim = 2).mean(dim = 0) / (2 * (sigma ** 2)) # (B,)

        # compute latent energy
        latent_energy = (z_star ** 2).sum(dim = 1) / 2 + (1 / eigvals).sum(dim = 1)/2

        # compute log det
        logdet_loss = torch.log(eigvals).sum(dim = 1)/2 # (B,)

        # compute sigma_loss
        sigma_loss = D * torch.log(sigma) # (B,)
        neg_log_prob = (recon_loss + latent_energy + logdet_loss + sigma_loss)/D

        d_return = {"neg_log_prob": neg_log_prob,
                    "recon_loss": recon_loss,
                   "latent_energy": latent_energy,
                    "log_det": logdet_loss,
                    "sigma": sigma}

        return d_return
    

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
            sampled_x = self.sample(batch_size=x.shape[0],  device=device)
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
        