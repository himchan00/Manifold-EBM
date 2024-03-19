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
        self, encoder, decoder, sigma_sq=1e-4,
        radius = 1.0, train_sigma = True, epsilon = 1e-3,
        gamma = 1.0, normalize = False
    ):
        super(EnergyAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = 16
        self.x_dim = 784
        self.radius = radius
        self.train_sigma = train_sigma
        self.epsilon = epsilon
        self.gamma = gamma
        self.normalize = normalize
        self.register_parameter("log_sigma_sq", nn.Parameter(torch.log(sigma_sq * torch.ones(1))))
        self.register_parameter("constant_term", nn.Parameter(torch.tensor(0.0)))

    def sample(self, batch_size, device, apply_noise = True):

        z = torch.randn((batch_size, self.z_dim), device=device)
        if self.normalize:
            z = z / z.norm(dim=1, keepdim=True)
        # decode
        with torch.no_grad():
            x = self.decoder(z)
            if apply_noise:
                    sigma = self.get_sigma(z)
                    x = x + torch.randn_like(x) * sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        return x
    
    def scaled_displacement(self, z, x):
        """
        calculate the scaled displacement, d = (x - f(z))/sigma(z)
        x : (B, D)
        z : (B, n)
        returns d : (B, D)
        """
        x_star = self.decoder(z).view(len(x), -1) # (B, D)
        sigma = self.get_sigma(z) # (B,)
        return (x - x_star) / sigma.unsqueeze(1)

    def function_for_hessian(self, z, x):
        """
        caculate d.detach().T @ d + D * log(sigma)
        x : (D)
        z : (n)
        returns (,)
        """
        x = x.unsqueeze(0) # (1, D)
        z = z.unsqueeze(0) # (1, n)
        d = self.scaled_displacement(z, x) # (1, D)
        d_detached = d.detach() # (1, D)
        sigma = self.get_sigma(z) # (1,)
        D = x.shape[1]
        return (d_detached @ d.T + D * torch.log(sigma)).squeeze()
    
    def get_sigma(self, z):
        """
        z : (B, n)
        returns sigma of shape (B,)
        """
        if self.train_sigma:
            return self.decoder.sigma(z).squeeze(-1)
        else:
            bs = z.shape[0]
            return torch.exp(self.log_sigma_sq).repeat(bs)
    
    def get_energy(self, x, z):
        """
        x : (D), z: (n)
        """
        x_star = self.decoder(z).view(-1) # (D)
        D = x.shape[0]
        if self.train_sigma:
            sigma = self.decoder.sigma(z)
        else:
            sigma = torch.exp(self.log_sigma_sq)
        energy = (x - x_star).pow(2).sum() / (2 * (sigma ** 2))
        energy = energy + D * torch.log(sigma)
        energy = energy + 0.5 * (z ** 2).sum()
        return energy.squeeze()
    
    def conditional_energy_function(self, x, z):
        """
        x : (D), z: (n)
        """
        x_star = self.decoder(z).view(-1) # (D)
        D = x_star.shape[0]
        if self.train_sigma:
            sigma = self.decoder.sigma(z)
        else:
            sigma = torch.exp(self.log_sigma_sq)
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
        d_train = self.neg_log_prob(x, eval = False)
        d_train_neg = self.neg_log_prob(neg_x, eval = False)
        pos_e = d_train["neg_log_prob"]
        neg_e = d_train_neg["neg_log_prob"]
        loss = pos_e - neg_e
        baseline = (pos_e.mean() + neg_e.mean()).detach() / 2
        reg_loss = (pos_e-baseline).pow(2).mean() + (neg_e-baseline).pow(2).mean()
        loss = loss.mean()
        loss = loss + self.gamma * reg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)

        optimizer.step()

        d_logging = {"loss" : loss.item()}
        return d_logging
    
    def neg_log_prob(self, x, eval = False):
        z_star = self.encoder(x)
        z_star_detached = z_star.detach()
        x = x.view(len(x), -1) # (B, D)
        sigma = self.get_sigma(z_star) # (B,)

        bs = x.shape[0]
        D = self.x_dim
        n = self.z_dim
        # compute recon loss
        recon = self.decoder(z_star).view(len(x), -1) # (B, D)
        recon_loss = ((recon - x) ** 2).sum(dim = 1) / (2 * (sigma ** 2)) # (B,)
        sigma_loss = D * torch.log(sigma) # (B,)

        # compute log det and latent energy
        if self.normalize:
            J = spherical_jacobian_of_f(partial(self.scaled_displacement, x = x.repeat(n-1, 1)), z_star_detached, create_graph=True)
            G = J.permute(0, 2, 1)@J + self.epsilon* torch.eye(n-1).to(z_star_detached)
            latent_energy = torch.zeros(bs).to(z_star_detached)       
        else:
            J = jacobian_of_f(partial(self.scaled_displacement, x = x.repeat(n, 1)), z_star_detached, create_graph=True)
            G = J.permute(0, 2, 1)@J + torch.eye(n).to(z_star_detached)
            latent_energy = (z_star ** 2).sum(dim = 1) / 2

        log_det = torch.logdet(G) / 2 
        
        
        # compute hessian
        compute_batch_hessian = vmap(hessian(self.function_for_hessian, argnums = 0), in_dims = (0, 0))
        hess = compute_batch_hessian(z_star_detached, x) # (B, n, n)

        # compute second order loss
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )
        second_order_loss = neg_log_approx.apply(Trace_hess/2)
        neg_log_prob = (recon_loss + sigma_loss + log_det + latent_energy + second_order_loss)/D #   ++ constant_term+ second_order_loss+ second_order_loss + invariant_energy 
        # neg_log_prob[failed_index] = 0

        d_return = {"neg_log_prob": neg_log_prob,
                    "recon_loss": recon_loss,
                   "latent_energy": latent_energy,
                    "log_det": log_det,
                    "second_order_loss": second_order_loss,
                    "sigma": sigma}
        
        # if not eval:
        #     d_return["grad"] = grad
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
        