import numpy as np
import torch
import torch.nn as nn
import copy
from functools import partial
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from functorch import hessian, vmap
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse
from geometry import (
    get_pullbacked_Riemannian_metric,
    get_flattening_scores,
    jacobian_of_f,
    relaxed_distortion_measure,
    curvature_reg
)


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
        self.encoder_target = copy.deepcopy(encoder)
        self.decoder = decoder
        self.z_dim = z_dim
        self.max_latent_variance = torch.tensor(max_latent_variance)
        self.log_sig_paral = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.log_sig_vert = nn.Parameter(torch.zeros(1, requires_grad=True))


    def sample(self, batch_size, device):

        z = torch.randn((batch_size, self.z_dim), device=device)
        # decode
        with torch.no_grad():
            x = self.decoder(z)
        
        return x
    
    def recon_loss(self, x, z):
        """
        x : (B, D)
        z : (B, n)
        """
        D = x.shape[1]
        n = z.shape[1]
        # x_star = self.decoder(z).view(-1, D) # (B, D)
        x_star, sigma = self.decoder.forward_with_sigma(z)
        x_star = x_star.view(-1, D)
        # sigma_paral = torch.exp(self.log_sig_paral)
        # sigma_vert = torch.exp(self.log_sig_vert)
        sigma_paral = sigma[:, 0] # (B,)
        sigma_vert = sigma[:, 1] # (B,)

        J = jacobian_of_f(self.decoder, z, create_graph=True)
        G = J.permute(0, 2, 1) @ J
        delta_x = (x - x_star) # (B, D)
        d_sq = (delta_x ** 2).sum(dim = 1) # (B,)
        t = J.permute(0, 2, 1) @ delta_x.unsqueeze(-1) # (B, n, 1)
        d_proj_sq = (t.permute(0, 2, 1) @ torch.linalg.solve(G, t)).squeeze() # (B,)
        
        sigma_term = n * torch.log(sigma_paral) + (D-n) * torch.log(sigma_vert) # (B,)
        return d_sq/(2 * sigma_vert ** 2) +  d_proj_sq * (1/(2 * sigma_paral ** 2) - 1/(2 * sigma_vert ** 2)) + sigma_term
    
    def get_proj_x(self, x, z):
        """
        x : (B, D)
        z : (B, n)
        """
        D = x.shape[1]
        n = z.shape[1]
        x_star = self.decoder(z).view(-1, D)
        J = jacobian_of_f(self.decoder, z, create_graph=True)
        G = J.permute(0, 2, 1) @ J
        delta_x = x_star - x
        t = J.permute(0, 2, 1) @ delta_x.unsqueeze(-1)
        proj_x = (J @ torch.linalg.solve(G, t)).squeeze()
        return proj_x


    def get_scaled_x_paral_and_x_vert(self, x, z):
        """
        x : (B, D)
        z : (B, n)
        """
        D = x.shape[1]
        n = z.shape[1]
        x_star, sigma = self.decoder.forward_with_sigma(z)
        sigma_paral = sigma[:, 0] # (B,)
        sigma_vert = sigma[:, 1] # (B,)
        x_star = x_star.view(-1, D)
        J = jacobian_of_f(self.decoder, z, create_graph=True)
        G = J.permute(0, 2, 1) @ J
        delta_x = x_star - x
        t = J.permute(0, 2, 1) @ delta_x.unsqueeze(-1)
        proj_x = (J @ torch.linalg.solve(G, t)).squeeze()
        vert_x = delta_x - proj_x
        scaled_proj_x = proj_x / sigma_paral.unsqueeze(1) # (B, D)
        scaled_vert_x = vert_x / sigma_vert.unsqueeze(1) # (B, D)
        return torch.cat([scaled_proj_x, scaled_vert_x], dim = 1)

    
    def function_for_hessian(self, z, x):
        """
        caculate d.detach().T @ d
        x : (D)
        z : (n)
        sigma : (1)
        returns (,)
        """
        x = x.unsqueeze(0) # (1, D)
        D = x.shape[1]
        z = z.unsqueeze(0) # (1, n)
        x_star = self.decoder(z).view(-1, D) # (1, D)
        delta_x = (x - x_star) # (1, D)
        out = (delta_x ** 2).sum(dim = 1)/2
        return out.squeeze()
    
        
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        d_train = self.neg_log_prob(x)
        loss = d_train["neg_log_prob"]
        loss = loss.mean()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)

        optimizer.step()
        
        d_return = {"loss": loss.item()}
        for key, val in d_train.items():
            key = "train/" + key + "_"
            d_return[key] = val.mean().item()
        return d_return

    def neg_log_prob(self, x, n_eval = 1, train = True):
        z_star  = self.encoder(x) # (B, n)
        x = x.view(len(x), -1) # (B, D)

        bs = x.shape[0]
        D = x.shape[1]
        n = z_star.shape[1]
        


        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        sigma = self.decoder.sigma(z_star) # (B, 2)
        sigma_paral = sigma[:,0]
        sigma_vert = sigma[:,1]
        dsigma_paral = torch.autograd.grad(sigma_paral.sum(), z_star, create_graph=True)[0] # (B, n)
        dsigma_vert = torch.autograd.grad(sigma_vert.sum(), z_star, create_graph=True)[0] # (B, n)
        sig_term = ((n * dsigma_paral.unsqueeze(2) @ dsigma_paral.unsqueeze(1))/(sigma_paral**2).unsqueeze(1).unsqueeze(2)\
                 + (D-n) * dsigma_vert.unsqueeze(2) @ dsigma_vert.unsqueeze(1)/(sigma_vert**2).unsqueeze(1).unsqueeze(2))/2 # (B, n, n)
        G = J.permute(0, 2, 1) @ J
        Precision = G/(sigma_paral**2).unsqueeze(1).unsqueeze(2) + sig_term + torch.eye(n).to(z_star).repeat(bs, 1, 1)

        # Find minimum eigenvalue
        eigvals = torch.linalg.eigvalsh(Precision)
        print("Precision eigenvalues:")
        print(eigvals[0, :])
        # eigvals_min = torch.min(eigvals, dim = 1).values # (B,)
        # delta = torch.maximum(1/self.max_latent_variance- eigvals_min, torch.zeros_like(eigvals_min))
        # Precision = Precision + torch.eye(n).to(z_star).repeat(bs, 1, 1) * delta.unsqueeze(1).unsqueeze(2)
        # eigvals = eigvals + delta.unsqueeze(1)
        # posterior sampling
        L = torch.linalg.cholesky(Precision, upper = True)
        eps = torch.randn((n_eval, bs, n)).to(z_star)
        z_sample = z_star.unsqueeze(0) + torch.linalg.solve_triangular(L.unsqueeze(0), eps.unsqueeze(-1), upper = True).squeeze(-1) # (n_eval, B, n)

        # compute recon loss
        x = x.repeat(n_eval, 1) # (n_eval * B, D)
        recon_loss = self.recon_loss(x, z_sample.view(-1, n)) # (n_eval * B,)
        recon_loss = recon_loss.view(n_eval, bs).mean(dim = 0) # (B,)
        

        # compute latent energy
        latent_energy = (z_star ** 2).sum(dim = 1) / 2 + (1 / eigvals).sum(dim = 1)/2

        # compute log det
        logdet_loss = torch.log(eigvals).sum(dim = 1)/2 # (B,)

        # compute curvature regularization
        # if train:
        #     curvature_loss = curvature_reg(self.decoder, z_star, create_graph=True, eta = 0.2)
        # else:
        #     curvature_loss = torch.zeros_like(recon_loss)

        neg_log_prob = (recon_loss + latent_energy + logdet_loss )/D # + 10*curvature_loss

        d_return = {"neg_log_prob": neg_log_prob,
                    "recon_loss": recon_loss,
                   "latent_energy": latent_energy,
                    "log_det": logdet_loss,
                    "sigma_paral": sigma_paral,
                    "sigma_vert": sigma_vert,
                    # "curvature_loss": curvature_loss,
                    }

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
        