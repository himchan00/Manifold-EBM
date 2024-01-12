import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
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
    get_projection_matrix,
    get_projection_coord_rep,
    conformal_distortion_measure,
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

class EnergyAE(AE):
    def __init__(
        self, encoder, decoder, ebm, sigma, sigma_sq=1e-4, harmonic_pretrain = True,
        energy_detach = True, harmonic_detach = True, 
        conformal_detach = True, reg = None, train_sigma = True
    ):
        super(EnergyAE, self).__init__(encoder, decoder)
        self.ebm = ebm
        self.sigma_sq = sigma_sq
        self.harmonic_pretrain = harmonic_pretrain
        self.energy_detach = energy_detach
        self.harmonic_detach = harmonic_detach
        self.conformal_detach = conformal_detach
        self.reg = reg
        self.train_sigma = train_sigma
        self.sigma = sigma


    def train_energy_step(self, x, optimizer_e, pretrain = True, **kwargs):
        optimizer_e.zero_grad()
        z = self.encode(x).detach().clone()
        energy_loss, _, pos_e, neg_e, _, neg_z_sample = self.ebm.energy_loss(z)
        loss = energy_loss
        loss.backward()
        optimizer_e.step()
        return {"loss": loss.item(), "EBM/pos_e_": pos_e.mean().item(), "EBM/neg_e_": neg_e.mean().item()}
    

    def sample(self, shape, sample_step, device, replay=True, apply_noise = True):
        # sample from latent space
        # z = self.ebm.sample(shape=shape, sample_step = sample_step, device=device, replay=replay)
        z = torch.randn(shape, device=device)
        # decode
        with torch.no_grad():
            x = self.decode(z)
            if apply_noise:
                    sigma_sq = self.sigma(z)
                    x = x + torch.randn_like(x) * torch.sqrt(sigma_sq).unsqueeze(1).unsqueeze(1)
        
        return x

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

    def new_pretrain_step(self, x, optimizer_pre, **kwargs):
        optimizer_pre.zero_grad()
        z = self.encoder(x)
        x = x.view(len(x), -1)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample).view(len(x), -1)
        T, coord, J_sq = get_projection_coord_rep(self.decoder, z_sample, x - recon, create_graph=True)
        sigma_sq = self.sigma(z_sample)
        half_chan = int(z.shape[1] / 2)
        sigma_tan = sigma_sq[:, :half_chan]
        sigma_nor = sigma_sq[:, -1]
        projected_x = torch.bmm(T, x.unsqueeze(2)).squeeze(2)
        tangential_recon_loss = J_sq * (coord ** 2)
        normal_recon_loss = ((x - projected_x) ** 2).sum(dim=1)
        D = torch.prod(torch.tensor(x.shape[1:]))
        tangential_nll = (tangential_recon_loss/(2 * (sigma_tan))).sum(dim = 1) + torch.log(sigma_tan).sum(dim = 1)/2
        normal_nll = normal_recon_loss/(2 * (sigma_nor)) + (D - half_chan) * torch.log(sigma_nor)/2
        conformal_loss = conformal_distortion_measure(self.decoder, z_sample, eta=0.2, create_graph=True)
        kl_loss = self.kl_loss(z)
        loss = (tangential_nll + normal_nll + kl_loss + 1e-1 * conformal_loss)/D
        loss = loss.mean()

        loss.backward()
        optimizer_pre.step()

        return {
            "loss": loss.item(),
            "pretrain/tangential_recon_error_": tangential_recon_loss.mean().item(),
            "pretrain/normal_recon_error_": normal_recon_loss.mean().item(),
            "pretrain/kl_loss_": kl_loss.mean(),
            "pretrain/tangential_sigma_sq_": sigma_tan.mean().item(),
            "pretrain/normal_sigma_sq_": sigma_nor.mean().item(),
            "pretrain/conformal_loss_": conformal_loss.mean().item(),
            "pretrain/tangential_nll_": tangential_nll.mean().item(),
            "pretrain/normal_nll_": normal_nll.mean().item(),

        }

    def pretrain_step(self, x, optimizer_pre, **kwargs):
        optimizer_pre.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample)
        sigma_sq = self.sigma(z_sample.detach()).view(-1) # detach 안해야함을 실험을 통해서 확인
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        D = torch.prod(torch.tensor(x.shape[1:]))
        D_eff = D - self.encoder.out_chan
        nll = recon_error/(2 * (sigma_sq)) + 1 / D * torch.log(sigma_sq)/2

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss/D
        loss = loss.mean()

        loss.backward()
        optimizer_pre.step()

        return {
            "loss": loss.item(),
            "pretrain/recon_error_": recon_error.mean().item(),
            "pretrain/kl_loss_": kl_loss.mean(),
            "pretrain/sigma_sq_": sigma_sq.mean().item(),

        }
    
    def minimizer_train_step(self, x, optimizer_min, **kwargs):
        optimizer_min.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        sigma_sq = self.sigma(z).view(-1)
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        log_det_jacobian = get_log_det_jacobian_new(self.decoder, z, create_graph=True)
        energy = (z**2).sum(dim = 1) / 2
        D = torch.prod(torch.tensor(x.shape[1:]))
        loss = (recon_error/(2 * (sigma_sq)) + torch.log(sigma_sq)/2 + (energy + log_det_jacobian)/D).mean()
        loss.backward()
        optimizer_min.step()
        return {
            "loss": loss.item(),
            "minimizer/recon_error_": recon_error.mean().item(),
            "minimizer/energy_": energy.mean().item(),
            "minimizer/log_det_jacobian_": log_det_jacobian.mean().item(),
            "minimizer/sigma_sq_": sigma_sq.mean().item(),
        }
    
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        pos_x = x
        pos_z = self.encode(pos_x).detach().clone()
        neg_x = self.sample(shape=pos_z.shape, sample_step = self.ebm.sample_step, device=x.device, replay=self.ebm.replay)
        neg_z = self.encode(neg_x).detach().clone()
        pos_recon = self.decoder(pos_z)
        neg_recon = self.decoder(neg_z)
        pos_sigma_sq = self.sigma(pos_z).view(-1)
        neg_sigma_sq = self.sigma(neg_z).view(-1)
        pos_recon_error = ((pos_recon - pos_x) ** 2).view(len(pos_x), -1).mean(dim=1)
        neg_recon_error = ((neg_recon - neg_x) ** 2).view(len(neg_x), -1).mean(dim=1)
        pos_log_det_jacobian = get_log_det_jacobian_new(self.decoder, pos_z, create_graph=True)
        neg_log_det_jacobian = get_log_det_jacobian_new(self.decoder, neg_z, create_graph=True)
        pos_energy = (pos_z**2).sum(dim = 1) / 2
        neg_energy = (neg_z**2).sum(dim = 1) / 2
        D = torch.prod(torch.tensor(x.shape[1:]))
        total_pos_e = (pos_recon_error/(2 * (pos_sigma_sq)) + torch.log(pos_sigma_sq)/2 + (pos_energy + pos_log_det_jacobian)/D).mean()
        total_neg_e = (neg_recon_error/(2 * (neg_sigma_sq)) + torch.log(neg_sigma_sq)/2 + (neg_energy + neg_log_det_jacobian)/D).mean()
        loss = total_pos_e - total_neg_e
        loss.backward()
        optimizer.step()
        return {
            "loss": loss.item(),
            "AE/total_pos_e_": total_pos_e.item(),
            "AE/total_neg_e_": total_neg_e.item(),
            "AE/pos_recon_error_": pos_recon_error.mean().item(),
            "AE/neg_recon_error_": neg_recon_error.mean().item(),
            "AE/pos_log_det_jacobian_": pos_log_det_jacobian.mean().item(),
            "AE/neg_log_det_jacobian_": neg_log_det_jacobian.mean().item(),
            "AE/pos_energy_": pos_energy.mean().item(),
            "AE/neg_energy_": neg_energy.mean().item(),
            "sigma/pos_sigma_sq_": pos_sigma_sq.mean().item(), "sigma/neg_sigma_sq_": neg_sigma_sq.mean().item(),
        }
    
    
    def neg_log_prob(self, x, pretrain = False):
        D = torch.prod(torch.tensor(x.shape[1:]))
        with torch.no_grad():
            z = self.encode(x)
            recon = self.decode(z)
            # energy = self.ebm.forward_with_x(recon)/self.ebm.temperature
            if self.train_sigma:
                # sigma_sq = self.sigma.forward_with_x(recon).view(-1)
                sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            else:
                sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            energy = (z**2).sum(dim = 1) / 2
            
        # log_det_jacobian = get_log_det_jacobian_new(self.decoder, z.detach(), create_graph=False)
        log_det_jacobian = torch.zeros_like(energy)
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        neg_log_prob = recon_error/(2 * (sigma_sq)) + torch.log(sigma_sq)/2 + (energy + log_det_jacobian)/D 
        return {"neg_log_prob": neg_log_prob, "recon_error": recon_error,
                "energy": energy, "log_det_jacobian": log_det_jacobian
               }

    def visualization_step(self, dl, procedure, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        z = self.encoder(x.to(device))
        z = self.sample_latent(z)
        recon = self.decode(z)
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        if procedure == 'train_energy' or procedure == "train":
            sampled_x = self.sample(shape = z.shape, sample_step = self.ebm.sample_step, device=device, replay=self.ebm.replay, apply_noise=False)
            sampled_img = make_grid(sampled_x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        
        if z.shape[1] == 3:
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
        