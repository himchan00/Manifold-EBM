import numpy as np
import torch
import torch.nn as nn

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
    relaxed_distortion_measure
)

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

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
        self, encoder, encoder_pre, decoder, ebm, sigma_net, sigma_sq=1e-4, harmonic_pretrain = True,
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
        self.encoder_pre = encoder_pre
        self.sigma = sigma_net
        self.train_sigma = train_sigma
        
    def pretrain_step(self, x, optimizer_pre, pretrain  = True, neg_sample = False, **kwargs):
        if not pretrain:
            for params in self.decoder.parameters():
                params.requires_grad = False
            for params in self.ebm.parameters():
                params.requires_grad = False
            for params in self.sigma.parameters():
                params.requires_grad = False
        optimizer_pre.zero_grad()
        if pretrain:
            z = self.encoder_pre(x)
        else:
            z = self.encode(x)
        
        # if self.conformal_detach:
        #     z_c = z.detach().clone()
        # else:
        #     z_c = z

        # ebm training
        # energy_loss, _, pos_e, neg_e, _, neg_z_sample = self.ebm.energy_loss(z_e)
        # neg_z_sample = self.ebm.sample(shape=z_e.shape, sample_step = self.ebm.sample_step, device=z_e.device, replay=self.ebm.replay)
       
        recon = self.decode(z)

        pos_recon = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        if not pretrain:
            pos_e = self.ebm(z)
            pos_log_det_jacobian = get_log_det_jacobian(self.decoder,z, training=False, return_avg=False, create_graph=True)
            D = torch.prod(torch.tensor(x.shape[1:]))
            if self.train_sigma:
                pos_sigma_sq = self.sigma(z).view(-1)
            else:
                pos_sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            total_pos_e = ((pos_recon)/(2 * pos_sigma_sq) + torch.log(pos_sigma_sq)/2 + (pos_log_det_jacobian + pos_e/self.ebm.temperature)/D)
            loss = total_pos_e.mean()
        
        else:
            # iso_loss = relaxed_distortion_measure(self.decoder, z_c, eta=0.2)
            loss = pos_recon.mean() # + self.conformal_reg * iso_loss

        # if self.harmonic_detach:
        #     z_h = z.detach().clone()
        # else:
        #     z_h = z
        # pos_scaled_harmonic_loss = harmonic_loss(self.decoder,z_h, training=False, return_avg=False)
        # iso_loss = relaxed_volume_preserving_measure(self.decoder, z_c, eta=0.2)
        # z_sample = torch.randn_like(z)
        # z_sample = z_sample / z_sample.norm(dim=1, keepdim=True)

        # + self.conformal_detach * iso_loss  # (2 * self.sigma_sq) * (pos_scaled_harmonic_loss + pos_e/self.ebm.temperature)
        # if self.harmonic_pretrain:
        #     loss += (2 * self.sigma_sq) * pos_scaled_harmonic_loss.mean() 
        # reg_loss = ((pos_e/self.ebm.temperature)**2).mean()
        # if self.ebm.gamma is not None:  
        #     loss += self.ebm.gamma * reg_loss

        loss.backward()
        optimizer_pre.step()
        if not pretrain:
            for params in self.decoder.parameters():
                params.requires_grad = True
            for params in self.ebm.parameters():
                params.requires_grad = True
            for params in self.sigma.parameters():
                params.requires_grad = True
            
        if neg_sample:
            return {"loss": loss.item(), "AE/pretrain_neg_recon_": pos_recon.mean().item(),}
        else:
            return {"loss": loss.item(), "AE/pretrain_pos_recon_": pos_recon.mean().item(), 
                    # "AE/pos_scaled_harmonic_loss_": pos_scaled_harmonic_loss.mean().item(),
                    # "EBM/pos_e_": pos_e.mean().item(),
                # "AE/sin_sq_loss_": sin_sq_loss.item(),
                # "AE/iso_loss_": iso_loss.item()
                    }
    
    def train_energy_step(self, x, optimizer_e, pretrain = True, **kwargs):
        optimizer_e.zero_grad()
        if pretrain:
            z = self.encoder_pre(x).detach().clone()
        else:
            z = self.encode(x).detach().clone()
        energy_loss, _, pos_e, neg_e, _, neg_z_sample = self.ebm.energy_loss(z)
        loss = energy_loss
        loss.backward()
        optimizer_e.step()
        return {"loss": loss.item(), "EBM/pos_e_": pos_e.mean().item(), "EBM/neg_e_": neg_e.mean().item()}
    

    def train_step(self, x, optimizer, **kwargs):

        for params in self.encoder.parameters():
            params.requires_grad = False


        optimizer.zero_grad()
        z = self.encode(x)
        # if self.conformal_detach:
        #     z_c = z.detach().clone()
        # else:
        #     z_c = z

        # ebm training
        # energy_loss, _, pos_e, neg_e, _, neg_z_sample = self.ebm.energy_loss(z_e)
        neg_z_sample = self.ebm.sample(shape=z.shape, sample_step = self.ebm.sample_step, device=z.device, replay=self.ebm.replay)
        # neg_z_sample = torch.randn_like(z)
        
        # neg_x = self.decode(neg_z_sample).detach().clone() + torch.randn_like(x) * (self.sigma_sq**(1/2))
        if self.train_sigma:
            sigma_sq = (self.sigma(neg_z_sample).detach().clone())
            # sigma_sq = (self.sigma(neg_z_sample.detach().clone()))
        else:
            sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
        neg_x = self.decode(neg_z_sample).detach().clone() + torch.randn_like(x) * torch.sqrt(sigma_sq)
        neg_z = self.encode(neg_x)

        recon = self.decode(z)
        recon_neg = self.decode(neg_z)

        pos_e = self.ebm(z)
        neg_e = self.ebm(neg_z)
        pos_log_det_jacobian = get_log_det_jacobian(self.decoder,z, training=False, return_avg=False, create_graph=True)
        neg_log_det_jacobian = get_log_det_jacobian(self.decoder, neg_z, training=False, return_avg=False, create_graph=True)
        
        D = torch.prod(torch.tensor(x.shape[1:]))

        pos_recon = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        neg_recon = ((recon_neg - neg_x) ** 2).view(len(neg_x), -1).mean(dim=1)
        
        if self.train_sigma:
            pos_sigma_sq = self.sigma(z).view(-1)
            neg_sigma_sq = self.sigma(neg_z).view(-1)
        else:
            pos_sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            neg_sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            
        total_pos_e = ((pos_recon)/(2 * pos_sigma_sq) + torch.log(pos_sigma_sq)/2 + (pos_e/self.ebm.temperature + pos_log_det_jacobian)/D) # )/100 #++ pos_log_det_jacobian/D (+pos_e/self.ebm.temperature)/D
        total_neg_e = ((neg_recon)/(2 * neg_sigma_sq) + torch.log(neg_sigma_sq)/2 + (neg_e/self.ebm.temperature + neg_log_det_jacobian)/D) # +()/100#+(neg_log_det_jacobian+ neg_e/self.ebm.temperature)/D)+neg_log_det_jacobian/D
        # iso_loss = relaxed_volume_preserving_measure(self.decoder, z_c, eta=0.2)
        # e_avg = (total_pos_e.mean() + total_neg_e.mean()).detach().clone()/2
        if self.reg is not None:
            reg_loss = self.reg * ((total_pos_e**2).mean() + (total_neg_e**2).mean())
        else:
            reg_loss = 0
        loss = total_pos_e.mean() - total_neg_e.mean() + reg_loss
         #+ self.conformal_reg * iso_loss
        #  + self.conformal_detach * iso_loss # (2 * self.sigma_sq) * (pos_scaled_harmonic_loss + pos_e/self.ebm.temperature)
        # reg_loss = (2 * self.sigma_sq) * ((pos_e**2).mean() + (neg_e**2).mean())/self.ebm.temperature
        # if self.ebm.gamma is not None:  
        #     loss += self.ebm.gamma * reg_loss

        loss.backward()
        optimizer.step()

        for params in self.encoder.parameters():
            params.requires_grad = True


        return {"loss": loss.item(), #"reg_loss": reg_loss.item(), 
                "AE/total_pos_e_": total_pos_e.mean().item(), "AE/total_neg_e_": total_neg_e.mean().item(),
                "AE/pos_recon_": pos_recon.mean().item(), 
                "AE/pos_log_det_jacobian_": pos_log_det_jacobian.mean().item(),
                "EBM/pos_e_": pos_e.mean().item(), 
                "AE/neg_recon_": neg_recon.mean().item(), 
                "AE/neg_log_det_jacobian_": neg_log_det_jacobian.mean().item(),
                "EBM/neg_e_": neg_e.mean().item(),
                "sigma/pos_sigma_sq_": (pos_sigma_sq).mean().item(), "sigma/neg_sigma_sq_": (neg_sigma_sq).mean().item(),
                #"AE/sin_sq_loss_": sin_sq_loss.item(),
                # "AE/iso_loss_": iso_loss.item()
                
        }, neg_x.detach().clone()
    
    def sample(self, shape, sample_step, device, replay=True, apply_noise = False):
        # sample from latent space
        z = self.ebm.sample(shape=shape, sample_step = sample_step, device=device, replay=replay)
        # decode
        with torch.no_grad():
            x = self.decode(z)
            if apply_noise:
                    sigma_sq = self.sigma(z)
                    x = x + torch.randn_like(x) * torch.sqrt(sigma_sq)
        
        return x
    
    def neg_log_prob(self, x, pretrain = False):
        D = torch.prod(torch.tensor(x.shape[1:]))
        with torch.no_grad():
            if pretrain:
                z = self.encoder_pre(x)
            else:
                z = self.encode(x)
            energy = self.ebm(z)
            if self.train_sigma:
                sigma_sq = self.sigma(z).view(-1)
            else:
                sigma_sq = torch.tensor(self.sigma_sq).to(z.device)
            # energy = (z**2).sum(dim = 1) / 2
            recon = self.decode(z)
        log_det_jacobian = get_log_det_jacobian(self.decoder, z.detach(), return_avg= False, training=False, create_graph=False)
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        neg_log_prob = recon_error/(2 * (sigma_sq)) + torch.log(sigma_sq)/2 + (energy/self.ebm.temperature + log_det_jacobian)/D 
        return {"neg_log_prob": neg_log_prob, "recon_error": recon_error,
                "energy": energy, "log_det_jacobian": log_det_jacobian
               }

    def visualization_step(self, dl, procedure, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        z_pre = self.encoder_pre(x.to(device))
        z = self.encoder(x.to(device))
        recon_pre = self.decode(z_pre)
        recon = self.decode(z)
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img_pre = make_grid(recon_pre.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        if procedure == 'train_energy' or procedure == "train":
            sampled_x = self.sample(shape = z.shape, sample_step = self.ebm.sample_step, device=device, replay=self.ebm.replay)
            sampled_img = make_grid(sampled_x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        
        if z.shape[1] == 3:
            # 2d graph (latent sapce)
            num_points_for_each_class = 200
            num_points_neg_samples = 200
            label_unique = torch.unique(dl.dataset.targets)
            z_ = []
            z_pre = []
            label_ = []
            for label in label_unique:
                temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class]
                temp_z = self.encoder(temp_data.to(device))
                temp_z_pre = self.encoder_pre(temp_data.to(device))

                z_.append(temp_z)
                z_pre.append(temp_z_pre)
                label_.append(label.repeat(temp_z.size(0)))


            z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
            z_pre = torch.cat(z_pre, dim=0).detach().cpu().numpy()
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
            plt.title('Encoded points via encoder_pre')
            for label in label_unique:
                label = label.item()
                ax.scatter(z_pre[label_==label,0], z_pre[label_==label,1], z_pre[label_==label, 2], s = 5, c=color_[label_==label]/255, label=label)
            if procedure == 'train_energy' or procedure == "train":    
                ax.scatter(neg_z[:,0], neg_z[:,1], neg_z[:, 2], c='black', marker='x')
            plt.legend()
            plt.close()
            f_np_pre = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]

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
                    'recon_pre@': torch.clip(recon_img_pre, min=0, max=1),
                    'sampled@': torch.clip(sampled_img, min=0, max=1),
                    'latent_space@': f_np,
                    'latent_space_pre@': f_np_pre,
                    'latent_space_energy@': f_np_energy,
                    'latent_space_mcmc@': f_np_mcmc
                }
            else:
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon_pre@': torch.clip(recon_img_pre, min=0, max=1),
                    'latent_space_pre@': f_np_pre
                }
        else:
            if procedure == 'train_energy' or procedure == "train":  
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon@': torch.clip(recon_img, min=0, max=1),
                    'recon_pre@': torch.clip(recon_img_pre, min=0, max=1),
                    'sampled@': torch.clip(sampled_img, min=0, max=1),
                }
            else:
                return {
                    'input@': torch.clip(x_img, min=0, max=1),
                    'recon_pre@': torch.clip(recon_img_pre, min=0, max=1),
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
        