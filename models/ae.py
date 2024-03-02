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
                    # sigma = self.minimizer.sigma(x)
                    sigma = torch.exp(self.log_sigma_sq)
                    x = x + torch.randn_like(x) * sigma # .unsqueeze(1).unsqueeze(2)
        
        return x
    
    def energy_function(self, x, z):
        sigma = torch.exp(self.log_sigma_sq)
        x_star = self.decoder(z).view(-1) # (D)
        recon_loss = (x - x_star).pow(2).sum() / (2 * (sigma ** 2))
        latent_energy_loss = (z ** 2).sum() / 2
        energy = recon_loss + latent_energy_loss
        return energy.squeeze()
    
    def half_riemannian_metric(self, z, create_graph=True):
        J = jacobian_of_f(self.decoder, z, create_graph=create_graph)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z.shape[1]).to(z)
        return G/2

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
        s = 1.0
        sigma_detached = sigma.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        grad_loss = Trace_grad * (radius ** 2) / (2*n+4)
        hess_loss = Trace_hess * (radius ** 2) / (2*n+4)

       # integration area part
        volume_loss = - c_sq * (radius ** 2) * n * (n+4)/ (8)
        moment_loss = - (grad.unsqueeze(1) @ v.unsqueeze(2)).squeeze() * n * (radius ** 2) /(2)
        
        second_order_loss = grad_loss + hess_loss + volume_loss + moment_loss
        
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
                    "AE/log_det_loss_neg_": log_det_loss.mean().item(),
                    "AE/energy_loss_neg_": energy_loss.mean().item(),
                    "AE/grad_loss_neg_": grad_loss.mean().item(),
                    "AE/hess_loss_neg_": hess_loss.mean().item(),
                    "AE/volume_loss_neg_": volume_loss.mean().item(),
                    "AE/moment_loss_neg_": moment_loss.mean().item(),
                    'AE/second_order_loss_neg_': second_order_loss.mean().item(),
                    "AE/sigma_neg_": sigma.mean().item(),
                    "AE/c_neg_": c.mean().item()}
        else:
            return {"loss": loss.item(), 
                    "AE/log_det_loss_": log_det_loss.mean().item(),
                    "AE/energy_loss_": energy_loss.mean().item(),
                    "AE/grad_loss_": grad_loss.mean().item(),
                    "AE/hess_loss_": hess_loss.mean().item(),
                    "AE/volume_loss_": volume_loss.mean().item(),
                    "AE/moment_loss_": moment_loss.mean().item(),
                    'AE/second_order_loss_': second_order_loss.mean().item(),
                    "AE/sigma_": sigma.mean().item(),
                    "AE/c_": c.mean().item()}
    
    def train_step(self, x, neg_x, optimizer, **kwargs):
        optimizer.zero_grad()
        s = 1.0

        def energy_function(decoder, x, z, sigma):
            """
            decoder : decoder network
            x : input data (D)
            z : latent variable (n)
            sigma : noise level (1)

            return : energy value ()
            """
            x_star = decoder(z).view(-1) # (D)
            recon_loss = (x - x_star).pow(2).sum() / (2 * (sigma ** 2))
            latent_energy_loss = (z ** 2).sum() / 2
            energy = recon_loss + latent_energy_loss
            return energy.squeeze()
        
        def half_riemannian_metric(decoder, z, create_graph=True):
            J = jacobian_of_f(decoder, z, create_graph=create_graph)
            G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(z.shape[1]).to(z)
            return G/2
        
        z_star = self.minimizer(x) # (B, n)

        x = x.view(len(x), -1) # (B, D)
        sigma = torch.exp(self.log_sigma_sq)
        
        D = x.shape[1]
        n = z_star.shape[1]

        # compute energy
        compute_energy = vmap(energy_function, in_dims = (None, 0, 0, None))
        energy = compute_energy(self.decoder, x, z_star, sigma) # (B, )

        # gradient of energy
        compute_grad = vmap(jacrev(energy_function, argnums = 2), in_dims = (None, 0, 0, None))
        grad = compute_grad(self.decoder, x, z_star, sigma) # (B, n)

        # hessian of energy
        compute_batch_hessian = vmap(hessian(energy_function, argnums = 2), in_dims = (None, 0, 0, None))
        hess = compute_batch_hessian(self.decoder, x, z_star, sigma) # (B, n, n)
        

        J = jacobian_of_f(self.decoder, z_star, create_graph=True)
        G = J.permute(0, 2, 1)@J + 1e-3* torch.eye(n).to(z_star)  # (B, n, n)
        sqrt_G = torch.linalg.cholesky(G, upper=True)

        n_hat = torch.randn_like(z_star)
        n_hat = n_hat / n_hat.norm(dim=1, keepdim=True)
        v = torch.linalg.solve(sqrt_G, n_hat.unsqueeze(2)).squeeze() # (B, n)
        _, C = torch.autograd.functional.jvp(partial(half_riemannian_metric, self.decoder), z_star, v, create_graph=True)
        c = (v.unsqueeze(1) @ C @ v.unsqueeze(2)).squeeze() # (B,)
        c_sq = c ** 2

        # energy_loss
        energy_loss = energy # (B, )

        # log det loss
        log_det_loss = torch.logdet(G) / 2 # (B, )
        log_det_loss[torch.isnan(log_det_loss)] = 0
        log_det_loss[torch.isinf(log_det_loss)] = 0

        # sigma loss
        sigma_loss = (D-n) * torch.log(sigma) # (B, )

        # second order loss
        # intrgrand part
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        grad_loss = Trace_grad * (s ** 2) * (sigma.detach() ** 2) / (2*n+4)
        hess_loss = Trace_hess * (s ** 2) * (sigma.detach() ** 2) / (2*n+4)

       # integration area part
        volume_loss = - c_sq * (s ** 2) * (sigma.detach() ** 2) * n * (n+4)/ (8)
        moment_loss = - (grad.unsqueeze(1) @ v.unsqueeze(2)).squeeze() * n * (s ** 2) * (sigma.detach() ** 2) /(2)
        
        second_order_loss = grad_loss + hess_loss + volume_loss + moment_loss

        loss  = (energy_loss + log_det_loss + sigma_loss + second_order_loss)/D 
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.minimizer.parameters(), 0.1)

        optimizer.step()
        return {"loss": loss.item(),
                "AE/pos_e_": pos_e.mean().item(),
                "AE/neg_e_": neg_e.mean().item(),
                "AE/recon_loss_": recon_loss.mean().item(),
                "AE/recon_loss_neg_": recon_loss_neg.mean().item(),
                "AE/log_det_loss_": log_det_loss.mean().item(),
                "AE/log_det_loss_neg_": log_det_loss_neg.mean().item(),
                # "AE/energy_loss_": energy_loss.mean().item(),
                # "AE/energy_loss_neg_": energy_loss_neg.mean().item(),
                "AE/sigma_": sigma.mean().item(),
                "AE/sigma_neg_": sigma_neg.mean().item()
        }

    
    def neg_log_prob(self, x, pretrain = False):    
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
        s = 1.0
        sigma_detached = sigma.detach()
        radius = s * sigma_detached
        Trace_grad = (- grad.unsqueeze(1) @ torch.linalg.solve(G, grad.unsqueeze(2))).squeeze() # (B, )
        Trace_hess = vmap(torch.trace)(torch.linalg.solve(G, hess)) # (B, )

        grad_loss = Trace_grad * (radius ** 2) / (2*n+4)
        hess_loss = Trace_hess * (radius ** 2) / (2*n+4)

       # integration area part
        volume_loss = - c_sq * (radius ** 2) * n * (n+4)/ (8)
        moment_loss = - (grad.unsqueeze(1) @ v.unsqueeze(2)).squeeze() * n * (radius ** 2) /(2)
        
        # constant term
        constant_term = -n/2 * torch.log(torch.tensor(np.pi)) + D/2 * torch.log(2*torch.tensor(np.pi))\
              + torch.lgamma(torch.tensor(n/2+1)) - n *torch.log(radius)

        neg_log_prob  = (energy_loss + log_det_loss + sigma_loss + grad_loss \
                         + hess_loss + volume_loss + moment_loss + constant_term)

        return {"neg_log_prob": neg_log_prob,
                'log_det_loss': log_det_loss,
                'energy_loss': energy_loss,
                'sigma_loss': sigma_loss,
                'grad_loss': grad_loss,
                'hess_loss': hess_loss,
                'volume_loss': volume_loss,
                'moment_loss': moment_loss}

    # def neg_log_prob(self, x, pretrain = False):    
    #     with torch.no_grad():
    #         z_star = self.minimizer(x) # (B, n)
    #         x = x.view(len(x), -1) # (B, D)

    #         D = x.shape[1]
    #         n = z_star.shape[1]

    #         x_star = self.decoder(z_star).view(len(x), -1) # (B, D)
    #         J = jacobian_of_f(self.decoder, z_star, create_graph=True) # (B, D, n)
    #         sigma = self.decoder.sigma(z_star).squeeze() # (B,)
    #         J_eff = J/(sigma.unsqueeze(1).unsqueeze(2)) # (B, D, n)
    #         G_eff = J_eff.permute(0, 2, 1)@J_eff # (B, n, n)
            

    #         pseudo_metric = torch.eye(n).to(z_star) + G_eff # (B, n, n)
    #         recon_loss = (x - x_star).pow(2).sum(dim=1) / (2 * (sigma**2)) # (B, )
    #         z_reg_loss = (z_star.unsqueeze(1) @ G_eff @ torch.linalg.pinv(pseudo_metric) @ z_star.unsqueeze(2)).squeeze() / 2 # (B, )
            
    #         log_det_loss = torch.logdet(pseudo_metric) / 2 # (B, )

    #         sigma_loss = D * torch.log(sigma) # (B, )
    #         costant_term = (D+n) * torch.log(torch.tensor(2*np.pi)) / 2
    #         neg_log_prob = (recon_loss + z_reg_loss + log_det_loss + sigma_loss + costant_term)

    #     return {"neg_log_prob": neg_log_prob,
    #             "recon_loss": recon_loss,
    #             'z_reg_loss': z_reg_loss,
    #             'log_det_loss': log_det_loss}
    

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
        