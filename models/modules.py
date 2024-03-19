import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)
    
class FC_image(nn.Module):
    def __init__(
        self,
        in_chan=784,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        out_chan_num=1,
    ):
        super(FC_image, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_chan_num = out_chan_num
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for i_layer, [n_hidden, act] in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)
        self.feature_net = nn.Sequential(*l_layer[:-1])

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.in_chan)
            out = self.net(x)
        else:
            dim = int(np.sqrt(self.out_chan / self.out_chan_num))
            out = self.net(x)
            out = out.reshape(-1, self.out_chan_num, dim, dim)
        return out

    def get_feature(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.in_chan)
        x = self.feature_net(x)
        return x

class FC_for_decoder_and_sigma(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        self.fc1 = nn.Linear(z_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        self.fc4_d = nn.Linear(1024, 1024)
        self.fc5_d = nn.Linear(1024, 1024)
        self.fc6_d = nn.Linear(1024, x_dim)

        self.fc4_s = nn.Linear(1024, 1024)
        self.fc5_s = nn.Linear(1024, 1024)
        self.fc6_s = nn.Linear(1024, 1)

    def forward(self, z):
        x = self.fc1(z)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        x_d = self.fc4_d(x)
        x_d = F.relu(x_d)
        x_d = self.fc5_d(x_d)
        x_d = F.relu(x_d)
        x_d = self.fc6_d(x_d)
        x_d = torch.sigmoid(x_d)
        dim = int(np.sqrt(x_d.shape[-1]))
        x_d = x_d.reshape(-1, 1, dim, dim)
        return x_d

    def sigma(self, z):
        x = self.fc1(z)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        x_s = self.fc4_s(x)
        x_s = F.relu(x_s)
        x_s = self.fc5_s(x_s)
        x_s = F.relu(x_s)
        x_s = self.fc6_s(x_s)
        x_s = torch.exp(x_s)
        return x_s

class FC_for_encoder_and_sigma(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        self.fc1 = nn.Linear(x_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        self.fc4_e = nn.Linear(1024, 1024)
        self.fc5_e = nn.Linear(1024, 1024)
        self.fc6_e = nn.Linear(1024, z_dim)

        self.fc4_s = nn.Linear(1024, 1024)
        self.fc5_s = nn.Linear(1024, 1024)
        self.fc6_s = nn.Linear(1024, 1)

        # self.sig_min = 1e-3
        # self.sig_max = 10
    def forward(self, z):
        if len(z.size()) == 4:
            z = z.view(z.size(0), -1)
        x = self.fc1(z)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        x_e = self.fc4_e(x)
        x_e = F.relu(x_e)
        x_e = self.fc5_e(x_e)
        x_e = F.relu(x_e)
        x_e = self.fc6_e(x_e)
    
        return x_e
    
    def sigma(self, z):
        if len(z.size()) == 4:
            z = z.view(z.size(0), -1)
        x = self.fc1(z)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        x_s = self.fc4_s(x)
        x_s = F.relu(x_s)
        x_s = self.fc5_s(x_s)
        x_s = F.relu(x_s)
        x_s = self.fc6_s(x_s)
        # x_s = torch.sigmoid(x_s)
        # x_s = torch.exp(torch.log(self.sig_min) + (torch.log(self.sig_max) - torch.log(self.sig_min)) * x_s)
        x_s = torch.exp(x_s)
        return x_s

class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""

    def __init__(self, net, sigma=1):
        super().__init__()
        self.net = net
        sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        self.register_parameter("sigma", sigma)

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - decoder_out) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        return loglik

    def forward(self, z):
        return self.net(z)

    def sample(self, z):
        x_hat = self.net(z)
        return x_hat + torch.randn_like(x_hat) * self.sigma


class normalized_net(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x / torch.norm(x, dim=1, keepdim=True)

class sigma_net_normalizer(nn.Module):
    def __init__(self, net, min_sigma_sq, max_sigma_sq):
        super().__init__()
        self.net = net
        self.min = torch.log(torch.tensor(min_sigma_sq))
        self.max = torch.log(torch.tensor(max_sigma_sq))
        self.min.requires_grad = False
        self.max.requires_grad = False
        
    def forward(self, x):
        x = self.net(x)

        x = torch.exp(self.min + (self.max - self.min) * x)
        return x
"""
ConvNet for (1, 28, 28) image, following architecture in (Ghosh et al., 2019)
"""

class ConvNet28(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet28, self).__init__()
        # self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

        layers = [#self.conv1,
                #   nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5,
                  nn.ReLU(),
                  self.conv6,]
        if self.out_activation is not None:
            layers.append(self.out_activation)


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)


class DeConvNet28(nn.Module):
    def __init__(
        self,
        in_chan=1,
        out_chan=1,
        nh=32,
        out_activation="sigmoid",
        activation="relu",
    ):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet28, self).__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(
            nh * 32, nh * 16, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            nh * 16, nh * 8, kernel_size=2, stride=2, padding=1, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        if activation == "relu":
            act = nn.ReLU
        elif activation == "leakyrelu":
            def act():
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError

        layers = [
            self.fc1,
            act(),
            self.conv1,
            act(),
            self.conv2,
            act(),
            self.conv3,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.net(x)
        return x

class energy_net(nn.Module):
    def __init__(self, encoder, decoder, nh_mlp=512, n_layers = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        l_layer = []
        for _ in range(n_layers-1):
            l_layer.append(nn.Linear(nh_mlp, nh_mlp))
            l_layer.append(nn.ReLU())
        l_layer.append(nn.Linear(nh_mlp, 1))
        self.fc_nets = nn.Sequential(*l_layer)

    def forward_for_energy(self, z):
        x_bar = self.decoder(z)
        x = self.encoder.get_feature(x_bar, grad=True).squeeze(2).squeeze(2)
        x = self.fc_nets(x)
        return x
    def forward(self, z, grad = True):
        with torch.no_grad():
            x_bar = self.decoder(z)
        x = self.encoder.get_feature(x_bar, grad=grad).squeeze(2).squeeze(2)
        x = self.fc_nets(x)
        return x
    def forward_with_x(self, x, grad=True):
        x = self.encoder.get_feature(x, grad=grad).squeeze(2).squeeze(2)
        x = self.fc_nets(x)
        return x

class sigma_net(nn.Module):
    def __init__(self, net, minimizer_target, decoder_target, min_sigma_sq = 1e-4, max_sigma_sq = 1e-2):
        super().__init__()
        # self.net = nn.Linear(1024, 1)
        self.net = net
        self.decoder_target = decoder_target
        self.minimizer_target = minimizer_target
        self.min = torch.log(torch.tensor(min_sigma_sq))
        self.max = torch.log(torch.tensor(max_sigma_sq))
        self.min.requires_grad = False
        self.max.requires_grad = False

    def forward(self, z):
        x = self.decoder_target(z)
        x = x.view(x.size(0), -1)
        x = self.minimizer_target.get_feature(x)
        x = self.net(x)
        x = torch.sigmoid(x)
        x = torch.exp(self.min + (self.max - self.min) * x)
        return x

    def forward_with_x(self, x):
        if len(x.size()) == 4:
            x = x.view(x.size(0), -1)
        x = self.minimizer_target.get_feature(x)
        x = self.net(x)
        x = torch.sigmoid(x)
        x = torch.exp(self.min + (self.max - self.min) * x)
        return x

class new_sigma_net(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, z):
        x = self.net(z)
        x = torch.exp(x)
        return x

class ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

        layers = [self.conv1,
                  nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5,
                  nn.ReLU(),
                  self.conv6,]
        if self.out_activation is not None:
            layers.append(self.out_activation)


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)

    def get_feature(self, x, grad=False):
        if grad:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.max1(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            x = F.relu(x)
            x = self.max2(x)
            x = self.conv5(x)
            x = F.relu(x)
        else:
            with torch.no_grad():
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.max1(x)
                x = self.conv3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = F.relu(x)
                x = self.max2(x)
                x = self.conv5(x)
                x = F.relu(x)
        return x



class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear',
                 use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)

        self.fc1 = nn.Linear(nh * 8 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, in_chan + 1)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation) 

    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def sigma(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x) # (bs, 64, 12, 12)
        x = F.relu(x)
        x = x.view(x.size(0), -1) # (bs, 64*12*12)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.exp(x)
        return x

class DeConvNet3(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear',
                 use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet3, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation) 


    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x) # (bs, 64, 12, 12)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x) # (bs, 32, 26, 26)
        x = F.relu(x)
        x = self.conv5(x) # (bs, 1, 28, 28)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
    
    def get_feature(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)

        return x