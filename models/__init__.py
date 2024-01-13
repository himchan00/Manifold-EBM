import os
from omegaconf import OmegaConf
import torch

from models.ae import (
    VAE,
    IRVAE,
    EnergyAE
)

from models.modules import (
    FC_vec,
    FC_image,
    IsotropicGaussian,
    ConvNet28,
    DeConvNet28,
    ConvNet2FC,
    DeConvNet2,
    DeConvNet3,
)

def get_net(in_dim, out_dim, **kwargs):
    nh = kwargs.get("nh", 8)
    out_activation = kwargs.get("out_activation", "linear")

    if kwargs["arch"] == "conv2fc":
        nh_mlp = kwargs["nh_mlp"]
        net = ConvNet2FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
        )

    elif kwargs["arch"] == "deconv2":
        net = DeConvNet3(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_image":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        out_chan_num = kwargs["out_chan_num"]
        net = FC_image(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            out_chan_num=out_chan_num
        )
    elif kwargs["arch"] == "conv28":
        nh_mlp = 1024
        net = ConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "dconv28":
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = DeConvNet28(
            in_chan=in_dim,
            out_chan=out_dim,
            activation=activation,
            out_activation=out_activation
        )
    return net

def get_ae(**model_cfg):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    min_sigma_sq = model_cfg["min_sigma_sq"]
    max_sigma_sq = model_cfg["max_sigma_sq"]
    if arch == "vae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = VAE(encoder, IsotropicGaussian(decoder))
    elif arch == "irvae":
        metric = model_cfg.get("metric", "identity")
        iso_reg = model_cfg.get("iso_reg", 1.0)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = IRVAE(encoder, IsotropicGaussian(decoder), iso_reg=iso_reg, metric=metric)
    elif arch == "eae":
        from models.modules import normalized_net, energy_net, sigma_net, new_sigma_net, FC_for_decoder_and_sigma
        encoder = get_net(in_dim=x_dim, out_dim=z_dim*2, **model_cfg["encoder"])
        # decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        decoder = FC_for_decoder_and_sigma(z_dim=z_dim, x_dim=x_dim)
        # sigma = sigma_net(get_net(in_dim=x_dim, out_dim=z_dim + 1, **model_cfg["sigma"]), decoder, min_sigma_sq, max_sigma_sq)
        sigma = new_sigma_net(get_net(in_dim = z_dim, out_dim = z_dim + 1, **model_cfg["sigma"]))
        # energy = energy_net(encoder, decoder, model_cfg["encoder"]["nh_mlp"], model_cfg["energy"]["n_layers"])
        from models.energy_based import EnergyBasedModel
        # ebm = EnergyBasedModel(energy, **model_cfg["ebm"])
        ebm = None
        model = EnergyAE(encoder, decoder, ebm, sigma, **model_cfg["energy_ae"])
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict)
    return model

def _get_model_instance(name):
    try:
        return {
            "vae": get_ae,
            "irvae": get_ae,
            "eae": get_ae,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg