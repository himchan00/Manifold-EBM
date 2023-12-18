import torch

def relaxed_distortion_measure(func, z, eta=0.2, metric='identity', create_graph=True):
    if metric == 'identity':
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = (alpha*z + (1-alpha)*z_perm)
            z_augmented = z_augmented / z_augmented.norm(dim=-1, keepdim=True)
        else:
            z_augmented = z
        bs, n = z.shape[0], z.shape[1]
        # Projection matrix
        X = z.unsqueeze(2).to(z)
        X_t = X.permute(0, 2, 1).to(z)
        P = torch.eye(n).repeat(bs, 1, 1).to(z) - torch.bmm(X, X_t).to(z)
        v = (P @ torch.randn(z.size()).to(z).unsqueeze(-1)).squeeze(-1)
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()
        return TrG2/TrG**2
    else:
        raise NotImplementedError
    
def jacobian_of_f(f, z, create_graph=True):
    batch_size, z_dim = z.size()
    v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(z)
    z = z.repeat(1, z_dim).view(-1, z_dim)
    out = (
        torch.autograd.functional.jvp(
            f, z, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return out 

def relaxed_volume_preserving_measure(func, z, eta=0.2, create_graph=True):
    bs = len(z)
    z_perm = z[torch.randperm(bs)]
    if eta is not None:
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = (alpha*z + (1-alpha)*z_perm)
        # z_augmented = z_augmented / z_augmented.norm(dim=-1, keepdim=True)
    else:
        z_augmented = z
    bs, n = z.shape[0], z.shape[1]
    # Projection matrix
    # X = z_augmented.unsqueeze(2).to(z)
    # X_t = X.permute(0, 2, 1).to(z)
    # P = torch.eye(n).repeat(bs, 1, 1).to(z) - torch.bmm(X, X_t).to(z)
    # J = jacobian_of_f(func, z_augmented, create_graph=create_graph)
    # JP = torch.bmm(J, P)
    # pullback_metric = JP.permute(0, 2, 1)@JP
    J = jacobian_of_f(func, z_augmented, create_graph=create_graph)
    pullback_metric = J.permute(0, 2, 1)@J
    eig_vals = torch.linalg.eigvalsh(pullback_metric)
    # eig_vals = eig_vals[:, 1:]
    det = torch.prod(eig_vals, dim=1)
    det_sq = det**2
    det_sq_mean = det_sq.mean()
    det_mean = det.mean()
    if det_mean.min() < 1e-12:
        print("det_mean is too small")
        return torch.tensor(0.0).to(z)
    
    return det_sq_mean/det_mean**2


def get_log_det_jacobian(f, z_samples, return_avg=True, training=True, create_graph=True):
    '''
    f:          torch.nn.module class 
    z_samples:  torch.tensor whose size = (n, 2) 
    out:        torch.tensor whose size = (1, )
    '''
    if training:
        bs = len(z_samples)
        v = torch.randn(z_samples.size()).to(z_samples)
        Jv = torch.autograd.functional.jvp(f, z_samples, v=v, create_graph=True)[1]
        losses = torch.sum(Jv.view(bs, -1)**2, dim=1)
    else:
        bs, n = z_samples.shape[0], z_samples.shape[1]
        # Projection matrix
        X = z_samples.unsqueeze(2).to(z_samples)
        X_t = X.permute(0, 2, 1).to(z_samples)
        P = torch.eye(n).repeat(bs, 1, 1).to(z_samples) - torch.bmm(X, X_t).to(z_samples)
        J = jacobian_of_f(f, z_samples, create_graph=create_graph)
        JP = torch.bmm(J, P)
        pullback_metric = JP.permute(0, 2, 1)@JP
        eig_vals = torch.linalg.eigvalsh(pullback_metric)
        eig_vals = eig_vals[:, 1:]
        # logdet = torch.log(eig_vals).sum(dim=1)
        # logdet[torch.isnan(logdet)] = 0
        # logdet[torch.isinf(logdet)] = 0
        # J = jacobian_of_f(f, z_samples)
        # pullback_metric = J.permute(0, 2, 1)@J
        # eig_vals = torch.linalg.eigvalsh(pullback_metric)
        logdet = torch.log(eig_vals).sum(dim=1)
        logdet[torch.isnan(logdet)] = 0
        logdet[torch.isinf(logdet)] = 0
    if return_avg:
        return logdet.mean()/2.0
    else:
        return logdet/2.0


def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values/S.min(1).values
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean)@G
        scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    else:
        pass
    return scores

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def get_pullbacked_Riemannian_metric(func, z):
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G

