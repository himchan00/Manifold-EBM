import torch
from functools import partial
def relaxed_distortion_measure(func, z, eta=0.2, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    z_perm = z[torch.randperm(bs)]
    alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
    z_augmented = alpha*z + (1-alpha)*z_perm
    v = torch.randn(z.size()).to(z)
    Jv = torch.autograd.functional.jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
    JTJv = (torch.autograd.functional.vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.sum(JTJv**2, dim=1).mean()
    return TrG2/TrG**2

def conformal_distortion_measure(func, z, eta=0.2, create_graph=True):
    '''
    func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
    '''
    bs = len(z)
    if eta is not None:
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z
    v = torch.randn(z.size()).to(z)
    Jv = torch.autograd.functional.jvp(
        func, z_augmented, v=v, create_graph=create_graph)[1]
    TrG = torch.mean(Jv.view(bs, -1)**2, dim=1) # (bs,)
    JTJv = (torch.autograd.functional.vjp(
        func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
    TrG2 = torch.mean(JTJv**2, dim=1) # (bs,)
    return TrG2/TrG**2
    
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

def spherical_jacobian_of_f(f, z, create_graph=True):
    batch_size, z_dim = z.size()
    A = torch.randn(batch_size, z_dim, z_dim).to(z)
    A[:, :, 0] = z.detach()
    Q, _ = torch.linalg.qr(A)
    Q = Q[:, :, 1:]
    Q = Q.permute(0, 2, 1) # (bs, z_dim-1, z_dim)
    v = Q.reshape(-1, z_dim)
    z = z.repeat(1, z_dim-1).view(-1, z_dim)
    out = (
        torch.autograd.functional.jvp(
            f, z, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim-1, -1).permute(0, 2, 1)
    )
    return out

def relaxed_volume_preserving_measure(func, z, eta=0.2, create_graph=True):
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
    X = z_augmented.unsqueeze(2).to(z)
    X_t = X.permute(0, 2, 1).to(z)
    P = torch.eye(n).repeat(bs, 1, 1).to(z) - torch.bmm(X, X_t).to(z)
    J = jacobian_of_f(func, z_augmented, create_graph=create_graph)
    JP = torch.bmm(J, P)
    pullback_metric = JP.permute(0, 2, 1)@JP
    # J = jacobian_of_f(func, z_augmented, create_graph=create_graph)
    # pullback_metric = J.permute(0, 2, 1)@J
    eig_vals = torch.linalg.eigvalsh(pullback_metric)
    eig_vals = eig_vals[:, 1:]
    logdet = torch.log(eig_vals).sum(dim=1)
    logdet_sq = logdet**2
    logdet_sq_mean = logdet_sq.mean()
    logdet_mean = logdet.mean()
    return logdet_sq_mean/logdet_mean**2

def get_log_det_jacobian_new(f, z_samples, create_graph=True):
    G = get_pullbacked_Riemannian_metric(f, z_samples, create_graph)
    logdet = torch.logdet(G)
    logdet[torch.isnan(logdet)] = 0
    logdet[torch.isinf(logdet)] = 0
    return logdet/2.0

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

def get_pullbacked_Riemannian_metric(func, z, create_graph=True):
    J = jacobian_decoder_jvp_parallel(func, z, v=None, create_graph=create_graph)
    G = torch.einsum('nij,nik->njk', J, J)
    return G

def get_projection_matrix_times_v(func, v, z):
    J = jacobian_of_f(func, z, create_graph=True)
    pull_back_metric = J.permute(0, 2, 1)@J
    inverse_pull_back_metric = torch.linalg.pinv(pull_back_metric, hermitian=True)
    projection_matrix = J @ inverse_pull_back_metric @ J.permute(0, 2, 1)
    Tv = (projection_matrix @ v).squeeze(-1) # (bs, x_dim)
    return Tv

def get_projection_coord_rep(func, z, x, create_graph = True, return_curvature = True, eta = None):
    J = jacobian_of_f(func, z, create_graph=create_graph)
    J_column_norm_sq = torch.sum(J**2, dim=1)
    pull_back_metric = J.permute(0, 2, 1)@J
    inverse_pull_back_metric = torch.linalg.pinv(pull_back_metric, hermitian=True)
    coord_rep = inverse_pull_back_metric @ J.permute(0, 2, 1)
    coord_rep_x = coord_rep @ x.unsqueeze(-1)
    projection_matrix = J @ coord_rep
    if return_curvature:
        if eta is not None:
            bs = len(z)
            z_perm = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
        v = torch.randn(x.size()).to(z).unsqueeze(-1)
        w = torch.randn(z.size()).to(z).unsqueeze(-1)
        w_tilda = inverse_pull_back_metric @ w
        grad_T_w = torch.autograd.functional.jvp(
            partial(get_projection_matrix_times_v, func, v), z_augmented, v=w.squeeze(), create_graph=create_graph)[1]
        grad_T_w_tilda = torch.autograd.functional.jvp(
            partial(get_projection_matrix_times_v, func, v), z_augmented, v=w_tilda.squeeze(), create_graph=create_graph)[1]
        extrinsic_curvature = torch.bmm(grad_T_w_tilda.unsqueeze(1), grad_T_w.unsqueeze(2)).squeeze(-1) / 2.0

        return projection_matrix, coord_rep_x.squeeze(-1), J_column_norm_sq, extrinsic_curvature
    else:
        return projection_matrix, coord_rep_x.squeeze(-1), J_column_norm_sq

def curvature_reg(func, z, create_graph = True, eta = None):
    if eta is not None:
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z
    
    J = jacobian_of_f(func, z_augmented, create_graph=create_graph)
    bs, x_dim, z_dim = J.shape
    pull_back_metric = J.permute(0, 2, 1)@J
    inverse_pull_back_metric = torch.linalg.pinv(pull_back_metric, hermitian=True)
    v_ = torch.randn((bs, x_dim)).to(z).unsqueeze(-1)
    w = torch.randn((bs, z_dim)).to(z).unsqueeze(-1)
    w_tilda = inverse_pull_back_metric @ w
    grad_T_w = torch.autograd.functional.jvp(
        partial(get_projection_matrix_times_v, func, v_), z_augmented, v=w.squeeze(), create_graph=create_graph)[1]
    grad_T_w_tilda = torch.autograd.functional.jvp(
        partial(get_projection_matrix_times_v, func, v_), z_augmented, v=w_tilda.squeeze(), create_graph=create_graph)[1]
    extrinsic_curvature = torch.bmm(grad_T_w_tilda.unsqueeze(1), grad_T_w.unsqueeze(2)).squeeze(-1) / 2.0
    return extrinsic_curvature

    
