import torch
import torch.nn.functional as F
import torch.distributions as dist
from numpy import prod
from utils import has_analytic_kl, log_mean_exp
import torch.nn.functional as F


def vae_loss(model, x, K, beta, components, analytical_kl, **kwargs):
    """
        Computes E_{p(x)}[ELBO]
    """
    # x       has shape (batch_size, channels, h, w)
    # px_z_mu has shape (k, batch_size, channels, h, w)
    # z       has shape (k, batch_size, latent_dim)
    qz_x, px_z_mu, _ = model(x, 8)

    # LOG-LIKELIHOOD log P(X|Z) (ASSUMED GAUSSIAN)
    nll_px_z = torch.square(x.unsqueeze(0) - px_z_mu).mean(dim=0).mean()

    # EMPIRICAL KLD BETWEEN WRAPPED NORMAL DISTRIBUTIONS
    # z_samples                has shape(k, batch_size, latent_dim)
    # qz_x.log_prob(z_samples) has shape(k, batch_size, 1)
    # pz.log_prob(z_samples)   has shape(k, batch_size, 1)
    pz            = model.pz(*model.pz_params)
    z_samples     = qz_x.rsample(torch.Size([K]))
    kl_divergence = qz_x.log_prob(z_samples) - pz.log_prob(z_samples) 
    kl_divergence = kl_divergence.mean(dim=0).mean()

    # Multivariate Euclidean Normal KLD to bootstrap the process
    nkl = torch.log(torch.prod(pz.scale, dim=-1)) - torch.log(torch.prod(qz_x.scale, dim=-1))
    nkl = nkl - len(pz.scale)
    nkl = nkl + (qz_x.scale / pz.scale).sum(dim=-1)
    nkl = nkl + torch.norm(torch.sqrt(1/pz.scale)*(pz.loc-qz_x.loc),p=2,dim=-1)**2
    nkl = nkl * 0.5
        
    divergence = nkl.mean() #(1.0*kl_divergence + 0.2*nkl.mean(dim=0).mean())
    loss = nll_px_z + beta * divergence
    
    return (qz_x, px_z_mu, nll_px_z, divergence, loss) if components else loss



