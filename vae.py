# Base VAE class definition

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

def get_mean_param(params):
    """
        Return the parameter used to show reconstructions or generations.
        For example, the mean for Normal, or probs for Bernoulli.
        For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    else:
        return params[0]
    
class VAE(nn.Module):
    def __init__(self, manifold, prior_dist, posterior_dist, encoder, decoder, learn_prior_std):
        super(VAE, self).__init__()
        self.manifold = manifold
        self.pz       = prior_dist
        self.qz_x     = posterior_dist
        self.encoder  = encoder
        self.decoder  = decoder
        
        # Prior distribution parameters
        self._pz_mu     = nn.Parameter(torch.zeros(1, manifold.coord_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.ones(1,1), requires_grad=learn_prior_std)
        
    def forward(self, x, K=1):
        qz_x      = self.qz_x(*self.encoder(x))
        z_samples = qz_x.rsample([K])
        px_z_mu   = self.decoder(z_samples)
        return qz_x, px_z_mu, z_samples
    
    @torch.no_grad()
    def generate(self, N, K):
        self.eval()
        latent_samples = self.pz(*self.pz_params).sample(torch.Size([N]))
        px_z_mu = self.decoder(latent_samples)
        return px_z_mu

    @torch.no_grad()
    def reconstruct(self, img):
        self.eval()
        qz_x     = self.qz_x(*self.encoder(img))
        z_sample = qz_x.rsample([1])
        px_z_mu  = self.decoder(z_sample)

        return px_z_mu

    @property
    def pz_params(self):
        """
            Parameters of the prior distribution
        """
        return self._pz_mu, F.softplus(self._pz_logvar).div(math.log(2)), self.manifold
    
    @torch.no_grad()
    def init_last_layer_bias(self, train_loader):
        if not hasattr(self.dec.dec.fc31, 'bias'): return
        p = torch.zeros(prod(data_size[1:]), device=self._pz_mu.device)
        N = 0
        for i, (data, _) in enumerate(train_loader):
            data = data.to(self._pz_mu.device)
            B = data.size(0)
            N += B
            p += data.view(-1, prod(data_size[1:])).sum(0)
        p /= N
        p += 1e-4
        self.dec.dec.fc31.bias.set_(p.log() - (1 - p).log())