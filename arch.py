import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

from manifold_layers import GeodesicLayer, ExpZero


class Resnet50(nn.Module):
    def __init__(self, unfrozen_layers):
        """
            unfrozen_layers should be list e.g., ['layer3', 'layer4']
        """
        super(Resnet50, self).__init__()
        
        self.backbone    = resnet50(weights='DEFAULT') 
        self.backbone.fc = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for layer in unfrozen_layers:
            for param in getattr(self.backbone, layer).parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        return x

    
class EncoderWrapped(nn.Module):
    def __init__(self, manifold, prior_isotropic):
        super(EncoderWrapped, self).__init__()
        self.manifold = manifold

        self.backbone = Resnet50(['layer3', 'layer4'])
        
        self.fc = nn.Sequential(*[nn.Linear(2048, 1024), nn.GELU(),
                                  nn.Linear(1024, 256),  nn.GELU()])
        
        self.fc_mu    = nn.Linear(256, manifold.coord_dim)
        self.fc_scale = nn.Linear(256, manifold.coord_dim if not prior_isotropic else 1)

    def forward(self, x):
        x   = self.backbone(x)
        e   = self.fc(x)
        mu  = self.fc_mu(e)
        mu  = self.manifold.expmap0(mu)

        scale = F.softplus(self.fc_scale(e)) + 1e-5
        
        return mu, scale, self.manifold

    
class Deconv(nn.Module):
    def __init__(self, hidden_dims):
        super(Deconv, self).__init__()
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i],
                                                            hidden_dims[i + 1],
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1), nn.GELU()))

        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1],
                                                        hidden_dims[-1],
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1,
                                                        output_padding=1), nn.GELU(),
                                     nn.Conv2d(in_channels=hidden_dims[-1],
                                               out_channels=3,
                                               kernel_size=3,
                                               padding=1)))

        self.deconv = nn.Sequential(*modules)

    def forward(self, z):
        x = self.deconv(z)
        return x[...,16:-16,16:-16]
    
class DecoderGeodesic(nn.Module):
    """ 
        First layer is a Hypergyroplane followed by usual decoder 
    """
    def __init__(self, manifold, hidden_dims=[512,256,128,128,64,32]):
        super(DecoderGeodesic, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.fc = nn.Sequential(GeodesicLayer(manifold.coord_dim,
                                              self.hidden_dims[0]*4,
                                              manifold),
                                nn.GELU(),
                                nn.Linear(self.hidden_dims[0]*4, self.hidden_dims[0]*16),
                                nn.GELU())
        
        self.deconv = Deconv(hidden_dims)

    def forward(self, z):
        d  = self.fc(z)
        # mu of the estimated likelihood px_z
        d  = d.view(z.shape[0]*z.shape[1], self.hidden_dims[0], 4, 4)
        
        mu = self.deconv(d)
        mu = mu.view(z.shape[0], z.shape[1], 3, 224, 224)
        return mu
    