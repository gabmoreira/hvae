import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.transforms as T
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from vae import VAE
from arch import EncoderWrapped, DecoderGeodesic
from loss import vae_loss
from wrapped_normal import WrappedNormal
from poincareball import PoincareBall

from loader import *
from tqdm import tqdm 


if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Configuration params
    cfg = {'img_path'            : '../../../localdisk/ifetch/deepfashion/in_shop',
           'train_dict_path'     : '../preprocessed/train_split.pt',
           'val_dict_path'       : '../preprocessed/val_split.pt',
           'taxonomy_path'       : None,
           'batch_size'          : 64,
           'lr'                  : 8e-4,
           'latent_dim'          : 64,
           'curvature'           : 1.0,
           'k'                   : 5000,
           'beta'                : 10.0}

    data_T = T.Compose([T.RandomHorizontalFlip(),
                        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.2),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_samples = DeepFashionData(img_path=cfg['img_path'],
                                    data_dict_path=cfg['train_dict_path'],
                                    taxonomy_path=cfg['taxonomy_path'],
                                    transforms=data_T)

    val_samples = DeepFashionData(img_path=cfg['img_path'],
                                  data_dict_path=cfg['val_dict_path'],
                                  taxonomy_path=cfg['taxonomy_path'],
                                  transforms=data_T)

    merge_samples = torch.utils.data.ConcatDataset([train_samples, val_samples])

    train_loader = DataLoader(merge_samples,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=train_samples.collate_fn,
                              pin_memory=True)
    
    manifold = PoincareBall(dim=cfg['latent_dim'], c=cfg['curvature'])
    encoder  = EncoderWrapped(manifold=manifold, prior_isotropic=False)
    decoder  = DecoderGeodesic(manifold=manifold) 

    vae = VAE(manifold=manifold,
              prior_dist=WrappedNormal,
              posterior_dist=WrappedNormal,
              encoder=encoder,
              decoder=decoder,
              learn_prior_std=False).to(device)
    
    #vae.load_state_dict(torch.load('./df_pvae_64b.pt', map_location=device))

    optimizer = torch.optim.Adam(vae.parameters(), weight_decay=0.0, lr=cfg['lr'] )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    vae.train()
    for epoch in range(1,150):
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='[EPOCH {}]'.format(epoch)) 

        agg = {'loss' : [], 'kl' : [], 'reconstruction' : []} 
        for i_batch, batch_dict in enumerate(train_loader):
            img = batch_dict['img'].to(device)

            optimizer.zero_grad()
            qz_x, px_z_mu, recon, kl, loss = vae_loss(vae, img, K=cfg['k'], beta=cfg['beta'], components=True, analytical_kl=False)
            loss.backward()
            optimizer.step()

            agg['loss'].append(loss.item())
            agg['reconstruction'].append(recon.item())
            agg['kl'].append(kl.sum(-1).mean(0).sum().item())

            batch_bar.set_postfix(Loss="{:1.5e}".format(np.mean(agg['loss'])), 
                                  KL="{:1.5e}".format(np.mean(agg['kl'])), 
                                  Reconstruction="{:1.5e}".format(np.mean(agg['reconstruction'])), 
                                  lr="{:1.2e}".format(optimizer.param_groups[0]['lr']),
                                  qz_x_scale="{:1.2e}".format(torch.max(qz_x.scale).item()))
            batch_bar.update()
        batch_bar.close()
        scheduler.step()
        torch.save(vae.state_dict(), './df_pvae_64_beta1.pt')