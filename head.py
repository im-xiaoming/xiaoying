import torch
import torch.nn as nn
from .utils import l2_norm
import numpy as np
class AdaFace(nn.Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=8631,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.99,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        nn.init.xavier_uniform_(self.kernel)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm, _ = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings, kernel_norm) # NxD x DxC -> NxC
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
        sine = torch.sqrt(1 - cosine**2)

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        one_hot = torch.zeros_like(cosine, device=cosine.device) # N x C
        one_hot.scatter_(1, label.reshape(-1, 1), 1.0)
        
        g_angular = -1 * self.m * margin_scaler
        g_add = self.m * margin_scaler + self.m
        
        phi = cosine * torch.cos(g_angular) - sine * torch.sin(g_angular) # phi = cos(theta + ....)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        logits = (one_hot * phi) - (one_hot * g_add) + (1 - one_hot) * cosine
        return logits * self.s
    
    

class SubAdaFace(nn.Module):
    def __init__(self,
                 K=3,
                 embedding_size=512,
                 classnum=8631,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.99,
                 ):
        super(SubAdaFace, self).__init__()
        self.classnum = classnum
        self.K = K
        kernel = torch.Tensor(embedding_size, classnum * K)
        nn.init.xavier_uniform_(kernel)
        self.kernel = nn.Parameter(kernel)

        # initial kernel
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm, _ = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
        
        cosine = cosine.view(-1, self.classnum, self.K)
        cosine, _ = cosine.max(dim=2) # (N, C)
        
        
        sine = torch.sqrt(1 - cosine**2)

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        one_hot = torch.zeros_like(cosine, device=cosine.device) # N x C
        one_hot.scatter_(1, label.reshape(-1, 1), 1.0)
        
        g_angular = -1 * self.m * margin_scaler
        g_add = self.m * margin_scaler + self.m
        
        phi = cosine * torch.cos(g_angular) - sine * torch.sin(g_angular) # phi = cos(theta + ....)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        logits = (one_hot * phi) - (one_hot * g_add) + (1 - one_hot) * cosine
        return logits * self.s