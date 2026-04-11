import torch
import torch.nn as nn
import os, shutil
from sklearn.decomposition import KernelPCA


def kernel_pca(features, n_components=512, kernel='rbf'):
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    transformed_features = kpca.fit_transform(features)
    return kpca, transformed_features

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis,True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms):

    assert stacked_embeddings.ndim == 3
    assert stacked_norms.ndim == 3

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm
 
 
class CheckPoint:
    def __init__(self, model, head, optimizer):
        self.model = model
        self.head = head
        self.optimizer = optimizer
        
    def save(self, file, epoch):
        model_statedict = self.model.state_dict()
        head_statedict = self.head.state_dict()
        optimizer_statedict = self.optimizer.state_dict()
        torch.save({
            'model_statedict': model_statedict,
            'head_statedict': head_statedict,
            'optimizer_statedict': optimizer_statedict,
            'epoch': epoch
        }, file)
        print("\nSave checkpoint successfully!\n")
            
        
def load_checkpoint(file, model, head, optimizer, device='gpu'):
    if os.path.isfile(file):
        statedict = torch.load(file, weights_only=False, map_location=device)
    
        model.load_state_dict(statedict['model_statedict'])
        head.load_state_dict(statedict['head_statedict'])
        optimizer.load_state_dict(statedict['optimizer_statedict'])
        epoch = statedict.get('epoch')

        print(f"Successfully load model statedict with epoch {epoch}.\n")
        
        return epoch
    
    return 1


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay






from .data import CustomImageFolderDataset
from torch.utils.data import DataLoader
from .head import AdaFace
from . import net

def get_loader(data_path, transforms, batch_size, shuffle=False,
               low_res_augmentation_prob=0.0, crop_augmentation_prob=0.0, photometric_augmentation_prob=0.0):
    dataset = CustomImageFolderDataset(data_path,
                                        transform=transforms,
                                        low_res_augmentation_prob=low_res_augmentation_prob,
                                        crop_augmentation_prob=crop_augmentation_prob,
                                        photometric_augmentation_prob=photometric_augmentation_prob
                                        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    return loader

def get_model(model_name, device):
    model = net.build_model(model_name)
    model.to(device)
    return model


def get_head(device, embedding_size=512, classnum=8631, m=0.4, h=0.333, s=64, t_alpha=0.99):
    head = AdaFace(embedding_size=embedding_size, classnum=classnum, m=m, h=h, s=s, t_alpha=t_alpha)
    head.to(device)
    return head


def get_optimizer(model, head, lr):
    paras_wo_bn, paras_only_bn = split_parameters(model)

    optimizer = torch.optim.SGD([{
                'params': paras_wo_bn + [head.kernel],
                'weight_decay': 1e-4
            }, {
                'params': paras_only_bn
            }], lr=lr, momentum=0.9)
    
    return optimizer

