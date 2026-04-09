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
 
 
class SaveCheckPoint:
    def __init__(self, model, head, optimizer):
        self.model = model
        self.head = head
        self.optimizer = optimizer
        
    def save(self, file, epoch):
        if os.path.isfile(file):
            model_statedict = self.model.state_dict()
            head_statedict = self.head.state_dict()
            optimizer_statedict = self.optimizer.state_dict()
            torch.save(file, {
                'model_statedict': model_statedict,
                'head_statedict': head_statedict,
                'optimizer_statedict': optimizer_statedict,
                'epoch': epoch
            })
            print("\nSave checkpoint successfully!\n")
        
    def load_checkpoint(self, file, device='gpu'):
        if os.path.isfile(file):
            statedict = torch.load(file, weights_only=False, map_location=device)
        
            self.model.load_state_dict(statedict['model_statedict'])
            self.head.load_state_dict(statedict['head_statedict'])
            self.optimizer.load_state_dict(statedict['optimizer_statedict'])
            epoch = statedict.get('epoch')

            print(f"Successfully load model statedict with epoch {epoch}.\n")
            
            return epoch
        
        return -1


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