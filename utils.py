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
        return file
            
        
def load_checkpoint(file, model, head, optimizer):
    if os.path.isfile(file):
        statedict = torch.load(file)
    
        model.load_state_dict(statedict['model_statedict'])
        head.load_state_dict(statedict['head_statedict'])
        optimizer.load_state_dict(statedict['optimizer_statedict'])
        epoch = statedict.get('epoch')

        print(f"Successfully load model statedict with epoch {epoch}.\n")
        
        return epoch + 1
    
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


def split_parameters_for_vit(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "pos_embed" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay



from .data import CustomImageFolderDataset
from torch.utils.data import DataLoader
from .head import AdaFace
from . import net
from .ViT import load_models
from .data import val_dataset

def get_train_loader(data_path, transforms, batch_size, shuffle=False,
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


def get_val_loader(data_path, batch_size=512, num_pro=4):
    val_ds = val_dataset(data_root=data_path)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_pro, pin_memory=True)
    
    return val_loader


def get_model(model_name, device):
    """
    model_name must be included "ir" or "vit"
    """
    if 'ir' in model_name.lower():
        model = net.build_model(model_name.lower())
    elif 'vit' in model_name.lower():
        model = load_models()
    else:
        raise ValueError("model_name is invalid.")
    
    model.to(device)
    return model


def load_weight(model, checkpoint):
    statedict = torch.load(checkpoint, weights_only=True)
    
    try:
        statedict = statedict['state_dict']
        
        new_state_dict = {}

        for k, v in statedict.items():
            if k.startswith("model."):
                new_key = k[len("model."):]
                new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)
        
        print("Load with .ckpt!")
        
    except:
        
        model.load_state_dict(statedict)
        print("Load with .pth!")
        


def get_head(device, embedding_size=512, classnum=8631, m=0.4, h=0.333, s=64, t_alpha=0.99):
    head = AdaFace(embedding_size=embedding_size, classnum=classnum, m=m, h=h, s=s, t_alpha=t_alpha)
    head.to(device)
    return head


def get_optimizer(model, model_name, head, lr, momentum=0.9, opt_type='sgd'):
    
    """
    model_name must be included "ir" or "vit".
    opt_type must be "sgd" or "adamw"
    """
    
    if 'ir' in model_name.lower():
        paras_wo_bn, paras_only_bn = split_parameters(model)
    elif 'vit' in model_name.lower():
        paras_wo_bn, paras_only_bn = split_parameters_for_vit(model)
    else:
        raise ValueError("model_name is invalid.")

    if opt_type.lower() == 'sgd':
        optimizer = torch.optim.SGD([{
                    'params': paras_wo_bn + [head.kernel],
                    'weight_decay': 1e-4
                }, {
                    'params': paras_only_bn
                }], lr=lr, momentum=momentum)
    elif opt_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW([{
                    'params': paras_wo_bn + [head.kernel],
                    'weight_decay': 1e-4
                }, {
                    'params': paras_only_bn
                }], lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError("opt_type must be 'sgd' or 'adamw'")
        
    return optimizer


def set_lr(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr
    
    
    
def combine_data(root, target_1, target_2, suffix_1, suffix_2):
    def copy(root, target, suffix):
        print("Count root folders: ", len(os.listdir(root)))
        for folder in os.listdir(target):
            if os.path.exists(os.path.join(root, folder)):
                shutil.move(
                    os.path.join(target, folder),
                    os.path.join(root, folder + suffix)
                )
            else:
                shutil.move(os.path.join(target, folder), os.path.join(root))

        print("Count root folders after moving: ", len(os.listdir(root)))
    
    copy(root, target_1, suffix_1)
    copy(root, target_2, suffix_2)
    
    
def free_memory():
    """
    Free memory
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
# git clone https://github.com/im-xiaoming/firework.git
from firework.loss import TaylorCrossEntropyLoss

def get_criterion(ctype='softmax'):
    """"
    Get Criterion
        type: strs, default: softmax; option: ["softmax", "taylor_softmax"]
    Returns:
        nn.Module: criterion
    """
    if ctype.lower() == "softmax":
        return nn.CrossEntropyLoss()
    elif ctype.lower() == "taylor_softmax":
        return TaylorCrossEntropyLoss()
    else:
        raise ValueError("type must be 'softmax' or 'taylor_softmax'")
    