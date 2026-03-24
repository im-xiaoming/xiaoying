import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from .import evaluate_utils
from ..validation_mixed.validate_IJB_BC import fuse_features_with_norm, get_features, evaluate
from ..expert import gabor
from ..utils import kernel_pca
from skimage.color import rgb2gray

def evaluate1(model, val_loader, device, expert=False, counter=-1):
    model.eval()
    
    count = 0

    all_embeddings = []
    all_features = []
    all_labels = []
    all_datanames = []
    all_indices = []
    
    if expert:
        bank = gabor.build_gabor_bank()
        print(f'Gabor bank built with {len(bank)} filters')
    
    with torch.no_grad():
        for images, labels, datanames, indices in tqdm(val_loader):
            if expert:
                images_ = images.permute(0, 2, 3, 1)
                grays = [rgb2gray(img) for img in images_]
                list_responses = [gabor.apply_gabor_bank(gray, bank) for gray in grays]
                features = []
                for response in list_responses:
                    images_ = []
                    for entry in response:
                        images_.append(entry['magnitude'].flatten())
                    features.append(np.concatenate(images_, axis=0)) # (32, 225792)
                features = torch.tensor(features, dtype=torch.float32)
                all_features.append(features) # (N, 32, _)
            
            # MODEL EMBEDDINGS
            images, labels = images.to(device), labels.to(device)

            embeddings, norms = model(images)

            flip_embeddings, flip_norms = model(torch.flip(images, dims=[3]))
            embeddings, norms = fuse_features_with_norm(
                torch.stack([embeddings, flip_embeddings], 0),
                torch.stack([norms, flip_norms], 0)
            )

            all_embeddings.append(embeddings.cpu()) # (N, 32, 512)
            all_labels.append(labels.cpu())
            all_datanames.append(datanames.cpu())
            all_indices.append(indices.cpu())
            
            count += 1
            if count == counter:
                break

        embeddings = torch.cat(all_embeddings) # (N, 512)
        labels = torch.cat(all_labels)
        datanames = torch.cat(all_datanames)
        indices = torch.cat(all_indices)
        
        if expert:
            print('Start to apply kernel PCA on Gabor features\n')
            print(f"Features shape before PCA: ({len(all_features)},{len(all_features[0][0])})")
            kpca, all_features = kernel_pca(torch.cat(all_features, dim=0).numpy(), n_components=512) # (N, 512)
            print(f"Features shape after PCA: {all_features.shape}\n")
            embeddings = torch.cat([embeddings, all_features], dim=1) # (N, 1024)
            
            print(f"Embeddings shape after concatenating with Gabor features: {embeddings.shape}\n")
        
        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cfp_ff": 3}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}

        val_acc = []

        for name, index in dataname_to_idx.items():
            # per dataset evaluation
            emb = embeddings[datanames == index].to('cpu').numpy()
            lab = labels[datanames == index].to('cpu').numpy()
            issame = lab[0::2]

            # evaluate
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(emb, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            val_acc.append(acc)

            print(f"\t[{name}] Acc={acc:.4f}, Th={best_threshold:.4f}")

    return np.mean(val_acc)


def evaluate2(root, model, model_name, data_name, batch_size=256, device='gpu'):    
    feats, save_path = get_features(root, model, model_name,
                                    data_name, batch_size, device)

    evaluate(root, data_name, feats, save_path)

    df = pd.read_csv(f'{save_path}/verification_result.csv')
    r = df[['1e-06', '1e-05', '0.0001']].to_dict()
    r = {key: val[0] for key, val in r.items()}
    return r