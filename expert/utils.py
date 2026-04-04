from tqdm  import tqdm
import torch
import numpy as np
from . import gabor
from ..utils import kernel_pca
from skimage.color import rgb2gray
from multiprocessing import Pool, cpu_count


def process_one_image(img, bank):
    gray = rgb2gray(img)
    responses = gabor.apply_gabor_bank(gray, bank)
    
    features = []
    for entry in responses:
        features.append(entry['magnitude'].flatten())
    
    return np.concatenate(features, axis=0)


def get_expert_features(val_loader):
    all_features = []
    bank = gabor.build_gabor_bank()

    pool = Pool(cpu_count())  # dùng hết CPU

    for images, *y in tqdm(val_loader):
        images_ = images.permute(0, 2, 3, 1).numpy()

        # chạy song song
        features = pool.starmap(
            process_one_image,
            [(img, bank) for img in images_]
        )

        features = np.array(features)

        _, pca_features = kernel_pca(features, 512)
            
        m, n = pca_features.shape
        if n != 512:
            if isinstance(pca_features, np.ndarray):
                pca_features = torch.from_numpy(pca_features)
            pca_features = torch.cat([pca_features, torch.zeros(m, 512 - n)], dim=-1)
        all_features.extend(pca_features)

    pool.close()
    pool.join()

    F = all_features
    
    return torch.tensor(all_features)


def combined_features(feats1, feats2):
    feats1 = torch.tensor(feats1)
    feats2 = torch.tensor(feats2)
    feats = torch.cat([feats1, feats2], axis=-1)
    return feats