from tqdm  import tqdm
import torch
import numpy as np
from . import gabor
from ..utils import kernel_pca
from skimage.color import rgb2gray

def get_expert_features(val_loader, counter=-1):
    all_features = []
    bank = gabor.build_gabor_bank()
    
    for images, _, _, _ in tqdm(val_loader):
        images_ = images.permute(0, 2, 3, 1)
        grays = [rgb2gray(img) for img in images_]
        list_responses = [gabor.apply_gabor_bank(gray, bank) for gray in grays]
        features = []
        for response in list_responses:
            images_ = []
            for entry in response:
                images_.append(entry['magnitude'].flatten())
            features.append(np.concatenate(images_, axis=0)) # (32, 225792)
        all_features.extend(features)
    return all_features