import cv2
import torch
from tqdm import tqdm

hog = cv2.HOGDescriptor(
    _winSize=(112, 112),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def get_hog_features(val_loader):
    all_features = []
    for images, *y in tqdm(val_loader):
        images_ = images.permute(0, 2, 3, 1).numpy()
        for img in images_:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
            features = hog.compute(gray)
            all_features.append(features.flatten())
    return torch.tensor(all_features)