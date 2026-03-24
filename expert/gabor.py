import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color, transform
from skimage.filters import gabor_kernel
from scipy.signal import fftconvolve
import warnings
warnings.filterwarnings('ignore')

def build_gabor_bank(n_orientations=6, n_scales=3):
    bank = []
    frequencies = np.logspace(np.log2(0.05), np.log2(0.4), n_scales, base=2)

    for freq in frequencies:
        sigma = 1.0 / (2 * np.pi * freq)
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            k = gabor_kernel(freq, theta=theta, sigma_x=sigma, sigma_y=sigma)
            bank.append({
                'theta_deg': np.degrees(theta),
                'freq'     : freq,
                'k_real'   : np.real(k),
                'k_imag'   : np.imag(k),
            })
    return bank

def apply_gabor_bank(gray, bank):
    responses = []
    for entry in bank:
        r = fftconvolve(gray, entry['k_real'], mode='same')
        i = fftconvolve(gray, entry['k_imag'], mode='same')
        responses.append({
            'theta_deg': entry['theta_deg'],
            'freq'     : entry['freq'],
            'magnitude': np.hypot(r, i),
        })
    return responses

def get_features_from_gabor_responses(responses):
    features = []
    for entry in responses:
        mag = entry['magnitude']
        features.append(mag)
    return np.array(features) # N x 112 x 112