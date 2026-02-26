import numpy as np
from scipy.signal import welch

def extract_band_power(signal, fs=256):
    freqs, psd = welch(signal, fs)

    delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
    theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
    alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
    beta  = np.mean(psd[(freqs >= 13) & (freqs < 30)])

    return [delta, theta, alpha, beta]

def extract_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    band_features = extract_band_power(signal)

    return [mean, std] + band_features