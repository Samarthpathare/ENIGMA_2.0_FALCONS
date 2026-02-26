# backend/core/feature_extraction.py

import numpy as np
from scipy.signal import welch

# backend/core/feature_extraction.py

def bandpower(data, sf, band):
    low, high = band
    nperseg = min(1024, data.shape[1])  # prevent warning on short signals
    freqs, psd = welch(data, sf, nperseg=nperseg)
    idx = (freqs >= low) & (freqs <= high)
    return np.mean(psd[:, idx])


def extract_features(preprocessed_data):
    """
    preprocessed_data: tuple (data, sfreq)
    """
    data, sf = preprocessed_data  # unpack tuple

    delta = bandpower(data, sf, (0.5, 4))
    theta = bandpower(data, sf, (4, 8))
    alpha = bandpower(data, sf, (8, 13))
    beta  = bandpower(data, sf, (13, 30))
    gamma = bandpower(data, sf, (30, 40))

    features = np.array([delta, theta, alpha, beta, gamma])
    return features.reshape(1, -1)