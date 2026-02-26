import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut=0.5, highcut=50, fs=256, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def normalize_signal(data):
    return (data - np.mean(data)) / np.std(data)