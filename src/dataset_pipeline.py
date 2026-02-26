import os
import numpy as np
import pandas as pd
from src.preprocessing import bandpass_filter, normalize_signal
from src.feature_extraction import extract_features

def create_feature_dataset(data_path):
    features = []
    labels = []

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            signal = np.loadtxt(os.path.join(data_path, file))

            signal = bandpass_filter(signal)
            signal = normalize_signal(signal)

            feature_vector = extract_features(signal)
            features.append(feature_vector)

            label = 1 if "schiz" in file.lower() else 0
            labels.append(label)

    df = pd.DataFrame(features, columns=[
        "mean","std","delta","theta","alpha","beta"
    ])
    df["label"] = labels

    return df