import pandas as pd
import numpy as np

def preprocess_data(file_path):
    """
    Reads CSV/XLSX and returns numeric EEG data + sampling frequency
    """
    print(f"🔄 Loading dataset from: {file_path}")
    # Detect extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

    # Keep only numeric columns (EEG channels)
    numeric_df = df.select_dtypes(include=[np.number])
    print("✅ Dataset loaded successfully")
    print("First 5 rows:\n", numeric_df.head())

    data = numeric_df.values.T  # channels x samples
    print("Data shape (channels x samples):", data.shape)

    sfreq = 256  # default sampling frequency
    return data, sfreq