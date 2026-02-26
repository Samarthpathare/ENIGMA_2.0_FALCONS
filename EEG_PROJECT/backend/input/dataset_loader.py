# input/dataset_loader.py

import mne

def load_dataset_file(file_path):
    """
    Load EEG EDF file
    """

    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw