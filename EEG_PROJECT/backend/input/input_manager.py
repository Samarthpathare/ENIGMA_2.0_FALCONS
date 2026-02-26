# backend/input/input_manager.py

import os

def get_input_data(mode="device", file_path=None):
    """
    Handles EEG input source
    """
    if mode == "device":
        return _handle_device_input()
    elif mode == "dataset":
        return _handle_dataset_input(file_path)
    else:
        raise ValueError("❌ Invalid mode. Choose 'device' or 'dataset'.")

def _handle_device_input():
    print("🔌 Waiting for EEG Device connection...")
    return {
        "mode": "device",
        "status": "waiting",
        "data": None,
        "message": "Connect EEG device to start acquisition"
    }

def _handle_dataset_input(file_path):
    if file_path is None:
        raise ValueError("❌ Dataset mode requires file_path.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    print(f"📂 Loading EEG file: {file_path}")
    return {
        "mode": "dataset",
        "status": "loaded",
        "file_path": file_path,
        "message": "EEG file loaded successfully"
    }