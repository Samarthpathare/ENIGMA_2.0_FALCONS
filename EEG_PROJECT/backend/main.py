# backend/main.py

import os
from core.preprocessing import preprocess_data   # Excel/CSV compatible
from core.feature_extraction import extract_features
from models.predict import predict
from models.risk import calculate_risk

def run_pipeline(mode, source=None):
    """
    mode: "device" or "dataset"
    source: file path if dataset, else None
    """
    if mode == "device":
        # Dummy device mode
        print("🔌 Waiting for EEG Device connection...")
        return {
            "mode": "device",
            "status": "waiting",
            "data": None,
            "message": "Connect EEG device to start acquisition"
        }

    elif mode == "dataset":
        print(f"📂 Dataset file path: {source}")

        # Step 1: Preprocess dataset
        raw = preprocess_data(source)  # returns (data, sfreq) tuple
        print("🔄 Feature extraction in progress...")

        # Step 2: Extract features
        features = extract_features(raw)
        print("⚡ Features extracted:", features)

        # Step 3: Make prediction
        prediction = predict(features)
        print("🤖 Prediction:", prediction)

        # Step 4: Calculate risk score
        risk = calculate_risk(prediction)
        print("📊 Risk Score:", risk)

        return {
            "mode": "dataset",
            "status": "done",
            "features": features.tolist(),
            "prediction": prediction,
            "risk_score": risk
        }

    else:
        print("❌ Invalid mode")
        return None


if __name__ == "__main__":
    print("Choose input mode:")
    print("1 -> Device")
    print("2 -> Dataset")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        mode = "device"
        source = None
    elif choice == "2":
        mode = "dataset"
        file_path = input("Enter EEG dataset file path (CSV or XLSX): ").strip().strip('"')
        source = file_path
    else:
        print("❌ Invalid choice, defaulting to device")
        mode = "device"
        source = None

    output = run_pipeline(mode, source)
    print(output)