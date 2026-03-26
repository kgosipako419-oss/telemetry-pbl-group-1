import pandas as pd
import numpy as np
import json
import os
from scipy.stats import kurtosis
from scipy.fft import fft

# ===============================
# CONFIG
# ===============================
BIT_DEPTH = 8

BASE_DATA_DIR = "data/raw"
OUTPUT_DIR = "results"
LOG_DIR = "logs"
JSON_DIR = "dashboard"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


# ===============================
# 1. LOAD CSV
# ===============================
def load_signal(csv_file):
    df = pd.read_csv(csv_file)
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise ValueError(f"No numeric data in {csv_file}")

    return df


# ===============================
# 2. ADAPTIVE QUANTIZATION
# ===============================
def adaptive_quantization(signal, csv_file, bits=8):
    L = 2 ** bits
    quantized_df = pd.DataFrame()

    for col in signal.columns:
        x = signal[col].values
        xmax = np.max(np.abs(x)) * 1.05
        delta = (2 * xmax) / L

        q = np.round((x + xmax) / delta)
        q = np.clip(q, 0, L - 1)

        quantized_df[col] = q.astype(int)

    filename = os.path.basename(csv_file).replace(".csv", "")
    quantized_df.to_csv(f"{OUTPUT_DIR}/{filename}_quantized.csv", index=False)

    return quantized_df


# ===============================
# 3. PCM ENCODING
# ===============================
def pcm_encode(quantized_df, csv_file, bits=8):
    bitstream = ""

    for _, row in quantized_df.iterrows():
        for val in row:
            bitstream += format(int(val), f'0{bits}b')

    filename = os.path.basename(csv_file).replace(".csv", "")
    with open(f"{OUTPUT_DIR}/{filename}_bitstream.txt", "w") as f:
        f.write(bitstream)

    return bitstream


# ===============================
# 4. MANCHESTER ENCODING
# ===============================
def manchester_encode(bitstream):
    encoded = ""

    for bit in bitstream:
        if bit == "1":
            encoded += "01"
        else:
            encoded += "10"

    return encoded


# ===============================
# 5. BER CALCULATION
# ===============================
def calculate_ber(original, received, csv_file):
    errors = sum(o != r for o, r in zip(original, received))
    ber = errors / len(original) if len(original) > 0 else 0

    filename = os.path.basename(csv_file).replace(".csv", "")
    with open(f"{LOG_DIR}/{filename}_integrity.txt", "w") as f:
        f.write(f"Total Bits: {len(original)}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"BER: {ber}\n")

    return ber


# ===============================
# 6. FEATURE EXTRACTION (UPGRADED)
# ===============================
def extract_features(signal, fs=100):
    features = {}

    for col in signal.columns:
        x = signal[col].values
        N = len(x)

        # ---- Time domain ----
        mean_val = np.mean(x)
        rms_val = np.sqrt(np.mean(x**2))
        variance_val = np.var(x)
        max_val = np.max(x)
        min_val = np.min(x)

        crest_factor = max(abs(x)) / rms_val if rms_val != 0 else 0
        kurt_val = kurtosis(x)

        # ---- Frequency domain ----
        X = np.abs(fft(x))
        freqs = np.fft.fftfreq(N, d=1/fs)

        mask = freqs > 0
        freqs = freqs[mask]
        X = X[mask]

        dominant_freq = freqs[np.argmax(X)] if len(freqs) > 0 else 0
        spectral_energy = np.sum(X**2)

        features[col] = {
            "mean": float(mean_val),
            "rms": float(rms_val),
            "variance": float(variance_val),
            "max": float(max_val),
            "min": float(min_val),
            "crest_factor": float(crest_factor),
            "kurtosis": float(kurt_val),
            "dominant_frequency": float(dominant_freq),
            "spectral_energy": float(spectral_energy)
        }

    return features


# ===============================
# 7. JSON OUTPUT
# ===============================
def save_json(features, csv_file):
    filename = os.path.basename(csv_file).replace(".csv", "")
    with open(f"{JSON_DIR}/{filename}_features.json", "w") as f:
        json.dump(features, f, indent=4)


# ===============================
# HELPER: GET ALL CSV FILES
# ===============================
def get_all_csv_files(base_dir):
    csv_files = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files


# ===============================
# MAIN PIPELINE
# ===============================
def run_pipeline(csv_file):
    print(f"\nProcessing: {csv_file}")

    try:
        signal = load_signal(csv_file)

        quantized = adaptive_quantization(signal, csv_file, BIT_DEPTH)

        bitstream = pcm_encode(quantized, csv_file, BIT_DEPTH)

        encoded = manchester_encode(bitstream)

        # Perfect channel (no noise yet)
        received = bitstream

        ber = calculate_ber(bitstream, received, csv_file)

        features = extract_features(signal)

        save_json(features, csv_file)

        print(f"Done: {os.path.basename(csv_file)} | BER = {ber}")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")


# ===============================
# RUN EVERYTHING
# ===============================
if __name__ == "__main__":
    if not os.path.exists(BASE_DATA_DIR):
        print(f"ERROR: Folder not found → {BASE_DATA_DIR}")
        exit()

    files = get_all_csv_files(BASE_DATA_DIR)

    if not files:
        print("No CSV files found in data/raw/")
        exit()

    print(f"Found {len(files)} CSV files.\n")

    for f in files:
        run_pipeline(f)

    print("\nALL FILES PROCESSED SUCCESSFULLY.")