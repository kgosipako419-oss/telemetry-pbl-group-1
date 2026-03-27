import os
import json
import time
import numpy as np
import pandas as pd
from collections import deque

# =========================
# CONFIG
# =========================
DATA_DIR = "data/raw"
WINDOW_SIZE = 100
BIT_DEPTH = 8
OUTPUT_JSON = "live_dashboard.json"

# =========================
# 1. LOAD ALL FILTERED FILES
# =========================
def load_filtered_files(folder):
    files = []
    for file in os.listdir(folder):
        if "filtered" in file and file.endswith(".csv"):
            files.append(os.path.join(folder, file))
    return files

# =========================
# 2. ADAPTIVE QUANTIZATION
# =========================
def adaptive_quantization(signal, bits=8):
    xmax = np.max(np.abs(signal)) + 1e-6
    L = 2 ** bits
    delta = (2 * xmax) / L

    quantized = np.floor((signal + xmax) / delta + 0.5) * delta - xmax
    return quantized, delta

# =========================
# 3. PCM ENCODING
# =========================
def pcm_encode(signal, delta, bits=8):
    xmax = np.max(np.abs(signal)) + 1e-6
    indices = np.floor((signal + xmax) / delta).astype(int)
    indices = np.clip(indices, 0, 2**bits - 1)

    bitstream = []
    for val in indices:
        binary = format(val, f'0{bits}b')
        bitstream.extend([int(b) for b in binary])

    return np.array(bitstream)

# =========================
# 4. MANCHESTER ENCODING
# =========================
def manchester_encode(bitstream):
    encoded = []
    for bit in bitstream:
        if bit == 0:
            encoded.extend([1, 0])
        else:
            encoded.extend([0, 1])
    return np.array(encoded)

# =========================
# 5. BER CALCULATION
# =========================
def compute_ber(original_bits, recovered_bits):
    length = min(len(original_bits), len(recovered_bits))
    errors = np.sum(original_bits[:length] != recovered_bits[:length])
    return errors / length if length > 0 else 0

# =========================
# 6. FEATURE EXTRACTION
# =========================
def extract_features(window):
    data = np.array(window)

    return {
        "mean": float(np.mean(data)),
        "variance": float(np.var(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "max": float(np.max(data)),
        "min": float(np.min(data))
    }

# =========================
# 7. MAIN PROCESSING PIPELINE
# =========================
def process_file(file_path):

    df = pd.read_csv(file_path)

    buffer = deque(maxlen=WINDOW_SIZE)

    for _, row in df.iterrows():

        signal = row['i3_photoresistor_baseband']
        timestamp = row['timestamp']

        buffer.append(signal)

        if len(buffer) < WINDOW_SIZE:
            continue

        window = np.array(buffer)

        # --- QUANTIZATION ---
        quantized, delta = adaptive_quantization(window, BIT_DEPTH)

        # --- PCM ---
        bitstream = pcm_encode(quantized, delta, BIT_DEPTH)

        # --- LINE CODING ---
        encoded_stream = manchester_encode(bitstream)

        # --- BER ---
        if 'i3_photoresistor_original_bits' in row and 'i3_photoresistor_ask_recovered' in row:
            original_bits = np.array([int(row['i3_photoresistor_original_bits'])])
            recovered_bits = np.array([int(row['i3_photoresistor_ask_recovered'])])
            ber = compute_ber(original_bits, recovered_bits)
        else:
            ber = 0

        # --- FEATURES ---
        features = extract_features(window)

        # --- OUTPUT JSON ---
        output = {
            "timestamp": timestamp,
            "file": os.path.basename(file_path),
            "features": features,
            "ber": float(ber),
            "state": int(row.get("current_state_binary", 0))
        }

        yield output


# =========================
# 8. RUN SYSTEM (LIVE SIMULATION)
# =========================
def run_system():

    files = load_filtered_files(DATA_DIR)

    while True:
        for file in files:
            for data in process_file(file):

                # Save to JSON (overwrite for live dashboard)
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(data, f, indent=4)

                print(data)

                time.sleep(0.05)  # simulate live feed


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_system()