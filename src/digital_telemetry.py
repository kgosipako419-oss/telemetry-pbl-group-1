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
OUTPUT_JSON = "results/logs/live_dashboard.json"

# =========================
# 1. LOAD ALL FILTERED FILES
# =========================
def load_filtered_files(folder):
    files = []
    for file in os.listdir(folder):
        if "filtered" in file and file.endswith(".csv"):
            files.append(os.path.join(folder, file))
    return sorted(files)

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
    if length == 0:
        return 0
    errors = np.sum(original_bits[:length] != recovered_bits[:length])
    return errors / length

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
# 7. PROCESS SINGLE FILE (GENERATOR)
# =========================
def process_file(file_path):

    df = pd.read_csv(file_path)
    buffer = deque(maxlen=WINDOW_SIZE)

    # Extract metadata
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    station = parts[0]
    modulation = parts[3]  # AM, ASK, FM, etc.

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
        try:
            original_bits = int(row.get('i3_photoresistor_original_bits', 0))
            recovered_bits = int(row.get('i3_photoresistor_ask_recovered', 0))
            ber = compute_ber(np.array([original_bits]), np.array([recovered_bits]))
        except:
            ber = 0

        # --- FEATURES ---
        features = extract_features(window)

        # --- OUTPUT ---
        output = {
            "timestamp": str(timestamp),
            "station": station,
            "modulation": modulation,
            "file": filename,
            "features": features,
            "ber": float(ber),
            "state": int(row.get("current_state_binary", 0))
        }

        yield output


# =========================
# 8. FIXED RUN SYSTEM (MULTI-FILE STREAMING)
# =========================
def run_system():

    files = load_filtered_files(DATA_DIR)

    print("FILES LOADED:")
    for f in files:
        print(f)

    # Create generator for each file
    generators = [process_file(file) for file in files]

    os.makedirs("results/logs", exist_ok=True)

    while True:
        for gen, file in zip(generators, files):
            try:
                data = next(gen)

                # Write latest data (live overwrite)
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(data, f, indent=4)

                print(f"[{os.path.basename(file)}] → {data}")

            except StopIteration:
                continue  # file finished

        time.sleep(0.05)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_system()