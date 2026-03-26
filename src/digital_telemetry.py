import pandas as pd
import numpy as np
import json
import os

# ===============================
# CONFIG
# ===============================
BIT_DEPTH = 8

DATA_DIR = "data/raw/student3_output"
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

    filename = os.path.basename(csv_file)
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

    filename = os.path.basename(csv_file)
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

    filename = os.path.basename(csv_file)
    with open(f"{LOG_DIR}/{filename}_integrity.txt", "w") as f:
        f.write(f"Total Bits: {len(original)}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"BER: {ber}\n")

    return ber


# ===============================
# 6. FEATURE EXTRACTION
# ===============================
def extract_features(signal):
    features = {}

    for col in signal.columns:
        x = signal[col].values

        features[col] = {
            "mean": float(np.mean(x)),
            "rms": float(np.sqrt(np.mean(x**2))),
            "variance": float(np.var(x)),
            "max": float(np.max(x)),
            "min": float(np.min(x))
        }

    return features


# ===============================
# 7. JSON OUTPUT
# ===============================
def save_json(features, csv_file):
    filename = os.path.basename(csv_file).replace(".csv", "_features.json")

    with open(f"{JSON_DIR}/{filename}", "w") as f:
        json.dump(features, f, indent=4)


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

        # Simulated perfect channel (for now)
        received = bitstream

        ber = calculate_ber(bitstream, received, csv_file)

        features = extract_features(signal)

        save_json(features, csv_file)

        print(f"Done: {csv_file} | BER = {ber}")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")


# ===============================
# RUN ALL FILES
# ===============================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Folder not found → {DATA_DIR}")
        exit()

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    ]

    if not files:
        print("No CSV files found.")
        exit()

    print(f"Found {len(files)} CSV files.\n")

    for f in files:
        run_pipeline(f)

    print("\nALL FILES PROCESSED SUCCESSFULLY.")