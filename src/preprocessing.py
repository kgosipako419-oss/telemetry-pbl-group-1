import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv_signal(filepath, column_name=None):
    """
    Load signal from CSV file.
    """
    data = pd.read_csv(filepath)

    if column_name is None:
        signal = data.iloc[:, 0].values
    else:
        signal = data[column_name].values

    return signal


def compute_rms(signal):
    """
    Compute RMS value of signal.
    """
    return np.sqrt(np.mean(signal**2))


def plot_signal(signal, fs=1000, title="Signal"):
    """
    Plot time-domain signal.
    """
    t = np.arange(len(signal)) / fs
    plt.figure()
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Preprocessing module ready.")