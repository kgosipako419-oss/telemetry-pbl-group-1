from preprocessing import compute_rms, plot_signal
import numpy as np


def main():
    # Temporary synthetic signal (will replace with real dataset soon)
    fs = 1000
    t = np.linspace(0, 1, fs)
    signal = 0.7 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.random.randn(len(t))

    rms = compute_rms(signal)
    print(f"RMS Value: {rms:.4f}")

    plot_signal(signal, fs, title="Test Signal")


if __name__ == "__main__":
    main()