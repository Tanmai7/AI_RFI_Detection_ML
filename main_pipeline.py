import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import entropy
import joblib


def extract_features(signal, fs=1000):
    mean_power = np.mean(signal**2)
    variance = np.var(signal)
    max_amp = np.max(np.abs(signal))

    yf = np.abs(fft(signal))
    xf = fftfreq(len(signal), 1 / fs)

    peak_freq = np.abs(xf[np.argmax(yf)])
    spec_entropy = entropy(yf + 1e-10)

    return np.array([[mean_power, variance, peak_freq, spec_entropy, max_amp]])


def main():
    model = joblib.load("models/random_forest_rfi.pkl")

    fs = 1000
    t = np.linspace(0, 1, fs)

    signal = (
        np.sin(2 * np.pi * 60 * t) +
        0.7 * np.sin(2 * np.pi * 140 * t)
    )

    features = extract_features(signal)
    prediction = model.predict(features)[0]

    labels = {
        0: "No Interference",
        1: "Narrowband Interference",
        2: "Broadband Interference",
        3: "Impulsive Interference"
    }

    os.makedirs("results", exist_ok=True)

    print("\n===== RFI ANALYSIS RESULT =====")
    print("Interference Detected:", "YES" if prediction != 0 else "NO")
    print("Interference Type    :", labels[prediction])
    print("Peak Frequency (Hz)  :", round(features[0][2], 2))

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title("Time Domain RF Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("results/time_domain_signal.png")

    yf = np.abs(fft(signal))
    xf = fftfreq(len(signal), 1 / fs)

    plt.figure(figsize=(10, 4))
    plt.plot(xf[:fs // 2], yf[:fs // 2])
    plt.axvline(x=features[0][2], color='r', linestyle='--')
    plt.title("FFT with Interference Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig("results/fft_interference.png")

    f, t_spec, Sxx = spectrogram(signal, fs)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_spec, f, Sxx, shading='gouraud')
    plt.title("Spectrogram (Time-Frequency)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/spectrogram.png")


if __name__ == "__main__":
    main()
