import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from utils import (
    load_and_preprocess,
    track_f0_pyin,
)


def make_plots(input_wav: str, out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)

    y, sr = load_and_preprocess(input_wav)
    basename = os.path.splitext(os.path.basename(input_wav))[0]

    # ===== 1. Осциллограмма =====
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Осциллограмма сигнала: {basename}")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    waveform_path = os.path.join(out_dir, f"{basename}_waveform.png")
    plt.tight_layout()
    plt.savefig(waveform_path, dpi=150)
    plt.close()

    # ===== 2. Спектрограмма =====
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Спектрограмма: {basename}")
    plt.xlabel("Время, с")
    plt.ylabel("Частота, Гц")
    spectrogram_path = os.path.join(out_dir, f"{basename}_spectrogram.png")
    plt.tight_layout()
    plt.savefig(spectrogram_path, dpi=150)
    plt.close()

    # ===== 3. pYIN: f0 + voiced_prob =====
    f0, voiced_flag, voiced_prob, hop_length = track_f0_pyin(y, sr)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Спектрограмма + контур f0
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.plot(times, f0, linewidth=2, label="f0 (pYIN)")
    plt.title(f"Спектрограмма и контур высоты тона: {basename}")
    plt.xlabel("Время, с")
    plt.ylabel("Частота, Гц")
    plt.legend()
    f0_path = os.path.join(out_dir, f"{basename}_f0_overlay.png")
    plt.tight_layout()
    plt.savefig(f0_path, dpi=150)
    plt.close()

    # voiced_prob
    plt.figure(figsize=(12, 4))
    plt.plot(times, voiced_prob, linewidth=1.5, label="voiced_prob")
    plt.axhline(0.5, linestyle="--", label="threshold = 0.5")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Вероятность озвученности: {basename}")
    plt.xlabel("Время, с")
    plt.ylabel("Вероятность")
    plt.legend()
    voiced_path = os.path.join(out_dir, f"{basename}_voiced_prob.png")
    plt.tight_layout()
    plt.savefig(voiced_path, dpi=150)
    plt.close()

    print("Готово. Построены файлы:")
    print(waveform_path)
    print(spectrogram_path)
    print(f0_path)
    print(voiced_path)


def main():
    parser = argparse.ArgumentParser(description="Построение отладочных графиков для WAV")
    parser.add_argument("input_wav", help="Путь к WAV-файлу")
    parser.add_argument("-o", "--out_dir", default="plots", help="Папка для сохранения графиков")
    args = parser.parse_args()

    make_plots(args.input_wav, args.out_dir)


if __name__ == "__main__":
    main()