import argparse
import os
from typing import List, Dict

import numpy as np
import librosa
import matplotlib.pyplot as plt


def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError(f"Пустой файл: {path}")
    y = y / max(np.max(np.abs(y)), 1e-9)
    return y, sr


def compute_rms_envelope(y: np.ndarray, sr: int, hop_length: int = 256, frame_length: int = 1024):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    return rms, times


def detect_onsets(y: np.ndarray, sr: int, hop_length: int = 256):
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=2,
    )
    return onset_frames


def estimate_decay_metrics(
    rms: np.ndarray,
    times: np.ndarray,
    onset_frames: np.ndarray,
    min_peak: float = 0.03,
    max_note_duration_sec: float = 1.5,
) -> List[Dict[str, float]]:
    results = []

    for i, onset in enumerate(onset_frames):
        if onset >= len(rms):
            continue

        peak = rms[onset]
        if peak < min_peak:
            continue

        start_t = times[onset]

        if i + 1 < len(onset_frames):
            next_onset_t = times[onset_frames[i + 1]]
            end_t = min(start_t + max_note_duration_sec, next_onset_t)
        else:
            end_t = min(start_t + max_note_duration_sec, times[-1])

        segment_mask = (times >= start_t) & (times <= end_t)
        seg_rms = rms[segment_mask]
        seg_times = times[segment_mask]

        if len(seg_rms) < 3:
            continue

        peak_val = np.max(seg_rms)
        if peak_val < min_peak:
            continue

        t50 = np.nan
        t20 = np.nan

        thr50 = 0.5 * peak_val
        thr20 = 0.2 * peak_val

        below50 = np.where(seg_rms <= thr50)[0]
        below20 = np.where(seg_rms <= thr20)[0]

        if len(below50) > 0:
            t50 = seg_times[below50[0]] - start_t
        if len(below20) > 0:
            t20 = seg_times[below20[0]] - start_t

        # грубая оценка наклона затухания
        valid = seg_rms > 1e-6
        if np.sum(valid) >= 3:
            x = seg_times[valid] - start_t
            y = np.log(seg_rms[valid])
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = np.nan

        results.append({
            "start_t": float(start_t),
            "peak": float(peak_val),
            "t50": float(t50) if np.isfinite(t50) else np.nan,
            "t20": float(t20) if np.isfinite(t20) else np.nan,
            "slope": float(slope) if np.isfinite(slope) else np.nan,
        })

    return results


def summarize_metrics(metrics: List[Dict[str, float]], label: str):
    t50_vals = np.array([m["t50"] for m in metrics if np.isfinite(m["t50"])])
    t20_vals = np.array([m["t20"] for m in metrics if np.isfinite(m["t20"])])
    slope_vals = np.array([m["slope"] for m in metrics if np.isfinite(m["slope"])])

    print(f"\n=== {label} ===")
    print(f"Количество проанализированных нот: {len(metrics)}")

    if len(t50_vals):
        print(f"Среднее время падения до 50%: {np.mean(t50_vals):.3f} с")
    else:
        print("Среднее время падения до 50%: нет данных")

    if len(t20_vals):
        print(f"Среднее время падения до 20%: {np.mean(t20_vals):.3f} с")
    else:
        print("Среднее время падения до 20%: нет данных")

    if len(slope_vals):
        print(f"Средний наклон log-огибающей: {np.mean(slope_vals):.3f}")
        print("Чем более отрицательное значение, тем быстрее затухание.")
    else:
        print("Средний наклон log-огибающей: нет данных")


def plot_envelope_with_onsets(y, sr, rms, times, onset_frames, out_path, title):
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, label="RMS envelope")
    onset_times = times[onset_frames[onset_frames < len(times)]]
    plt.vlines(onset_times, ymin=0, ymax=np.max(rms), linestyles="--", label="Onsets")
    plt.title(title)
    plt.xlabel("Время, с")
    plt.ylabel("Энергия")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Сравнение затухания нот для двух записей")
    parser.add_argument("file1", help="Первый WAV-файл")
    parser.add_argument("file2", help="Второй WAV-файл")
    parser.add_argument("--label1", default="Запись 1")
    parser.add_argument("--label2", default="Запись 2")
    parser.add_argument("--out_dir", default="plots_decay")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for path, label, png_name in [
        (args.file1, args.label1, "env1.png"),
        (args.file2, args.label2, "env2.png"),
    ]:
        y, sr = load_audio(path)
        rms, times = compute_rms_envelope(y, sr)
        onset_frames = detect_onsets(y, sr)
        metrics = estimate_decay_metrics(rms, times, onset_frames)

        summarize_metrics(metrics, label)

        plot_envelope_with_onsets(
            y, sr, rms, times, onset_frames,
            os.path.join(args.out_dir, png_name),
            f"Огибающая энергии и атаки: {label}"
        )

    print(f"\nГрафики сохранены в папку: {args.out_dir}")


if __name__ == "__main__":
    main()