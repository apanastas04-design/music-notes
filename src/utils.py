from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa
import music21 as m21


@dataclass
class Event:
    pitch_midi: Optional[int]  # None = пауза
    qlen: float                # длительность в четвертях


def load_and_preprocess(
    path: str,
    sr: Optional[int] = None,
    trim_db: float = 30.0
) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr


def estimate_tempo_quarter_length(y: np.ndarray, sr: int) -> Tuple[float, float]:
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        from librosa.feature.rhythm import tempo as lr_tempo
        tempo = float(lr_tempo(onset_envelope=oenv, sr=sr, aggregate=np.median)[0])
    except Exception:
        tempo = 100.0

    if not np.isfinite(tempo) or tempo < 40 or tempo > 220:
        tempo = 100.0

    return tempo, 60.0 / tempo

# Уточняет BPM в диапазоне ±30 от начального значения, минимизируя суммарную ошибку квантования длительностей.
def find_best_bpm(durations_sec: List[float], initial_bpm: float) -> float:

    if not durations_sec:
        return initial_bpm

    allowed = np.array([4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125])
    best_bpm = initial_bpm
    best_error = float('inf')

    bpm_low = max(40, int(initial_bpm) - 30)
    bpm_high = min(220, int(initial_bpm) + 30)

    for bpm in range(bpm_low, bpm_high + 1):
        qlen = 60.0 / bpm
        total_error = 0.0
        for d in durations_sec:
            ratio = d / qlen
            nearest = allowed[np.argmin(np.abs(allowed - ratio))]
            total_error += abs(ratio - nearest)
        if total_error < best_error:
            best_error = total_error
            best_bpm = float(bpm)

    return best_bpm

def track_f0_pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin_note: str = "C2",
    fmax_note: str = "C7"
):
    fmin = librosa.note_to_hz(fmin_note)
    fmax = librosa.note_to_hz(fmax_note)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
        center=True
    )

    # устраняем внезапные скачки > 12 полутонов между соседними кадрами
    midi = librosa.hz_to_midi(f0)
    for i in range(1, len(midi)):
        if (
            np.isfinite(midi[i])
            and np.isfinite(midi[i - 1])
            and abs(midi[i] - midi[i - 1]) > 12
        ):
            f0[i] = np.nan

    return f0, voiced_flag, voiced_prob, hop_length


def _quantize_duration_to_quarters(q: float) -> float:
    candidates = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.125])
    val = float(candidates[np.argmin(np.abs(candidates - q))])
    return max(val, 0.125)


def group_segments_by_pitch(midi_series: np.ndarray,
                            voiced: np.ndarray,
                            hop_length: int,
                            sr: int,
                            qlen_sec: float,
                            min_note_sec: float = 0.08,
                            min_silence_sec: float = 0.06,
                            pitch_change_threshold: int = 1,
                            min_pitch_frames: int = 2) -> List[Event]:
    events: List[Event] = []
    frame_dur = hop_length / sr
    n = len(midi_series)
    i = 0

    def flush_note(start_idx: int, end_idx: int, pitch_vals: List[float]):
        dur_sec = (end_idx - start_idx) * frame_dur
        if dur_sec < min_note_sec:
            return
        med_pitch = np.nanmedian(pitch_vals)
        if not np.isfinite(med_pitch):
            return
        pitch_midi = int(np.round(med_pitch))
        qlen_q = _quantize_duration_to_quarters(dur_sec / qlen_sec)
        events.append(Event(pitch_midi=pitch_midi, qlen=qlen_q))

    def flush_rest(start_idx: int, end_idx: int):
        dur_sec = (end_idx - start_idx) * frame_dur
        if dur_sec < min_silence_sec:
            return
        qlen_q = _quantize_duration_to_quarters(dur_sec / qlen_sec)
        events.append(Event(pitch_midi=None, qlen=qlen_q))

    while i < n:
        # Пауза
        if not voiced[i] or not np.isfinite(midi_series[i]):
            start = i
            i += 1
            while i < n and (not voiced[i] or not np.isfinite(midi_series[i])):
                i += 1
            flush_rest(start, i)
            continue

        # Начало озвученного сегмента
        note_start = i
        current_vals = [midi_series[i]]
        current_pitch = int(np.round(midi_series[i]))
        i += 1

        while i < n and voiced[i] and np.isfinite(midi_series[i]):
            cand_pitch = int(np.round(midi_series[i]))

            # если высота почти не изменилась, продолжаем текущую ноту
            if abs(cand_pitch - current_pitch) < pitch_change_threshold:
                current_vals.append(midi_series[i])
                i += 1
                continue

            # проверяем, устойчива ли новая высота несколько кадров подряд
            j = i
            new_vals = []
            while (
                j < n
                and voiced[j]
                and np.isfinite(midi_series[j])
                and abs(int(np.round(midi_series[j])) - cand_pitch) < pitch_change_threshold
            ):
                new_vals.append(midi_series[j])
                j += 1

            if len(new_vals) >= min_pitch_frames:
                # завершаем текущую ноту
                flush_note(note_start, i, current_vals)

                # начинаем новую ноту
                note_start = i
                current_vals = new_vals.copy()
                current_pitch = int(np.round(np.nanmedian(new_vals)))
                i = j
            else:
                # короткий выброс — считаем частью текущей ноты
                current_vals.append(midi_series[i])
                i += 1

        flush_note(note_start, i, current_vals)

    # Склейка только одинаковых соседних нот без паузы между ними
    merged: List[Event] = []
    for ev in events:
        if (
            merged
            and ev.pitch_midi is not None
            and merged[-1].pitch_midi == ev.pitch_midi
        ):
            merged[-1].qlen += ev.qlen
        else:
            merged.append(ev)

    return merged

def events_to_score(events: List[Event], tempo_bpm: float, time_sig: str = "4/4") -> m21.stream.Score:
    score = m21.stream.Score()
    part = m21.stream.Part()

    part.insert(0, m21.tempo.MetronomeMark(number=round(tempo_bpm)))
    part.insert(0, m21.meter.TimeSignature(time_sig))

    for ev in events:
        if ev.pitch_midi is None:
            n = m21.note.Rest(quarterLength=ev.qlen)
        else:
            n = m21.note.Note(m21.pitch.Pitch(midi=ev.pitch_midi))
            n.quarterLength = ev.qlen
        part.append(n)

    score.insert(0, part)

    try:
        k = score.analyze("key")
        score.insert(0, k)
    except Exception:
        pass

    return score