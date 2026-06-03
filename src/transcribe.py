import argparse
import numpy as np
import librosa

from utils import (
    load_and_preprocess,
    estimate_tempo_quarter_length,
    track_f0_pyin,
    group_segments_by_pitch,
    events_to_score,
    _quantize_duration_to_quarters,
    find_best_bpm,
)

def postprocess_events(events):
    cleaned = []

    # порог паузы = половина медианной длительности ноты, но не меньше 0.25
    note_qlens_raw = [ev.qlen for ev in events if ev.pitch_midi is not None]
    if note_qlens_raw:
        median_note = float(np.median(note_qlens_raw))
        rest_threshold = max(0.25, median_note * 0.5)
    else:
        rest_threshold = 0.5

    for ev in events:
        # убираем слишком короткие паузы
        if ev.pitch_midi is None and ev.qlen <= rest_threshold:
            continue

        # убираем слишком короткие ноты
        if ev.pitch_midi is not None and ev.qlen < 0.125:
            continue

        cleaned.append(ev)

    # убрать паузы в конце
    while cleaned and cleaned[-1].pitch_midi is None:
        cleaned.pop()

    for ev in cleaned:
        ev.qlen = _quantize_duration_to_quarters(ev.qlen)

        # выравнивание: если все ноты примерно одной длины — привести к медиане
    note_qlens = [ev.qlen for ev in cleaned if ev.pitch_midi is not None]
    if note_qlens:
        median_qlen = float(np.median(note_qlens))
        # приводим все ноты к медиане безусловно
        for ev in cleaned:
            if ev.pitch_midi is not None:
                ev.qlen = _quantize_duration_to_quarters(median_qlen)

        # убрать паузы в конце ещё раз (после выравнивания могли появиться новые)
    while cleaned and cleaned[-1].pitch_midi is None:
        cleaned.pop()

    return cleaned

# Функция загрузки и предобработки
def wav_to_musicxml(input_wav: str, output_xml: str, time_sig: str = "4/4"):
    # y - массив отсчётов аудиосигнала;
    # sr - частоту дискретизации.
    y, sr = load_and_preprocess(input_wav)

    # Оценка темпа записи
    # Возюращаеться :
    # tempo_bpm - темп в ударах в минуту;
    # qlen_sec - длительность одной четвертной ноты в секундах.
    tempo_bpm, qlen_sec = estimate_tempo_quarter_length(y, sr)

    #Алгоритм PYIN
    # f0 - основная частота по кадрам;
    # voiced_flag - логический массив, где сигнал считается озвученным;
    # voiced_prob - вероятность озвученности;
    # hop - шаг между кадрами.    
    f0, voiced_flag, voiced_prob, hop = track_f0_pyin(y, sr)
    # Переводит частоты f0 из герц в MIDI-номера.
    midi = librosa.hz_to_midi(f0)

    # Сглаживание контура высоты
    try:
        from scipy.signal import medfilt # Импорт медианного фильтра

        midi_sm = midi.copy()
        valid = np.isfinite(midi_sm) # Создаётся маска valid, которая показывает, какие значения в массиве являются нормальными числами.

        # Проверяется, есть ли вообще хотя бы одно корректное значение.
        if valid.any(): 
            t = np.arange(len(midi_sm)) # Создаётся массив индексов кадров
            midi_filled = np.interp(t, t[valid], midi_sm[valid]) # если где-то высота временно пропала, программа приблизительно восстанавливает её по соседним значениям.
            midi_filled = medfilt(midi_filled, kernel_size=3) # К интерполированному массиву применяется медианный фильтр с окном 5.
            midi_filled = np.round(midi_filled) # Значения округляются до ближайших MIDI-нот.
            midi_sm = np.where(valid, midi_filled, np.nan) # Там, где исходно были валидные значения, сохраняется сглаженный результат. Там, где исходно были невалидные значения, снова ставится NaN.
            midi = midi_sm # Итоговый сглаженный массив записывается обратно в midi

    except Exception as e:
        print(f"[WARN] Не удалось сгладить контур: {e}")

    # фильтрация по озвученности сомнительных voiced-кадров
    voiced_mask = voiced_prob > 0.5
    midi = np.where(voiced_mask, midi, np.nan) # Если кадр озвученный - сохраняется его MIDI-значение. Если нет - ставится NaN.
    voiced_flag = voiced_mask # Старый voiced_flag заменяется на более строгую маску voiced_mask.

    # fallback на YIN, если pYIN не справился
    if np.all(np.isnan(midi)): # если после pYIN и фильтрации все значения midi оказались NaN, значит pYIN не смог извлечь высоту.
        f0_yin = librosa.yin(
            # Алгоритму задаются:
            # y - сигнал;
            # fmin - минимальная допустимая частота; 
            # fmax - максимальная допустимая частота;
            # sr - частота дискретизации.
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_flag = np.isfinite(f0_yin) # Создаётся маска озвученности
        midi = librosa.hz_to_midi(f0_yin) # Частоты, полученные из YIN, переводятся в MIDI.
        hop = 512 # Явно задаётся шаг кадров.

    # Сегментация
    events = group_segments_by_pitch( #покадровый массив высоты превращается в список событий events.
        midi_series=midi, # высота по кадрам;
        voiced=voiced_flag, #где звук, а где пауза;
        hop_length=hop, # шаг кадров;
        sr=sr, # частота дискретизации;
        qlen_sec=qlen_sec, # длительность четверти;
        min_note_sec=0.10, # минимальная длительность ноты;
        min_silence_sec=0.18, # минимальная длительность паузы.
    )

    events = postprocess_events(events)


    durations_sec = []
    for ev in events:
        durations_sec.append(ev.qlen * qlen_sec)

    tempo_bpm = find_best_bpm(durations_sec, initial_bpm=tempo_bpm)
    qlen_sec = 60.0 / tempo_bpm

    # повторное квантование с уточнённым BPM
    for ev in events:
        raw_dur_sec = ev.qlen * (60.0 / round(tempo_bpm))
        ev.qlen = _quantize_duration_to_quarters(raw_dur_sec / qlen_sec)

    events = postprocess_events(events)

    # DEBUG — удали после проверки
    print(f"[DEBUG] BPM после уточнения: {tempo_bpm}")
    print(f"[DEBUG] Медиана длительности нот: {float(np.median([ev.qlen for ev in events if ev.pitch_midi is not None])):.2f} четвертей")

    
    score = events_to_score(events, tempo_bpm=tempo_bpm, time_sig=time_sig) # Список событий events превращается в объект нотной записи score.
    score.write("musicxml", fp=output_xml) # Готовая нотная запись сохраняется в файл MusicXML по пути output_xml.

    note_count = sum(1 for ev in events if ev.pitch_midi is not None) # Считается количество нот.
    rest_count = sum(1 for ev in events if ev.pitch_midi is None) # Считается количество пауз

    # Создаётся словарь result с итоговой информацией:
        # путь к файлу;
        # темп;
        # число событий;
        # число нот;
        # число пауз.
    result = {
        "output_xml": output_xml,
        "tempo_bpm": round(float(tempo_bpm), 1),
        "event_count": len(events),
        "note_count": note_count,
        "rest_count": rest_count,
    }

    print(
        f"Готово: {output_xml} | Темп: ~{result['tempo_bpm']} BPM | "
        f"Событий: {result['event_count']} | Нот: {result['note_count']} | "
        f"Пауз: {result['rest_count']}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="WAV → MusicXML (монофония)")
    parser.add_argument("input_wav", help="Путь к входному WAV (моно/стерео)")
    parser.add_argument(
        "-o",
        "--output",
        default="output.musicxml",
        help="Выходной MusicXML (по умолчанию output.musicxml)",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Размер такта, например 2/4, 3/4, 4/4 (по умолчанию 4/4)"
    )
    args = parser.parse_args()

    wav_to_musicxml(args.input_wav, args.output, time_sig=args.time_signature)


if __name__ == "__main__":
    main()