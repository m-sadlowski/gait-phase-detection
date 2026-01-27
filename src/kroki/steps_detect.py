from __future__ import annotations

"""
Detekcja kroków (baseline).
Wykrywam kroki jako peaki w przefiltrowanym sygnale a_mag (a_filt)

"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class StepsResult:
    """Wynik detekcji kroków."""
    peaks_idx: np.ndarray       # indeksy próbek, gdzie wykryto kroki
    peaks_t: np.ndarray         # czas (seconds_elapsed) dla kroków
    cadence_spm: float          # kroki/min
    mean_step_time: float       # średni czas między krokami [s]


def detect_steps(
    t: np.ndarray,
    a_filt: np.ndarray,
    fs_hz: float,
    min_dist_s: float = 0.35,
    prom_factor: float = 0.30,
) -> StepsResult:
    """
    Wykrywa kroki jako piki w a_filt

    Args:
        t: wektor czasu w sekundach (seconds_elapsed), długości N
        a_filt: przefiltrowany sygnał a_mag, długości N
        fs_hz: częstotliwość próbkowania
        min_dist_s: minimalny odstęp czasu między kolejnymi krokami [s]
        prom_factor: jak "wysoki" musi być pik: prom = prom_factor * std(a_filt)
    Returns:
        StepsResult z indeksami peaków i kadencją
    """
    if len(t) != len(a_filt):
        raise ValueError("t i a_filt muszą mieć tę samą długość.")

    # Minimalny odstęp między krokami w próbkach
    distance = max(1, int(min_dist_s * fs_hz))

    # Próg prominence adaptacyjny (zależny od sygnału).
    prom = float(prom_factor * np.std(a_filt))
    prom = max(prom, 1e-12)  # na prawie stayl sygnal

    peaks, _ = find_peaks(a_filt, distance=distance, prominence=prom)

    if len(peaks) >= 2:
        step_times = np.diff(t[peaks])
        mean_step = float(np.mean(step_times))
        cadence = 60.0 / mean_step
    else:
        mean_step = float("nan")
        cadence = 0.0

    return StepsResult(
        peaks_idx=peaks,
        peaks_t=t[peaks],
        cadence_spm=float(cadence),
        mean_step_time=mean_step,
    )
