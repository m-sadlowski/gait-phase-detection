from __future__ import annotations

"""
Fazy chodu: STANCE (podpora) i SWING (przenoszenie).

Zalozenia:
kolejne wykryte kroki (peaki) wyznaczaja granice cyklu kroku
każdy cykl dzielimy na stala proporcję:
  - STANCE 60%
  - SWING 40%
Tak, jest to przyblizenie
"""

import numpy as np
import pandas as pd

def assign_phases_by_peaks(
    df: pd.DataFrame,
    peaks_idx: np.ndarray,
    stance_ratio: float = 0.60,
) -> pd.DataFrame:
    """
    Nadaje fazy STANCE/SWING na podstawie odcinkow miedzy peakami

    Args:
        df: DataFrame zawierający co najmniej: seconds_elapsed, a_mag.
        peaks_idx: indeksy kroków (z detect_steps).
        stance_ratio: jaka część cyklu to STANCE (0-1). Reszta to SWING.
    Returns:
        Kopia df z dodatkowymi kolumnami:
        - phase: OTHER / STANCE / SWING
        - step_id: numer cyklu kroku (0,1,2,...) -1 dla OTHER
    """
    if "seconds_elapsed" not in df.columns:
        raise ValueError("Brak kolumny seconds_elapsed")
    if "a_mag" not in df.columns:
        raise ValueError("Brak kolumny a_mag (najpierw add_magnitude)")

    if not (0.0 < stance_ratio < 1.0):
        raise ValueError("stance_ratio musi być w zakresie (0, 1)")

    out = df.copy()
    n = len(out)

    phase = np.array(["OTHER"] * n, dtype=object)
    step_id = np.full(n, -1, dtype=int)

    peaks_idx = np.asarray(peaks_idx, dtype=int)
    peaks_idx = peaks_idx[(peaks_idx >= 0) & (peaks_idx < n)]
    peaks_idx = np.unique(peaks_idx)
    peaks_idx.sort()

    # Potrzebujemy co najmniej dwóch krokó
    if len(peaks_idx) < 2:
        out["phase"] = phase
        out["step_id"] = step_id
        return out

    for i in range(len(peaks_idx) - 1):
        start = int(peaks_idx[i])
        end = int(peaks_idx[i + 1])

        #Jesli peaki są podejrzanie blisko to pomijamy
        if end <= start + 2:
            continue

        #Punkt podzialu cyklu na STANCE / SWING.
        cut = start + int((end - start) * stance_ratio)

        phase[start:cut] = "STANCE"
        phase[cut:end] = "SWING"
        step_id[start:end] = i

    out["phase"] = phase
    out["step_id"] = step_id
    return out
