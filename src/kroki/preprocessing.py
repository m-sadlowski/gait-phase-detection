from __future__ import annotations

"""
Preprocessing danych i sgnalu do analizy

prepare_dataframe - wybieranie i porzadkowanie kolumny
estimate_sampling - czestotliwosc probkowania (szacowanie)
a_mag = sqrt(x^2 + y^2 + z^2) + filtr dolnoprzepustowy (lowpass_filter)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class SignalInfo:
    """Podstawowe informacje o sygnale"""
    fs_hz: float
    dt_median: float
    n_samples: int
    duration_s: float


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolicie formatu danych
    seconds_elapsed: czas w sekundach (float)
    x, y, z: przyspieszenia na osiach

    Co robi funkcja
    wybiera tylko potrzebne kolumny,
    sortuje po czasie,
    usuwa braki.
    """
    needed = {"seconds_elapsed", "x", "y", "z"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn: {missing}. Mam: {list(df.columns)}")

    out = df[["seconds_elapsed", "x", "y", "z"]].copy()
    out = out.sort_values("seconds_elapsed").reset_index(drop=True)
    out = out.dropna()
    return out


def estimate_sampling(df: pd.DataFrame) -> SignalInfo:
    """
    Szacowanie czestotliwosci probkowania na podstawie 'seconds_elapsed`
    Uzywam medainy dt bo jest odporna na pojedyncze skoki/zaklocenia
    """
    t = df["seconds_elapsed"].to_numpy(dtype=float)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Nie da się policzyć dt (za mało próbek albo niepoprawny czas)")

    dt_median = float(np.median(dt))
    fs_hz = 1.0 / dt_median
    duration_s = float(t[-1] - t[0])

    return SignalInfo(
        fs_hz=fs_hz,
        dt_median=dt_median,
        n_samples=int(len(df)),
        duration_s=duration_s,
    )


def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje kolumnę `a_mag`;modul wektora przyspieszenia

    Czyli pozbydamy sie zalenosci od ustawienia telefonu/etc (do testu tylko dane z telefonu)
    """
    out = df.copy()

    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    z = out["z"].to_numpy(dtype=float)

    out["a_mag"] = np.sqrt(x * x + y * y + z * z)
    return out


def lowpass_filter(
    signal: np.ndarray,
    fs_hz: float,
    cutoff_hz: float = 5.0,
    order: int = 4,
) -> np.ndarray:
    """
    Filtr dolnoprzepustowy Butterworth (zero-phase)

    Dlaczego
    surowy akcelerometr ma szum i krótkie drgania,
    chód jest stosunkowo wolny (kilka Hz),
    filtr ułatwia wykrywanie kroków przez piki.

    cutoff_hz: dla 100Hz na aplikacji - 5Hz jest chyba okay
    Zwracm sygnał o tej samej długości co wejście
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz musi być > 0")

    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq

    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, signal)
