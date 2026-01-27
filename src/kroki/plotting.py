from __future__ import annotations

"""
Wizualizacja wyników (wykresy PNG)

rysujemy downsamplingiem (ds) bo przy 100Hz bylo tak gesto ze nie moglem stwierdzic czy peaki sa dobrze zaznaczen
Downsampling dotyczy tylko linii; peaki sa zaznaczane normalnie
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_steps(
    t: np.ndarray,
    a_raw: np.ndarray,
    a_filt: np.ndarray,
    peaks_idx: np.ndarray,
    out_path: str | Path,
    title: str = "Detekcja kroków (a_mag)",
    ds: int = 5,
    figsize: tuple[int, int] = (16, 4),
) -> None:
    """Rysowanie a_mag(raw + filtered) i zaznaczanie peakow"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)

    # Downsampling
    plt.plot(t[::ds], a_raw[::ds], label="a_mag (raw)")
    plt.plot(t[::ds], a_filt[::ds], label="a_mag (filtered)")

    # Piki bez downsamplingu
    if len(peaks_idx) > 0:
        plt.scatter(t[peaks_idx], a_filt[peaks_idx], marker="x", label="kroki (peaks)")

    plt.xlabel("czas [s]")
    plt.ylabel("a_mag")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_phases(
    t: np.ndarray,
    a_filt: np.ndarray,
    phase: np.ndarray,
    out_path: str | Path,
    title: str = "Fazy kroku",
    ds: int = 5,
    figsize: tuple[int, int] = (16, 4),
) -> None:
    """Rysuje a_filt + tlo faz STANCE/SWING"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.plot(t[::ds], a_filt[::ds], label="a_mag (filtered)")

    #Rysowanie tla w segmencie o stalej fazie
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            ph = phase[start]
            if ph == "STANCE":
                plt.axvspan(t[start], t[i - 1], alpha=0.15, color="green")
            elif ph == "SWING":
                plt.axvspan(t[start], t[i - 1], alpha=0.15, color="orange")
            start = i

    #Legenda faz
    phase_legend = [
        Patch(facecolor="green", alpha=0.15, label="STANCE (podpora)"),
        Patch(facecolor="orange", alpha=0.15, label="SWING (wykrok)"),
    ]

    plt.xlabel("czas [s]")
    plt.ylabel("a_mag (filtered)")
    plt.title(title)

    handles1, labels1 = plt.gca().get_legend_handles_labels()
    handles = handles1 + phase_legend
    labels = labels1 + [p.get_label() for p in phase_legend]
    plt.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
