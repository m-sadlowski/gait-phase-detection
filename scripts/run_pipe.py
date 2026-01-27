from __future__ import annotations

"""
Pipeline uruchamiany z konsoli.

glowny skrypt:
wczytuje dane (CSV/ZIP),
przygotowuje sygnał (a_mag + filtr),
wykrywa kroki (piki),
wyznacza fazy (STANCE/SWING)
zapisuje wyniki do CSV i wykresów PNG.

Odpalanie:
  python scripts/run_pipe.py TEST1
  python scripts/run_pipe.py K_NORMALNIE2X --trim-start 1.5 --trim-end 1.5
  python scripts/run_pipe.py data/raw/TEST1.zip --cutoff 4.0 --stance 0.62
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kroki.import_data import load_acc_data
from kroki.preprocessing import add_magnitude, estimate_sampling, lowpass_filter, prepare_dataframe
from kroki.steps_detect import detect_steps
from kroki.phases import assign_phases_by_peaks
from kroki.plotting import plot_phases, plot_steps


def resolve_input(name_or_path: str) -> Path:
    """Zamienia alias na konkretny plik w data/raw/

    Jesli "name_or_path" nie istnieje to ja zwracamy
    jezeli istnieje to szukamy w data/raw pliku ktory zawiera podany alias w nazwie
    obslugiwane  .zip, .csv
    """
    p = Path(name_or_path)
    if p.exists():
        return p

    raw_dir = PROJECT_ROOT / "data" / "raw"
    candidates: list[Path] = []
    for ext in (".zip", ".csv"):
        # w razie gdyby nie wpisano calej nazwy
        candidates += list(raw_dir.glob(f"*{name_or_path}*{ext}"))
        candidates += list(raw_dir.glob(f"*{name_or_path.lower()}*{ext}"))
        candidates += list(raw_dir.glob(f"*{name_or_path.upper()}*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"Nie znaleziono pliku dla: {name_or_path} w {raw_dir}")
    # bierzemy po prostu pierwszy pasujacy plik; no tak nie powinno byc...
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Ścieżka do pliku albo alias (np. TEST1, K_WOLNIEJ1X)")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff filtra low-pass [Hz]")
    parser.add_argument("--stance", type=float, default=0.60, help="Udział podpory w cyklu (0-1)")
    parser.add_argument("--trim-start", type=float, default=1.0, help="Utnij początek [s]")
    parser.add_argument("--trim-end", type=float, default=1.0, help="Utnij koniec [s]")
    args = parser.parse_args()

    in_path = resolve_input(args.input)
    name = Path(args.input).stem if Path(args.input).exists() else args.input

    # Wczytanie i przygotowanie DataFrame
    df = load_acc_data(in_path)
    df = prepare_dataframe(df)

    #Estymacja częstotliwości próbkowania
    info = estimate_sampling(df)

    #uniwersalnosc w kontekscie trzymania telefonu
    df = add_magnitude(df)

    #Trim - odciecie startu/konca (wkladanie/wykladanietelefonu)
    t_all = df["seconds_elapsed"].to_numpy(float)
    t0 = t_all[0] + args.trim_start
    t1 = t_all[-1] - args.trim_end
    mask = (t_all >= t0) & (t_all <= t1)

    df = df.loc[mask].reset_index(drop=True)
    t = df["seconds_elapsed"].to_numpy(float)
    a = df["a_mag"].to_numpy(float)

    #Filtracja + detekcja krokow
    a_filt = lowpass_filter(a, fs_hz=info.fs_hz, cutoff_hz=args.cutoff)
    res = detect_steps(t, a_filt, fs_hz=info.fs_hz)

    #Fazy chodu wzgledem peakow; no tak tez nie powinno byc
    df["a_filt"] = a_filt
    df = assign_phases_by_peaks(df, peaks_idx=res.peaks_idx, stance_ratio=args.stance)

    #Zapis wynikow; wykresy kroko (data.raw i po filtrze) + wykres z zaznaczonymi fazami juz na filtered
    out_steps = PROJECT_ROOT / "outputs" / f"{name}_steps.png"
    out_phases = PROJECT_ROOT / "outputs" / f"{name}_phases.png"
    out_csv = PROJECT_ROOT / "data" / "processed" / f"{name}_processed.csv"

    plot_steps(t, a, a_filt, res.peaks_idx, out_steps, title=f"{name} - kroki")
    plot_phases(t, a_filt, df["phase"].to_numpy(), out_phases, title=f"{name} - fazy")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    #Podsumowanie w konsoli
    duration_after_trim = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    print(f"Wejście: {in_path}")
    print(f"Parametry: cutoff={args.cutoff} Hz, stance={args.stance}, trim=({args.trim_start}s, {args.trim_end}s)")
    print(f"fs ≈ {info.fs_hz:.2f} Hz, czas ≈ {duration_after_trim:.2f} s")
    print(f"Kroki: {len(res.peaks_idx)}, kadencja ≈ {res.cadence_spm:.1f} kroków/min")
    print(f"Zapisano: {out_csv}")
    print(f"Wykresy: {out_steps.name}, {out_phases.name}")


if __name__ == "__main__":
    main()
