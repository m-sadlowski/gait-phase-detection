from __future__ import annotations

"""
 Import_data - wczytywanie danych (.csv , .zip(z .csv))
"""

import zipfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _read_csv_smart(file_obj_or_path) -> pd.DataFrame:
    """
    Czyta CSV sprawdzajac najpierw ' a pozniej ;
    """
    try:
        return pd.read_csv(file_obj_or_path)
    except Exception:
        return pd.read_csv(file_obj_or_path, sep=";")


def load_acc_data(path: PathLike, encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Wczytywanie danych
    Args:
        path: Sciezkaka do pliku *.csv lub *.zip.
        encoding: Opcjonalne wymuszenie kodowania
    Returns:
        DataFrame z surowymi danymi z pliku
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("W ZIP nie znaleziono żadnego pliku CSV.")
            csv_name = csv_names[0]  # bierzemy pierwszy CSV w archiwum

            with zf.open(csv_name) as f:
                if encoding:
                    return pd.read_csv(f, encoding=encoding)
                return _read_csv_smart(f)

    if path.suffix.lower() == ".csv":
        if encoding:
            return pd.read_csv(path, encoding=encoding)
        return _read_csv_smart(path)

    raise ValueError("Obsługuję tylko pliki .csv albo .zip (z csv w srodku).")
