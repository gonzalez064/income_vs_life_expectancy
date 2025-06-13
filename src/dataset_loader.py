from pathlib import Path
import pandas as pd
import chardet

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _detect_encoding(path: Path, sample: int = 20000) -> str:
    with open(path, "rb") as f:
        raw = f.read(sample)
    return chardet.detect(raw)["encoding"] or "utf-8"


def load_csv(filename: str, sep: str = ";") -> pd.DataFrame:
    path = DATA_DIR / filename
    enc = _detect_encoding(path)
    return pd.read_csv(path, sep=sep, encoding=enc)


def load_life_expectancy() -> pd.DataFrame:
    return load_csv("life_expectancy.csv")


def load_income() -> pd.DataFrame:
    return load_csv("rent.csv")
