import pandas as pd

def _clean_province(col: pd.Series) -> pd.Series:
    return col.str.replace(r"^\d+\s+", "", regex=True).str.strip()


def _clean_numeric(col: pd.Series) -> pd.Series:
    col = (
        col.astype(str)
           .str.strip()
           .str.replace(".", "", regex=False)
           .str.replace(",", ".", regex=False)
           .replace({"": pd.NA, "..": pd.NA})
    )
    return pd.to_numeric(col, errors="coerce")


def clean_life_expectancy(
    df: pd.DataFrame, year: int | None = None
) -> pd.DataFrame:
    if year is not None:
        df = df[df["Periodo"] == year]

    df = df.copy()
    df["provincia"] = _clean_province(df["Provincias"])
    df["life_expectancy_65"] = _clean_numeric(df["Total"])
    df["Periodo"] = df["Periodo"].astype(int)

    return df[["provincia", "Periodo", "life_expectancy_65"]].dropna().drop_duplicates()


def clean_income(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    if year is not None:
        df = df[df["Periodo"] == year]

    df = df.copy()

    df["provincia"] = _clean_province(df["Provincias"])
    df["avg_gross_income"] = _clean_numeric(df["Total"])
    df["Periodo"] = df["Periodo"].astype(int)

    df = df.dropna(subset=["avg_gross_income"])

    df["avg_gross_income"] = df["avg_gross_income"].round().astype(int)

    return (
        df[["provincia", "Periodo", "avg_gross_income"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )