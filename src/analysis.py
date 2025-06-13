from typing import Tuple
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def merge_datasets(
    life_df: pd.DataFrame, income_df: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(life_df, income_df, on=["provincia", "Periodo"], how="inner")


def correlations_by_year(merged: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    rows = []
    for year, g in merged.groupby("Periodo"):
        if len(g) >= 2:
            r, p = pearsonr(
                np.asarray(g["avg_gross_income"], dtype=float),
                np.asarray(g["life_expectancy_65"], dtype=float),
            )
            rows.append({"year": int(year), "n_provinces": len(g), "r": r, "p": p})

    r_all, p_all = pearsonr(
        merged["avg_gross_income"].to_numpy(dtype=np.float64),
        merged["life_expectancy_65"].to_numpy(dtype=np.float64),
    )
    summary = pd.DataFrame(rows).sort_values("year")
    return summary, r_all, p_all