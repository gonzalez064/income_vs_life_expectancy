import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import linregress


def scatter_with_regression(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str = "Average gross income (EUR)",
    ylabel: str = "Life expectancy at 65 (years)",
) -> None:
    """Crea y guarda un scatter plot con recta de regresión."""
    slope, intercept, *_ = linregress(df["avg_gross_income"], df["life_expectancy_65"])
    x_vals = df["avg_gross_income"].sort_values()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["avg_gross_income"], df["life_expectancy_65"], alpha=0.7)
    plt.plot(x_vals, intercept + slope * x_vals, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
