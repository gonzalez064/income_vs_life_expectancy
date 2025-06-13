from pathlib import Path

from src.dataset_loader import load_income, load_life_expectancy
from src.cleaner import clean_income, clean_life_expectancy
from src.analysis import merge_datasets, correlations_by_year
from src.plotter import scatter_with_regression


YEAR_MIN = 2015
YEAR_MAX = 2022


def main() -> None:
    life_raw = load_life_expectancy()
    income_raw = load_income()

    life = clean_life_expectancy(life_raw, year=None)
    income = clean_income(income_raw, year=None)

    merged = merge_datasets(life, income)

    merged = merged[
        (merged["Periodo"] >= YEAR_MIN) & (merged["Periodo"] <= YEAR_MAX)
    ]
    if merged.empty:
        raise ValueError(
            f"No se encontraron datos entre {YEAR_MIN} y {YEAR_MAX}"
        )

    year_summary, r_all, p_all = correlations_by_year(merged)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    for _, row in year_summary.iterrows():
        year = int(row["year"])
        df_year = merged[merged["Periodo"] == year]
        scatter_with_regression(
            df_year,
            plots_dir / f"scatter_{year}.png",
            f"Income vs Life Expectancy · {year}",
        )

    scatter_with_regression(
        merged,
        plots_dir / "scatter_all_years.png",
        "Income vs Life Expectancy · 2015-2022",
    )

    print("\nPearson correlation per year (2015-2022):")
    print(
        year_summary.to_string(
            index=False,
            formatters={"r": "{:.2f}".format, "p": "{:.1e}".format},
        )
    )

    print("\nGLOBAL correlation 2015-2022")
    print(
        f"   r = {r_all:.2f}   p-value = {p_all:.1e}   "
        f"N = {len(merged)} observations"
    )
    print(f"\nScatter plots saved in {plots_dir.resolve()}")

if __name__ == "__main__":
    main()
