from pathlib import Path

from src.dataset_loader import load_income, load_life_expectancy
from src.cleaner import clean_income, clean_life_expectancy
from src.analysis import merge_and_correlate
from src.plotter import scatter_with_regression


def main() -> None:
    life_raw = load_life_expectancy()
    income_raw = load_income()

    life = clean_life_expectancy(life_raw, year=2022)
    income = clean_income(income_raw, year=2022)
    print("Life rows 2022:", len(life))
    print("Income rows 2022:", len(income))

    merged, r, p = merge_and_correlate(life, income)

    plot_path = scatter_with_regression(merged)
    print(f"\nPearson r = {r:.2f}   (p = {p:.2e})")
    print(f"Scatter plot → {Path(plot_path).resolve()}\n")


if __name__ == "__main__":
    main()
