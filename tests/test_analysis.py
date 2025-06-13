from src.analysis import correlations_by_year, merge_datasets
import pandas as pd


def test_multi_year_summary():
    life = pd.DataFrame({
        "provincia": ["A", "A", "B", "B"],
        "Periodo":   [2021, 2022, 2021, 2022],
        "life_expectancy_65": [20, 21, 22, 23],
    })
    income = pd.DataFrame({
        "provincia": ["A", "A", "B", "B"],
        "Periodo":   [2021, 2022, 2021, 2022],
        "avg_gross_income": [10000, 11000, 20000, 21000],
    })
    summary, r_all, _ = correlations_by_year(merge_datasets(life, income))
    assert len(summary) == 2
    assert r_all > 0
