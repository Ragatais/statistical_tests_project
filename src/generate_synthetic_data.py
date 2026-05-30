import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot

from config import DATA_DIR, FIGURES_DIR, RANDOM_STATE


def generate_independent_groups():
    rng = np.random.default_rng(RANDOM_STATE)

    group_a = rng.normal(loc=100, scale=10, size=100)
    group_b = rng.normal(loc=108, scale=10, size=100)

    df = pd.DataFrame({
        "subject_id": range(1, 201),
        "group": ["A"] * 100 + ["B"] * 100,
        "measurement": np.concatenate([group_a, group_b])
    })

    return df


def generate_dependent_groups():
    rng = np.random.default_rng(RANDOM_STATE)

    before = rng.normal(loc=100, scale=10, size=100)
    improvement = rng.normal(loc=5, scale=4, size=100)
    after = before + improvement

    df = pd.DataFrame({
        "subject_id": range(1, 101),
        "before": before,
        "after": after,
        "difference": after - before
    })

    return df


def plot_histogram(series, title, xlabel, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(series, kde=True, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Liczba obserwacji")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_qq(series, title, output_path):
    plt.figure(figsize=(6, 6))
    probplot(series, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_distributions(independent_df, dependent_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    group_a = independent_df[independent_df["group"] == "A"]["measurement"]
    group_b = independent_df[independent_df["group"] == "B"]["measurement"]

    plot_histogram(
        group_a,
        "Histogram rozkładu pomiarów — grupa A",
        "Wartość pomiaru",
        FIGURES_DIR / "histogram_independent_group_A.png"
    )

    plot_histogram(
        group_b,
        "Histogram rozkładu pomiarów — grupa B",
        "Wartość pomiaru",
        FIGURES_DIR / "histogram_independent_group_B.png"
    )

    plot_histogram(
        dependent_df["before"],
        "Histogram rozkładu pomiarów przed",
        "Wartość pomiaru",
        FIGURES_DIR / "histogram_dependent_before.png"
    )

    plot_histogram(
        dependent_df["after"],
        "Histogram rozkładu pomiarów po",
        "Wartość pomiaru",
        FIGURES_DIR / "histogram_dependent_after.png"
    )

    plot_histogram(
        dependent_df["difference"],
        "Histogram rozkładu różnic pomiarów",
        "Różnica: after - before",
        FIGURES_DIR / "histogram_dependent_difference.png"
    )

    plot_qq(
        group_a,
        "QQ-plot — grupa A",
        FIGURES_DIR / "qqplot_independent_group_A.png"
    )

    plot_qq(
        group_b,
        "QQ-plot — grupa B",
        FIGURES_DIR / "qqplot_independent_group_B.png"
    )

    plot_qq(
        dependent_df["difference"],
        "QQ-plot — różnice pomiarów",
        FIGURES_DIR / "qqplot_dependent_difference.png"
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    independent_df = generate_independent_groups()
    dependent_df = generate_dependent_groups()

    independent_df.to_csv(
        DATA_DIR / "synthetic_independent_groups.csv",
        index=False,
        encoding="utf-8"
    )

    dependent_df.to_csv(
        DATA_DIR / "synthetic_dependent_groups.csv",
        index=False,
        encoding="utf-8"
    )

    plot_distributions(independent_df, dependent_df)

    print("\nWygenerowano dane syntetyczne o rozkładzie normalnym.")
    print("\nPliki danych:")
    print(DATA_DIR / "synthetic_independent_groups.csv")
    print(DATA_DIR / "synthetic_dependent_groups.csv")

    print("\nWykresy rozkładów zapisano w:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()