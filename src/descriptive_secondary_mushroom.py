import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_DIR, FIGURES_DIR, TABLES_DIR


DATASET_PATH = DATA_DIR / "secondary_data.csv"

CLASS_COL = "class"
CATEGORY_COL = "season"

FEATURES = [
    "cap-diameter",
    "stem-height",
]

SEASON_LABELS = {
    "s": "spring",
    "u": "summer",
    "a": "autumn",
    "w": "winter",
}

SEASON_ORDER = [
    "spring",
    "summer",
    "autumn",
    "winter",
]


def load_data():
    df = pd.read_csv(DATASET_PATH, sep=";")

    required_columns = [CLASS_COL, CATEGORY_COL] + FEATURES

    df = df.dropna(subset=required_columns).copy()

    df[CATEGORY_COL] = df[CATEGORY_COL].replace(SEASON_LABELS)

    df = df[df[CATEGORY_COL].isin(SEASON_ORDER)].reset_index(drop=True)

    return df


def coefficient_of_variation(values):
    mean = values.mean()

    if mean == 0 or np.isnan(mean):
        return np.nan

    return values.std(ddof=1) / mean * 100


def get_mode(values):
    rounded_values = values.round(2)
    mode_values = rounded_values.mode()

    if mode_values.empty:
        return np.nan

    return mode_values.iloc[0]


def descriptive_statistics_for_feature(df, feature):
    rows = []

    for season in SEASON_ORDER:
        values = df.loc[df[CATEGORY_COL] == season, feature].dropna()

        q1 = values.quantile(0.25)
        q2 = values.quantile(0.50)
        q3 = values.quantile(0.75)

        iqr = q3 - q1
        lower_outlier_threshold = q1 - 1.5 * iqr
        upper_outlier_threshold = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_outlier_threshold) |
            (values > upper_outlier_threshold)
            ]

        rows.append({
            "feature": feature,
            "category_variable": CATEGORY_COL,
            "category": season,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "mode": get_mode(values),
            "variance": values.var(ddof=1),
            "std": values.std(ddof=1),
            "skewness": values.skew(),
            "kurtosis": values.kurtosis(),
            "Q1": q1,
            "Q2_median": q2,
            "Q3": q3,
            "IQR": iqr,
            "lower_outlier_threshold": lower_outlier_threshold,
            "upper_outlier_threshold": upper_outlier_threshold,
            "n_outliers": len(outliers),
            "outliers_percent": len(outliers) / len(values) * 100,
            "D1": values.quantile(0.10),
            "D9": values.quantile(0.90),
            "C10": values.quantile(0.10),
            "C90": values.quantile(0.90),
            "coefficient_of_variation_percent": coefficient_of_variation(values),
            "min": values.min(),
            "max": values.max(),
        })

    return rows


def plot_distribution(df, feature):
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue=CATEGORY_COL,
        hue_order=SEASON_ORDER,
        kde=True,
        bins=40
    )

    plt.title(f"Histogram rozkładu cechy: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Liczba obserwacji")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"histogram_{feature.replace('-', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_single_boxplot(df, feature, season):
    values = df.loc[df[CATEGORY_COL] == season, feature].dropna()

    flierprops = {
        "marker": "o",
        "markersize": 2,
        "alpha": 0.25,
    }

    plt.figure(figsize=(5, 6))

    plt.boxplot(
        values,
        labels=[season],
        showmeans=True,
        flierprops=flierprops
    )

    plt.title(f"Wykres pudełkowy: {feature} — {season}")
    plt.ylabel(feature)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"boxplot_{feature.replace('-', '_')}_{season}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_grouped_boxplot(df, feature):

    flierprops = {
        "marker": "o",
        "markersize": 2,
        "alpha": 0.25,
    }

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x=CATEGORY_COL,
        y=feature,
        order=SEASON_ORDER,
        showmeans=True,
        flierprops=flierprops
    )

    plt.title(f"Porównanie rozkładu cechy: {feature} według pory roku")
    plt.xlabel("Pora roku")
    plt.ylabel(feature)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"grouped_boxplot_{feature.replace('-', '_')}_by_season.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    all_statistics = []

    for feature in FEATURES:
        all_statistics.extend(
            descriptive_statistics_for_feature(df, feature)
        )

        plot_distribution(df, feature)
        plot_grouped_boxplot(df, feature)

        for season in SEASON_ORDER:
            plot_single_boxplot(df, feature, season)

    statistics_df = pd.DataFrame(all_statistics)

    output_path = TABLES_DIR / "secondary_mushroom_descriptive_statistics_by_season.csv"
    statistics_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nStatystyka opisowa dla zbioru Secondary Mushroom")
    print(statistics_df)

    print("\nWyniki zapisano w:")
    print(TABLES_DIR)

    print("\nWykresy zapisano w:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()