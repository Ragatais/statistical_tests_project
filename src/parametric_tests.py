import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, ttest_ind, ttest_rel

from config import DATA_DIR, FIGURES_DIR, TABLES_DIR


ALPHA = 0.05


def interpret_p_value(p_value, alpha=ALPHA):
    if p_value < alpha:
        return "odrzucamy H0"
    return "brak podstaw do odrzucenia H0"


def test_normality(series, variable_name, group_name):
    statistic, p_value = shapiro(series)

    return {
        "test": "Shapiro-Wilk",
        "variable": variable_name,
        "group": group_name,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": ALPHA,
        "interpretation": interpret_p_value(p_value),
        "normal_distribution": p_value >= ALPHA,
    }


def plot_independent_groups(independent_df):
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=independent_df,
        x="group",
        y="measurement",
        showmeans=True,
        showfliers=False
    )

    sns.stripplot(
        data=independent_df,
        x="group",
        y="measurement",
        alpha=0.4
    )

    plt.title("Porównanie pomiarów w dwóch grupach niezależnych")
    plt.xlabel("Grupa")
    plt.ylabel("Wartość pomiaru")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "ttest_independent_groups_boxplot.png", dpi=300)
    plt.close()


def plot_dependent_groups(dependent_df):
    long_df = dependent_df.melt(
        id_vars="subject_id",
        value_vars=["before", "after"],
        var_name="measurement_time",
        value_name="measurement"
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=long_df,
        x="measurement_time",
        y="measurement",
        showmeans=True,
        showfliers = False
    )

    sns.stripplot(
        data=long_df,
        x="measurement_time",
        y="measurement",
        alpha=0.4
    )

    plt.title("Porównanie pomiarów przed i po dla danych zależnych")
    plt.xlabel("Moment pomiaru")
    plt.ylabel("Wartość pomiaru")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "ttest_dependent_groups_boxplot.png", dpi=300)
    plt.close()


def plot_dependent_differences(dependent_df):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        dependent_df["difference"],
        kde=True,
        bins=20
    )

    plt.axvline(
        dependent_df["difference"].mean(),
        linestyle="--",
        label="Średnia różnica"
    )

    plt.title("Rozkład różnic pomiarów dla danych zależnych")
    plt.xlabel("Różnica: after - before")
    plt.ylabel("Liczba obserwacji")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "ttest_dependent_differences_histogram.png", dpi=300)
    plt.close()


def independent_t_test():
    path = DATA_DIR / "synthetic_independent_groups.csv"
    df = pd.read_csv(path)

    group_a = df[df["group"] == "A"]["measurement"]
    group_b = df[df["group"] == "B"]["measurement"]

    normality_results = [
        test_normality(group_a, "measurement", "A"),
        test_normality(group_b, "measurement", "B"),
    ]

    levene_statistic, levene_p_value = levene(group_a, group_b)

    equal_variance = levene_p_value >= ALPHA

    t_statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=equal_variance
    )

    result = {
        "test": "t-Student dla prób niezależnych",
        "data_type": "niezależne",
        "group_a_n": len(group_a),
        "group_b_n": len(group_b),
        "group_a_mean": group_a.mean(),
        "group_b_mean": group_b.mean(),
        "group_a_std": group_a.std(ddof=1),
        "group_b_std": group_b.std(ddof=1),
        "levene_statistic": levene_statistic,
        "levene_p_value": levene_p_value,
        "equal_variance_assumption": equal_variance,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "alpha": ALPHA,
        "interpretation": interpret_p_value(p_value),
    }

    plot_independent_groups(df)

    return pd.DataFrame([result]), pd.DataFrame(normality_results)


def dependent_t_test():
    path = DATA_DIR / "synthetic_dependent_groups.csv"
    df = pd.read_csv(path)

    before = df["before"]
    after = df["after"]
    difference = df["difference"]

    normality_results = [
        test_normality(before, "before", "dependent"),
        test_normality(after, "after", "dependent"),
        test_normality(difference, "difference", "dependent"),
    ]

    t_statistic, p_value = ttest_rel(before, after)

    result = {
        "test": "t-Student dla prób zależnych",
        "data_type": "zależne",
        "n_pairs": len(df),
        "before_mean": before.mean(),
        "after_mean": after.mean(),
        "difference_mean": difference.mean(),
        "difference_std": difference.std(ddof=1),
        "t_statistic": t_statistic,
        "p_value": p_value,
        "alpha": ALPHA,
        "interpretation": interpret_p_value(p_value),
    }

    plot_dependent_groups(df)
    plot_dependent_differences(df)

    return pd.DataFrame([result]), pd.DataFrame(normality_results)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    independent_result, independent_normality = independent_t_test()
    dependent_result, dependent_normality = dependent_t_test()

    all_results = pd.concat(
        [independent_result, dependent_result],
        ignore_index=True
    )

    all_normality = pd.concat(
        [independent_normality, dependent_normality],
        ignore_index=True
    )

    all_results.to_csv(
        TABLES_DIR / "parametric_tests_results.csv",
        index=False,
        encoding="utf-8"
    )

    all_normality.to_csv(
        TABLES_DIR / "parametric_tests_normality_shapiro.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nWyniki testów parametrycznych")
    print(all_results)

    print("\nWyniki testów normalności Shapiro-Wilka")
    print(all_normality)

    print("\nWyniki zapisano w:")
    print(TABLES_DIR)

    print("\nWykresy zapisano w:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()