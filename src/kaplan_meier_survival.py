import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from config import DATA_DIR, FIGURES_DIR, TABLES_DIR


DATASET_PATH = DATA_DIR / "heart_failure_clinical_records_dataset.csv"

TIME_COL = "time"
EVENT_COL = "DEATH_EVENT"

GROUPING_COL = "ejection_fraction"
GROUP_COL = "ejection_fraction_group"

ALPHA = 0.05


def load_data():
    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        TIME_COL,
        EVENT_COL,
        GROUPING_COL,
    ]

    df = df.dropna(subset=required_columns).copy()

    return df


def create_groups(df):
    median_value = df[GROUPING_COL].median()

    df[GROUP_COL] = df[GROUPING_COL].apply(
        lambda value: "niska frakcja wyrzutowa"
        if value <= median_value
        else "wysoka frakcja wyrzutowa"
    )

    return df, median_value


def survival_table_by_group(df):
    rows = []

    for group_name, group_df in df.groupby(GROUP_COL):
        rows.append({
            "group": group_name,
            "n_patients": len(group_df),
            "n_events": int(group_df[EVENT_COL].sum()),
            "n_censored": int((group_df[EVENT_COL] == 0).sum()),
            "event_rate_percent": group_df[EVENT_COL].mean() * 100,
            "median_follow_up_time": group_df[TIME_COL].median(),
            "mean_follow_up_time": group_df[TIME_COL].mean(),
        })

    return pd.DataFrame(rows)


def plot_kaplan_meier_curves(df, median_value):
    plt.figure(figsize=(9, 6))

    kmf = KaplanMeierFitter()

    for group_name, group_df in df.groupby(GROUP_COL):
        kmf.fit(
            durations=group_df[TIME_COL],
            event_observed=group_df[EVENT_COL],
            label=group_name
        )

        kmf.plot_survival_function(
            ci_show=True,
            show_censors=True,
            censor_styles={
                "marker": "+",
                "ms": 7,
            }
        )

    plt.title(
        f"Krzywe przeżycia Kaplana-Meiera\n"
        f"podział według mediany frakcji wyrzutowej = {median_value:.1f}"
    )
    plt.xlabel("Czas obserwacji [dni]")
    plt.ylabel("Prawdopodobieństwo przeżycia")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "kaplan_meier_ejection_fraction_groups.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_group_counts(df):
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=GROUP_COL,
        hue=EVENT_COL
    )

    plt.title("Liczność zdarzeń i obserwacji uciętych w grupach")
    plt.xlabel("Grupa")
    plt.ylabel("Liczba pacjentów")
    plt.legend(
        title="DEATH_EVENT",
        labels=["obserwacja ucięta", "zdarzenie"]
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = FIGURES_DIR / "kaplan_meier_events_censored_counts.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_logrank_test(df):
    groups = list(df[GROUP_COL].unique())

    if len(groups) != 2:
        raise ValueError("Test log-rank w tym skrypcie wymaga dokładnie dwóch grup.")

    group_1 = df[df[GROUP_COL] == groups[0]]
    group_2 = df[df[GROUP_COL] == groups[1]]

    result = logrank_test(
        durations_A=group_1[TIME_COL],
        durations_B=group_2[TIME_COL],
        event_observed_A=group_1[EVENT_COL],
        event_observed_B=group_2[EVENT_COL],
    )

    if result.p_value < ALPHA:
        interpretation = "odrzucamy H0 - krzywe przeżycia różnią się istotnie statystycznie"
    else:
        interpretation = "brak podstaw do odrzucenia H0 - nie wykazano istotnej różnicy między krzywymi"

    return pd.DataFrame([{
        "test": "log-rank",
        "grouping_variable": GROUPING_COL,
        "group_1": groups[0],
        "group_2": groups[1],
        "test_statistic": result.test_statistic,
        "p_value": result.p_value,
        "alpha": ALPHA,
        "interpretation": interpretation,
    }])


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df, median_value = create_groups(df)

    grouped_summary = survival_table_by_group(df)
    logrank_results = run_logrank_test(df)

    grouped_summary.to_csv(
        TABLES_DIR / "kaplan_meier_group_summary.csv",
        index=False,
        encoding="utf-8"
    )

    logrank_results.to_csv(
        TABLES_DIR / "kaplan_meier_logrank_test.csv",
        index=False,
        encoding="utf-8"
    )

    df.to_csv(
        TABLES_DIR / "kaplan_meier_dataset_with_groups.csv",
        index=False,
        encoding="utf-8"
    )

    plot_kaplan_meier_curves(df, median_value)
    plot_group_counts(df)

    print("\nAnaliza przeżycia Kaplana-Meiera")
    print("\nPodsumowanie grup")
    print(grouped_summary)

    print("\nWynik testu log-rank")
    print(logrank_results)

    print("\nWyniki zapisano w:")
    print(TABLES_DIR)

    print("\nWykresy zapisano w:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()