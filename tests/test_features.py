from __future__ import annotations

import numpy as np
import polars as pl


REQUIRED_COLS = [
    "Datetime",
    "hour",
    "dayofweek",
    "month",
    "year",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "target_lag_1",
    "target_lag_24",
    "target_roll_mean_24",
    "target_roll_std_24",
]


def get_target_col(df: pl.DataFrame) -> str:
    """
    Pipeline mantığına göre hedef: Datetime dışındaki ana ölçüm sütunu.
    REQUIRED_COLS + Datetime dışında kalan ilk sütunu hedef kabul eder.
    """
    candidates = [
        c for c in df.columns
        if c not in REQUIRED_COLS and c != "Datetime"
    ]

    assert len(candidates) >= 1, (
        "Target column not found. Expected at least 1 non-feature column "
        "besides Datetime and engineered columns."
    )

    return candidates[0]


# =====================================================
# A) Multi-file contract testleri
# =====================================================

def test_all_files_have_required_columns(df: pl.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_datetime_type_and_not_null(df: pl.DataFrame):
    assert "Datetime" in df.columns

    # Datetime timezone'lu olabilir; temporal tip kontrolü daha güvenli
    assert df["Datetime"].dtype.is_temporal(), (
        f"Datetime dtype is {df['Datetime'].dtype}, "
        "expected a temporal/datetime type"
    )

    assert df["Datetime"].null_count() == 0


def test_datetime_sorted_and_unique(df: pl.DataFrame):
    assert df["Datetime"].is_sorted(), (
        "Datetime must be sorted ascending"
    )

    n_unique = df.select(pl.col("Datetime").n_unique()).item()

    assert n_unique == df.height, (
        f"Datetime has duplicates: "
        f"n_unique={n_unique}, rows={df.height}"
    )


def test_no_nulls_in_engineered_columns(df: pl.DataFrame):
    for c in REQUIRED_COLS:
        assert df[c].null_count() == 0, f"Column {c} has nulls"


def test_feature_ranges(df: pl.DataFrame):
    # hour: 0-23
    assert 0 <= df["hour"].min() <= 23
    assert 0 <= df["hour"].max() <= 23

    # month: 1-12
    assert 1 <= df["month"].min() <= 12
    assert 1 <= df["month"].max() <= 12

    # dayofweek güvenli aralık
    assert df["dayofweek"].min() >= 0
    assert df["dayofweek"].max() <= 7

    # is_weekend: 0/1
    vals = set(df["is_weekend"].unique().to_list())
    assert vals.issubset({0, 1}), (
        f"is_weekend unexpected values: {vals}"
    )

    # sin/cos range
    assert df["sin_hour"].abs().max() <= 1.0001
    assert df["cos_hour"].abs().max() <= 1.0001

    # rolling std negatif olamaz
    assert df["target_roll_std_24"].min() >= 0, (
        "Rolling std must be >= 0"
    )


def test_target_is_numeric(df: pl.DataFrame):
    target = get_target_col(df)
    assert df[target].dtype.is_numeric(), (
        f"Target column {target} dtype={df[target].dtype} "
        "is not numeric"
    )


# =====================================================
# B) Baseline doğruluk testleri
# =====================================================

def test_baseline_weekend_and_dayofweek_consistent(
    baseline_df: pl.DataFrame,
):
    recomputed_day = baseline_df.select(
        pl.col("Datetime").dt.weekday().alias("dw")
    )["dw"]

    assert (recomputed_day == baseline_df["dayofweek"]).all(), (
        "dayofweek column is not consistent with Datetime"
    )

    recomputed_weekend = (recomputed_day >= 6).cast(pl.Int8)

    assert (recomputed_weekend == baseline_df["is_weekend"]).all(), (
        "is_weekend is not consistent with dayofweek"
    )


def test_baseline_sin_cos_correct(baseline_df: pl.DataFrame):
    hour = baseline_df["hour"].to_numpy()

    exp_sin = np.sin(2 * np.pi * hour / 24)
    exp_cos = np.cos(2 * np.pi * hour / 24)

    got_sin = baseline_df["sin_hour"].to_numpy()
    got_cos = baseline_df["cos_hour"].to_numpy()

    assert np.max(np.abs(got_sin - exp_sin)) < 1e-9, (
        "sin_hour formula mismatch"
    )

    assert np.max(np.abs(got_cos - exp_cos)) < 1e-9, (
        "cos_hour formula mismatch"
    )


def test_baseline_lag_1_correct(baseline_df: pl.DataFrame):
    target = get_target_col(baseline_df)

    df_small = baseline_df.select(
        [target, "target_lag_1"]
    ).head(10000)

    expected = df_small[target].shift(1)
    diff = (
        df_small["target_lag_1"] - expected
    ).abs().drop_nulls().max()

    assert diff == 0, (
        f"target_lag_1 != {target}.shift(1). "
        f"max abs diff={diff}"
    )


def test_baseline_lag_24_correct(baseline_df: pl.DataFrame):
    target = get_target_col(baseline_df)

    df_small = baseline_df.select(
        [target, "target_lag_24"]
    ).head(10000)

    expected = df_small[target].shift(24)
    diff = (
        df_small["target_lag_24"] - expected
    ).abs().drop_nulls().max()

    assert diff == 0, (
        f"target_lag_24 != {target}.shift(24). "
        f"max abs diff={diff}"
    )


def test_baseline_rolling_mean_24_correct(
    baseline_df: pl.DataFrame,
):
    target = get_target_col(baseline_df)

    df_small = baseline_df.select(
        [target, "target_roll_mean_24"]
    ).head(10000)

    expected = df_small[target].shift(1).rolling_mean(
        window_size=24
    )

    diff = (
        df_small["target_roll_mean_24"] - expected
    ).abs().drop_nulls().max()

    assert diff == 0, (
        f"target_roll_mean_24 incorrect. "
        f"max abs diff={diff}"
    )


def test_baseline_rolling_std_24_correct(
    baseline_df: pl.DataFrame,
):
    target = get_target_col(baseline_df)

    df_small = baseline_df.select(
        [target, "target_roll_std_24"]
    ).head(10000)

    expected = df_small[target].shift(1).rolling_std(
        window_size=24
    )

    diff = (
        df_small["target_roll_std_24"] - expected
    ).abs().drop_nulls().max()

    assert diff == 0, (
        f"target_roll_std_24 incorrect. "
        f"max abs diff={diff}"
    )
