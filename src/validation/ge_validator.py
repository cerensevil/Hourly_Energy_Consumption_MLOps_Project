import great_expectations as gx
import pandas as pd


def validate_dataframe(df: pd.DataFrame, state: str):
    """
    Basic + slightly advanced validation rules
    """

    gx_df = gx.from_pandas(df)

    # =========================
    # BASIC EXPECTATIONS
    # =========================
    gx_df.expect_column_values_to_not_be_null("Datetime")
    gx_df.expect_column_values_to_not_be_null("target")

    gx_df.expect_column_values_to_be_between(
        "target", min_value=0, mostly=0.99
    )

    gx_df.expect_column_values_to_be_unique("Datetime")

    gx_df.expect_column_values_to_be_increasing("Datetime")

    # =========================
    # EXTRA (time-series sanity)
    # =========================
    gx_df.expect_column_values_to_not_be_null("hour")

    gx_df.expect_column_values_to_be_in_set(
        "state", [state]
    )

    # =========================
    # VALIDATE
    # =========================
    results = gx_df.validate()

    if not results["success"]:
        raise ValueError(f"Validation failed for state={state}")

    return results