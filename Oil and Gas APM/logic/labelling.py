import pandas as pd
from src.data_prep import load_data
from src.features import add_rolling_features, PREDICTION_WINDOW_DAYS


def add_failure_label(sensor_df, fail_df, window_days=PREDICTION_WINDOW_DAYS):
    """
    For every sensor reading, checks whether that asset failed within
    the next `window_days` days. Adds a new column: will_fail_soon (1 or 0).
    """
    sensor_df = sensor_df.copy()
    sensor_df["will_fail_soon"] = 0  # default: assume no failure coming

    # Go asset by asset
    for asset_id in sensor_df["asset_id"].unique():
        # All failure dates for this specific asset
        asset_failures = fail_df.loc[fail_df["asset_id"] == asset_id, "failure_date"]

        if asset_failures.empty:
            continue  # this asset never failed in our data, skip it

        # All sensor rows for this specific asset
        asset_mask = sensor_df["asset_id"] == asset_id
        asset_rows = sensor_df.loc[asset_mask, "timestamp"]

        # For each failure date, find sensor rows within the window BEFORE it
        for failure_date in asset_failures:
            window_start = failure_date - pd.Timedelta(days=window_days)
            in_window = (asset_rows >= window_start) & (asset_rows <= failure_date)

            # Mark those specific rows as "1" (failure coming soon)
            rows_to_flag = asset_rows[in_window].index
            sensor_df.loc[rows_to_flag, "will_fail_soon"] = 1

    return sensor_df


if __name__ == "__main__":
    data = load_data()
    sensor_with_features = add_rolling_features(data["sensor"])
    sensor_labeled = add_failure_label(sensor_with_features, data["failure"])

    print("Total rows:", len(sensor_labeled))
    print("Rows labeled 'will fail soon' (1):", sensor_labeled["will_fail_soon"].sum())
    print("Rows labeled 'no failure soon' (0):", (sensor_labeled["will_fail_soon"] == 0).sum())
    print("\nSample of flagged rows:")
    print(sensor_labeled[sensor_labeled["will_fail_soon"] == 1][["asset_id", "timestamp", "will_fail_soon"]].head(10))