import pandas as pd
import numpy as np
from src.data_prep import load_data

PREDICTION_WINDOW_DAYS = 21  # "will this asset fail within the next 21 days?"

def add_rolling_features(sensor_df):
    """
    Adds 7-day rolling average and rate-of-change columns for each sensor,
    computed SEPARATELY per asset (so one asset's history never bleeds into another's).
    """
    sensor_df = sensor_df.sort_values(["asset_id", "timestamp"]).copy()

    sensor_cols = ["vibration_mm_s", "temperature_c", "pressure_psi", "flow_rate_m3h", "rpm"]

    for col in sensor_cols:
        # 7-day rolling average, grouped by asset
        sensor_df[f"{col}_roll7"] = (
            sensor_df.groupby("asset_id")[col]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        )

        # Rate of change: today's rolling average minus the rolling average 7 days ago
        sensor_df[f"{col}_rate_of_change"] = (
            sensor_df.groupby("asset_id")[f"{col}_roll7"]
            .transform(lambda x: x.diff(periods=7))
        )

    return sensor_df


if __name__ == "__main__":
    data = load_data()
    sensor_with_features = add_rolling_features(data["sensor"])
    print(sensor_with_features[["asset_id", "timestamp", "vibration_mm_s",
                                  "vibration_mm_s_roll7", "vibration_mm_s_rate_of_change"]].head(15))