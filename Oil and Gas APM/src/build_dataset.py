import pandas as pd
from src.data_prep import load_data
from src.features import add_rolling_features
from logic.labelling import add_failure_label

OUTPUT_PATH = "data/model_ready_data.csv"


def build_dataset():
    # Step 1: Load all four raw tables
    data = load_data()

    # Step 2: Add rolling averages + rate of change to sensor data
    sensor_with_features = add_rolling_features(data["sensor"])

    # Step 3: Add the failure label (will this asset fail within 21 days?)
    sensor_labeled = add_failure_label(sensor_with_features, data["failure"])

    # Step 4: Bring in static asset info (criticality, asset_type, etc.)
    merged = sensor_labeled.merge(data["asset"], on="asset_id", how="left")

    # Step 5: Drop the early rows that don't have a full 7-day history yet
    feature_cols = [c for c in merged.columns if "roll7" in c or "rate_of_change" in c]
    before_count = len(merged)
    merged = merged.dropna(subset=feature_cols)
    after_count = len(merged)

    print(f"Dropped {before_count - after_count} rows with incomplete rolling history.")
    print(f"Final dataset shape: {merged.shape}")

    # Step 6: Save to CSV so we don't have to rerun everything each time
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved model-ready dataset to {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    build_dataset()