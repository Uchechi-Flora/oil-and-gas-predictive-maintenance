import pandas as pd
from datetime import datetime
from src.data_prep import load_data


def calculate_maintenance_intervals(maint_df):
    """
    For each asset, calculates the average number of days between
    its preventive maintenance events.
    """
    preventive = maint_df[maint_df["maintenance_type"] == "Preventive"].copy()
    preventive = preventive.sort_values(["asset_id", "date"])

    # Days between consecutive preventive events, per asset
    preventive["days_since_last"] = preventive.groupby("asset_id")["date"].diff().dt.days

    avg_interval_per_asset = preventive.groupby("asset_id")["days_since_last"].mean()

    # Fallback: overall average, for assets with too little history
    overall_avg_interval = preventive["days_since_last"].mean()

    return avg_interval_per_asset, overall_avg_interval


def apply_criticality_adjustment(recommended_date, criticality):
    """
    Nudges the recommended checkup date earlier for higher-criticality assets,
    as a safety margin - since the cost of missing a check is higher for them.
    """
    if criticality == "High":
        buffer_days = 14
    elif criticality == "Medium":
        buffer_days = 7
    else:  # Low
        buffer_days = 0

    return recommended_date - pd.Timedelta(days=buffer_days)


def recommend_next_checkup(asset_id, maint_df, asset_df, avg_interval_per_asset, overall_avg_interval, reference_date=None):
    """
    Calculates a recommended next checkup date for a given asset, adjusted
    for criticality, and flags whether that date is already overdue.
    """
    asset_maint = maint_df[maint_df["asset_id"] == asset_id].sort_values("date")

    if asset_maint.empty:
        return None, "No maintenance history available for this asset.", None

    last_maintenance_date = asset_maint["date"].max()

    if asset_id in avg_interval_per_asset.index and pd.notna(avg_interval_per_asset[asset_id]):
        interval = avg_interval_per_asset[asset_id]
        source = "based on this asset's own maintenance history"
    else:
        interval = overall_avg_interval
        source = "based on fleet-wide average (limited history for this asset)"

    recommended_date = last_maintenance_date + pd.Timedelta(days=interval)

    # --- Adjust for criticality ---
    criticality = asset_df.loc[asset_df["asset_id"] == asset_id, "criticality"].values[0]
    recommended_date = apply_criticality_adjustment(recommended_date, criticality)
    #source += f", adjusted for {criticality} criticality"

    # --- Overdue detection ---
    if reference_date is None:
        reference_date = pd.Timestamp(datetime.today().date())

    days_overdue = (reference_date - recommended_date).days

    if days_overdue > 0:
        status = f"OVERDUE by {days_overdue} days"
    else:
        status = f"On schedule ({abs(days_overdue)} days remaining)"

    return recommended_date, source, status


if __name__ == "__main__":
    data = load_data()
    avg_per_asset, overall_avg = calculate_maintenance_intervals(data["maintenance"])

    # Anchor "today" to the dataset's own timeline, not the real-world date,
    # since this is a fictional dataset that ends Dec 31, 2025.
    dataset_reference_date = data["sensor"]["timestamp"].max()

    print(f"Overall average interval between preventive checkups: {overall_avg:.0f} days")
    print(f"Reference date (end of dataset): {dataset_reference_date.date()}\n")

    for asset_id in sorted(data["asset"]["asset_id"].unique()):
        rec_date, source, status = recommend_next_checkup(
            asset_id, data["maintenance"], data["asset"], avg_per_asset, overall_avg,
            reference_date=dataset_reference_date
        )
        print(f"{asset_id}: {rec_date.date() if rec_date else 'N/A'} — {status} ({source})")