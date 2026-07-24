import pandas as pd

# --- File path ---
RAW_DATA_PATH = "data/asset_predictive_maintenance.xlsx"

def load_data():
    """
    Loads all four sheets from the Excel workbook into separate DataFrames.
    Returns a dictionary so we can access each table by name.
    """
    asset_df = pd.read_excel(RAW_DATA_PATH, sheet_name="Asset_Master")
    sensor_df = pd.read_excel(RAW_DATA_PATH, sheet_name="Sensor_Readings")
    maint_df = pd.read_excel(RAW_DATA_PATH, sheet_name="Maintenance_Log")
    fail_df = pd.read_excel(RAW_DATA_PATH, sheet_name="Failure_Events")

    # Make sure date columns are actually read as dates, not text
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    maint_df["date"] = pd.to_datetime(maint_df["date"])
    fail_df["failure_date"] = pd.to_datetime(fail_df["failure_date"])
    asset_df["install_date"] = pd.to_datetime(asset_df["install_date"])

    return {
        "asset": asset_df,
        "sensor": sensor_df,
        "maintenance": maint_df,
        "failure": fail_df,
    }

if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (not when another file imports functions from it)
    data = load_data()
    for name, df in data.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Shape: {df.shape}")
        print(df.head(3))