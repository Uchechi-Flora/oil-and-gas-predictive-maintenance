import pandas as pd

DATA_PATH = "data/model_ready_data.csv"

# We'll use the last ~20% of the date range as our test set
TEST_SIZE_FRACTION = 0.2


def time_based_split(df, date_col="timestamp", test_fraction=TEST_SIZE_FRACTION):
    """
    Splits data chronologically: earliest dates go to training,
    most recent dates go to testing. This avoids data leakage between
    neighboring days.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Find the exact date that marks the 80% mark of our timeline
    cutoff_index = int(len(df) * (1 - test_fraction))
    cutoff_date = df.iloc[cutoff_index][date_col]

    train_df = df[df[date_col] < cutoff_date]
    test_df = df[df[date_col] >= cutoff_date]

    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Training rows: {len(train_df)}  ({train_df[date_col].min().date()} to {train_df[date_col].max().date()})")
    print(f"Testing rows:  {len(test_df)}  ({test_df[date_col].min().date()} to {test_df[date_col].max().date()})")
    print(f"\nFailures in training set: {train_df['will_fail_soon'].sum()}")
    print(f"Failures in testing set:  {test_df['will_fail_soon'].sum()}")

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = time_based_split(df)