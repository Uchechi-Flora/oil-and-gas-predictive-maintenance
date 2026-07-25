import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from logic.split_data import time_based_split

DATA_PATH = "data/model_ready_data.csv"
MODEL_OUTPUT_PATH = "models/failure_risk_model.pkl"

# Columns the model will actually learn from
FEATURE_COLS = [
    "vibration_mm_s", "temperature_c", "pressure_psi", "flow_rate_m3h", "rpm",
    "vibration_mm_s_roll7", "temperature_c_roll7", "pressure_psi_roll7",
    "flow_rate_m3h_roll7", "rpm_roll7",
    "vibration_mm_s_rate_of_change", "temperature_c_rate_of_change",
    "pressure_psi_rate_of_change", "flow_rate_m3h_rate_of_change", "rpm_rate_of_change",
]
TARGET_COL = "will_fail_soon"

# The threshold we've decided to use going forward, based on the
# cost trade-off (missed failures are far more expensive than false alarms)
FINAL_THRESHOLD = 0.40


def train_model():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = time_based_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # class_weight="balanced" tells the model to pay extra attention to
    # the rare "will fail" examples, since they're heavily outnumbered
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Evaluate on the test set (data the model never trained on) ---
    y_pred_default = model.predict(X_test)  # uses the standard 50% cutoff
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of "will fail"

    print("\n=== Classification Report (Default Threshold = 0.50) ===")
    print(classification_report(y_test, y_pred_default, target_names=["No Failure Soon", "Failure Soon"]))

    print("=== Confusion Matrix (Default Threshold = 0.50) ===")
    print(confusion_matrix(y_test, y_pred_default))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

    # --- Apply our chosen threshold (30%) - this is the "real" prediction going forward ---
    y_pred_final = (y_pred_proba >= FINAL_THRESHOLD).astype(int)

    print(f"\n=== Classification Report (Final Threshold = {FINAL_THRESHOLD}) ===")
    print(classification_report(y_test, y_pred_final, target_names=["No Failure Soon", "Failure Soon"]))

    print(f"=== Confusion Matrix (Final Threshold = {FINAL_THRESHOLD}) ===")
    print(confusion_matrix(y_test, y_pred_final))

    # --- Save the trained model ---
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

    return model


if __name__ == "__main__":
    train_model()
