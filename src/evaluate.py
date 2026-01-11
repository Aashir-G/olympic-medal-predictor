import argparse
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

from src.config import PROCESSED_CSV, MODEL_PATH


TARGET = "medals"


def main(data_path: Path, model_path: Path, top_k: int) -> None:
    artifact = load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    test_year = artifact["test_year"]

    df = pd.read_csv(data_path)
    test_df = df[df["Year"] == test_year].copy()

    if test_df.empty:
        raise ValueError(f"No rows found for test_year={test_year} in {data_path}")

    X_test = test_df[features]
    y_test = test_df[TARGET]

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    out = test_df[["Year", "NOC"] + features + [TARGET]].copy()
    out["predicted_medals"] = preds
    out["abs_error"] = (out[TARGET] - out["predicted_medals"]).abs()

    print(f"Model evaluation for test_year={test_year}")
    print(f"MAE: {mae:.3f}")
    print("\nTop errors:")
    print(out.sort_values("abs_error", ascending=False).head(top_k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved medal prediction model.")
    parser.add_argument("--data", type=str, default=str(PROCESSED_CSV), help="Path to processed clean.csv")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to saved model joblib")
    parser.add_argument("--top_k", type=int, default=10, help="Show top K biggest errors")
    args = parser.parse_args()

    main(Path(args.data), Path(args.model), args.top_k)
