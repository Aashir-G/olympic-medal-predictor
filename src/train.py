import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_CSV, MODELS_DIR, MODEL_PATH


FEATURES = ["athletes", "prev_medals"]
TARGET = "medals"


def time_split(df: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["Year"] < test_year].copy()
    test_df = df[df["Year"] == test_year].copy()

    if train_df.empty:
        raise ValueError("Train split is empty. Choose a smaller test_year.")
    if test_df.empty:
        raise ValueError("Test split is empty. Choose a test_year that exists in the dataset.")

    return train_df, test_df


def main(data_path: Path, model_path: Path, test_year: int, alpha: float) -> None:
    df = pd.read_csv(data_path)

    for col in ["Year", "NOC"] + FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    train_df, test_df = time_split(df, test_year=test_year)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURES,
        "test_year": test_year,
        "alpha": alpha,
        "mae": float(mae),
    }

    dump(artifact, model_path)

    print(f"Saved model -> {model_path}")
    print(f"Test year: {test_year} | MAE: {mae:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train medal prediction model.")
    parser.add_argument("--data", type=str, default=str(PROCESSED_CSV), help="Path to processed clean.csv")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Output model path (joblib)")
    parser.add_argument("--test_year", type=int, default=2016, help="Holdout Olympics year for evaluation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    args = parser.parse_args()

    main(Path(args.data), Path(args.model), args.test_year, args.alpha)
