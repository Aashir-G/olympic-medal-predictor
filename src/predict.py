import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from src.config import MODEL_PATH


def main(model_path: Path, country: str, athletes: int, prev_medals: int) -> None:
    artifact = load(model_path)
    model = artifact["model"]
    features = artifact["features"]

    # Build a single-row dataframe in the correct feature order
    row = pd.DataFrame([[athletes, prev_medals]], columns=features)

    pred = float(model.predict(row)[0])

    # Clamp to 0+ since negative medals make no sense
    pred = max(0.0, pred)

    print("Prediction")
    print(f"Country: {country}")
    print(f"Athletes: {athletes}")
    print(f"Previous medals: {prev_medals}")
    print(f"Predicted medals: {pred:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool to predict Olympic medals for a country.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to saved model joblib")
    parser.add_argument("--country", type=str, default="Unknown", help="Country name (for display only)")
    parser.add_argument("--athletes", type=int, required=True, help="Number of athletes sent")
    parser.add_argument("--prev_medals", type=int, required=True, help="Medals from previous Olympics")
    args = parser.parse_args()

    main(Path(args.model), args.country, args.athletes, args.prev_medals)
