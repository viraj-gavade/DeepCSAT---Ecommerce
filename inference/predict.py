from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch CSAT inference.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output-csv",
        default="predictions.csv",
        help="Path to save prediction CSV.",
    )
    parser.add_argument(
        "--model-path",
        default="csat_model.h5",
        help="Path to saved model file.",
    )
    parser.add_argument(
        "--preprocessor-path",
        default="artifacts/preprocessor.joblib",
        help="Path to fitted preprocessing artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    model_path = Path(args.model_path)
    preprocessor_path = Path(args.preprocessor_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    df = pd.read_csv(input_path)

    preprocessor = joblib.load(preprocessor_path)
    model = tf.keras.models.load_model(model_path)

    X = preprocessor.transform(df)
    probs = model.predict(X, verbose=0)

    predicted_class_index = np.argmax(probs, axis=1)
    predicted_csat_score = predicted_class_index + 1

    output_df = df.copy()
    output_df["predicted_csat_score"] = predicted_csat_score

    for class_idx in range(probs.shape[1]):
        score = class_idx + 1
        output_df[f"prob_csat_{score}"] = probs[:, class_idx]

    output_df.to_csv(output_path, index=False)

    print(f"Predictions generated for {len(output_df)} rows.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
