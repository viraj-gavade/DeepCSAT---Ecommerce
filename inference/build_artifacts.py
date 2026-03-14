from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.preprocessing import NotebookPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing artifacts for CSAT inference.")
    parser.add_argument(
        "--data-path",
        default="eCommerce_Customer_support_data.csv",
        help="Path to training CSV used to fit preprocessing artifacts.",
    )
    parser.add_argument(
        "--model-path",
        default="csat_model.h5",
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(data_path)

    preprocessor = NotebookPreprocessor().fit(df)
    transformed = preprocessor.transform(df)

    model = tf.keras.models.load_model(model_path)
    expected_features = int(model.input_shape[-1])
    actual_features = int(transformed.shape[1])

    if expected_features != actual_features:
        raise ValueError(
            "Feature mismatch between saved model and preprocessing output: "
            f"model expects {expected_features}, but preprocessor produced {actual_features}."
        )

    preprocessor_path = output_dir / "preprocessor.joblib"
    schema_path = output_dir / "feature_schema.json"

    joblib.dump(preprocessor, preprocessor_path)

    audit = preprocessor.feature_audit()
    schema = {
        "model_input_features": expected_features,
        "rows_used_for_fit": int(len(df)),
        "final_feature_order": preprocessor.final_feature_order_,
        "feature_audit": asdict(audit),
    }

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Saved preprocessor: {preprocessor_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Final feature count: {actual_features}")


if __name__ == "__main__":
    main()
