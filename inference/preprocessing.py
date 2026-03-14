from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif


@dataclass
class FeatureAudit:
    final_feature_order: List[str]
    high_corr_dropped: List[str]
    variance_selected: List[str]
    corr_selected: List[str]
    kbest_selected: List[str]


class NotebookPreprocessor:
    """Replicates the final preprocessing path used in the notebook.

    The implementation intentionally mirrors the notebook flow so the saved ANN
    model receives the same 15-feature input contract at inference time.
    """

    target_col = "CSAT Score"

    base_drop_cols = ["Unique id", "Order_id"]

    date_columns = [
        "order_date_time",
        "Issue_reported at",
        "issue_responded",
        "Survey_response_Date",
    ]

    numeric_impute_cols = [
        "connected_handling_time",
        "Item_price",
        "response_time_minutes",
        "order_month",
        "order_day",
        "order_hour",
    ]

    date_mode_cols = ["order_date_time", "Issue_reported at", "issue_responded"]

    drop_after_engineering = ["order_month", "order_day", "order_hour"]

    high_card_drop_cols = [
        "Unique id",
        "Order_id",
        "order_date_time",
        "Issue_reported at",
        "issue_responded",
        "Survey_response_Date",
        "Sub-category",
        "Customer Remarks",
        "Customer_City",
        "Agent_name",
        "Supervisor",
        "Manager",
    ]

    base_safe_categorical = [
        "channel_name",
        "category",
        "Product_category",
        "Tenure Bucket",
        "Agent Shift",
    ]

    def __init__(self) -> None:
        self.fitted_ = False

        self.numeric_medians_: Dict[str, float] = {}
        self.date_modes_: Dict[str, pd.Timestamp] = {}

        self.first_iqr_bounds_: Dict[str, Tuple[float, float]] = {}
        self.second_iqr_bounds_: Dict[str, Tuple[float, float]] = {}

        self.safe_categorical_cols_: List[str] = []
        self.encoded_columns_: List[str] = []

        self.high_corr_features_: List[str] = []
        self.variance_filter_: VarianceThreshold | None = None
        self.selected_columns_var_: List[str] = []

        self.selected_features_corr_: List[str] = []
        self.select_k_best_: SelectKBest | None = None
        self.selected_columns_kbest_: List[str] = []

        self.final_feature_order_: List[str] = []

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_impute_cols:
            if col not in df.columns:
                df[col] = np.nan

        for col in self.date_mode_cols:
            if col not in df.columns:
                df[col] = pd.NaT

        return df

    def _compute_iqr_bounds(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}

        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bounds[col] = (float(lower), float(upper))

        return bounds

    def _apply_iqr_bounds(self, df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        for col, (lower, upper) in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df

    def _base_wrangling(self, raw_df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = raw_df.copy()

        df = df.drop(columns=[c for c in self.base_drop_cols if c in df.columns])

        for col in self.date_columns:
            if col in df.columns:
                # Keep default parsing behavior from the notebook (dayfirst=False).
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "order_date_time" in df.columns:
            df["order_month"] = df["order_date_time"].dt.month
            df["order_day"] = df["order_date_time"].dt.day
            df["order_hour"] = df["order_date_time"].dt.hour

        if "issue_responded" in df.columns and "Issue_reported at" in df.columns:
            df["response_time_minutes"] = (
                df["issue_responded"] - df["Issue_reported at"]
            ).dt.total_seconds() / 60

        df = self._ensure_required_columns(df)

        for col in self.numeric_impute_cols:
            if fit:
                median = df[col].median()
                if pd.isna(median):
                    median = 0.0
                self.numeric_medians_[col] = float(median)

            fill_value = self.numeric_medians_.get(col, 0.0)
            df[col] = df[col].fillna(fill_value)

        for col in self.date_mode_cols:
            if fit:
                mode_vals = df[col].mode(dropna=True)
                if mode_vals.empty:
                    mode_val = pd.Timestamp("1970-01-01")
                else:
                    mode_val = mode_vals.iloc[0]
                self.date_modes_[col] = mode_val

            mode_value = self.date_modes_.get(col, pd.Timestamp("1970-01-01"))
            df[col] = df[col].fillna(mode_value)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("Unknown")

        if "Customer Remarks" in df.columns:
            df["Customer Remarks"] = (
                df["Customer Remarks"].astype(str).str.lower().str.strip()
            )

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target_col]

        if fit:
            self.first_iqr_bounds_ = self._compute_iqr_bounds(df, numeric_cols)
        df = self._apply_iqr_bounds(df, self.first_iqr_bounds_)

        df = df.drop(columns=[c for c in self.drop_after_engineering if c in df.columns])

        numeric_cols_2 = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols_2 = [col for col in numeric_cols_2 if col != self.target_col]

        if fit:
            self.second_iqr_bounds_ = self._compute_iqr_bounds(df, numeric_cols_2)
        df = self._apply_iqr_bounds(df, self.second_iqr_bounds_)

        return df

    def _encode_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.drop(columns=[c for c in self.high_card_drop_cols if c in X.columns])

        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].fillna("Unknown")

        if fit:
            safe_categorical_cols = self.base_safe_categorical.copy()
            for col in categorical_cols:
                if col not in safe_categorical_cols:
                    safe_categorical_cols.append(col)
            self.safe_categorical_cols_ = safe_categorical_cols

        for col in self.safe_categorical_cols_:
            if col not in X.columns:
                X[col] = "Unknown"

        encode_cols = [c for c in self.safe_categorical_cols_ if c in X.columns]
        X = pd.get_dummies(X, columns=encode_cols, drop_first=True)

        if fit:
            self.encoded_columns_ = X.columns.tolist()
        else:
            X = X.reindex(columns=self.encoded_columns_, fill_value=0)

        return X

    def fit(self, raw_df: pd.DataFrame) -> "NotebookPreprocessor":
        df = self._base_wrangling(raw_df=raw_df, fit=True)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in input data.")

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        X = self._encode_features(X, fit=True)

        numeric_X = X.select_dtypes(include=["int64", "float64"])
        corr_matrix = numeric_X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.high_corr_features_ = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > 0.85)
        ]
        X = X.drop(columns=self.high_corr_features_, errors="ignore")

        self.variance_filter_ = VarianceThreshold(threshold=0.01)
        X_var = self.variance_filter_.fit_transform(X)
        self.selected_columns_var_ = X.columns[self.variance_filter_.get_support()].tolist()
        X = pd.DataFrame(X_var, columns=self.selected_columns_var_, index=X.index)

        temp_df = X.copy()
        temp_df[self.target_col] = y.values
        corr_with_target = temp_df.corr()[self.target_col].abs().sort_values(ascending=False)

        selected_corr = corr_with_target[corr_with_target > 0.02].index.tolist()
        if self.target_col in selected_corr:
            selected_corr.remove(self.target_col)
        self.selected_features_corr_ = selected_corr

        X = X.reindex(columns=self.selected_features_corr_, fill_value=0)

        self.select_k_best_ = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        _ = self.select_k_best_.fit_transform(X, y)

        self.selected_columns_kbest_ = X.columns[self.select_k_best_.get_support()].tolist()
        self.final_feature_order_ = self.selected_columns_kbest_.copy()

        self.fitted_ = True
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("NotebookPreprocessor must be fitted before calling transform().")

        df = self._base_wrangling(raw_df=raw_df, fit=False)
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        X = self._encode_features(X=df, fit=False)

        X = X.drop(columns=self.high_corr_features_, errors="ignore")
        X = X.reindex(columns=self.selected_columns_var_, fill_value=0)
        X = X.reindex(columns=self.selected_features_corr_, fill_value=0)
        X = X.reindex(columns=self.selected_columns_kbest_, fill_value=0)

        return X.astype(np.float32)

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(raw_df)
        return self.transform(raw_df)

    def feature_audit(self) -> FeatureAudit:
        if not self.fitted_:
            raise RuntimeError("NotebookPreprocessor must be fitted before generating feature audit.")

        return FeatureAudit(
            final_feature_order=self.final_feature_order_.copy(),
            high_corr_dropped=self.high_corr_features_.copy(),
            variance_selected=self.selected_columns_var_.copy(),
            corr_selected=self.selected_features_corr_.copy(),
            kbest_selected=self.selected_columns_kbest_.copy(),
        )
