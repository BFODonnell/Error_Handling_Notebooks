"""Small data contracts: fail at boundaries, recover only where policy is explicit."""

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ArtifactError, DataContractError

REQUIRED_COLUMNS = {
    "order_id",
    "order_date",
    "region",
    "category",
    "quantity",
    "unit_price",
    "unit_cost",
}


def read_sales(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as error:
        raise ArtifactError(
            f"Sales CSV not found at {path}. Check the input_csv parameter."
        ) from error
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataContractError(
            "Sales input must be a nonempty, UTF-8, rectangular CSV."
        ) from error


def validate_columns(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DataContractError(f"Sales CSV is missing required columns: {sorted(missing)}.")
    if frame.empty:
        raise DataContractError("Sales CSV contains no rows.")


def valid_sales_rows(frame: pd.DataFrame) -> pd.Series:
    """Apply to normalized columns. This example excludes returns and missing dimensions."""
    numeric = frame[["quantity", "unit_price", "unit_cost"]]
    return (
        frame["order_date"].notna()
        & frame["order_id"].notna()
        & frame["order_id"].ne("")
        & frame["region"].isin(["East", "West", "North", "South"])
        & frame["category"].isin(["Home", "Office", "Technology"])
        & np.isfinite(numeric).all(axis=1)
        & frame["quantity"].gt(0)
        & frame["quantity"].mod(1).eq(0)
        & frame["unit_price"].gt(0)
        & frame["unit_cost"].ge(0)
    ).fillna(False)


def enforce_quality(total: int, rejected: int, *, max_rejected_fraction: float = 0.02) -> None:
    if total == 0 or rejected == total:
        raise DataContractError("No valid sales rows remain after cleaning.")
    if rejected / total > max_rejected_fraction:
        raise DataContractError(
            f"Rejected {rejected}/{total} rows; "
            f"exceeds the {max_rejected_fraction:.0%} quality budget. "
            "Fix the source data before publishing results."
        )
