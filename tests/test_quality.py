import pandas as pd
import pytest

from notebook_pipeline.errors import ArtifactError, DataContractError
from notebook_pipeline.quality import (
    enforce_quality,
    read_sales,
    valid_sales_rows,
    validate_columns,
)


def test_missing_file_preserves_cause(tmp_path):
    with pytest.raises(ArtifactError) as caught:
        read_sales(tmp_path / "missing.csv")
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_schema_and_quality_budget():
    with pytest.raises(DataContractError, match="required columns"):
        validate_columns(pd.DataFrame({"quantity": [1]}))
    enforce_quality(100, 2)
    with pytest.raises(DataContractError, match="exceeds"):
        enforce_quality(100, 3)
    with pytest.raises(DataContractError, match="No valid"):
        enforce_quality(0, 0)


def test_row_contract_rejects_nonfinite_fractional_and_unknown_values():
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d", "e"],
            "order_date": pd.to_datetime(["2025-01-01"] * 5),
            "region": ["East"] * 4 + ["Unknown"],
            "category": ["Home"] * 5,
            "quantity": [1, 1.5, 1, 1, 1],
            "unit_price": [10, 10, float("inf"), 10, 10],
            "unit_cost": [5, 5, 5, -1, 5],
        }
    )
    assert valid_sales_rows(frame).tolist() == [True, False, False, False, False]
