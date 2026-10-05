"""Generate a reproducible fictional dataset, including deliberate quality defects."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", "2025-06-30")
    rows = []
    for day in dates:
        # More orders later in the period; category economics differ.
        for _ in range(int(rng.poisson(12 + day.month * 2))):
            category = str(rng.choice(["Home", "Office", "Technology"]))
            price = {"Home": 45, "Office": 18, "Technology": 120}[category]
            price = round(price * rng.uniform(0.8, 1.2), 2)
            rows.append(
                {
                    "order_id": f"ORD-{len(rows) + 1:06d}",
                    "order_date": day.strftime("%Y-%m-%d"),
                    "region": str(rng.choice(["East", "West", "North", "South"])),
                    "category": category,
                    "quantity": int(rng.integers(1, 6)),
                    "unit_price": price,
                    "unit_cost": round(price * rng.uniform(0.45, 0.85), 2),
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[5, "order_date"] = "not-a-date"
    frame.loc[10, "quantity"] = -2
    frame.loc[20, "unit_price"] = np.nan
    frame.loc[30, "region"] = "  east  "
    frame = pd.concat([frame, frame.iloc[[40, 41, 42, 43]]], ignore_index=True)
    destination = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame):,} rows to {destination}")


if __name__ == "__main__":
    main()
