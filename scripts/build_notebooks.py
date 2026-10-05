"""Keep notebook generation reproducible; generated .ipynb files are the deliverables."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source, **metadata):
    return nbf.v4.new_code_cell(dedent(source).strip(), metadata=metadata)


def setup(stage, imports):
    return code(f"""\
from notebook_pipeline import start_notebook
session = start_notebook({stage!r}, run_id=run_id)
run_id = session.run_id
{imports}
""")


def parameters():
    return code(
        """
        # The runner injects these values. For manual use, copy run_id from notebook 01.
        run_id = None
        input_csv = None
        demo_failure = None
    """,
        tags=["parameters"],
    )


def save(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    # Stable cell IDs make records comparable across runs and rebuilds.
    for index, cell in enumerate(notebook.cells):
        cell.id = f"{name}-{index:02d}"
    nbf.validate(notebook)
    path = ROOT / "notebooks" / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbf.write(notebook, path)


save(
    "01_ingest",
    [
        markdown("""
        # 01 · Ingest and validate retail sales
        **Question:** Which categories drive sales growth, and does that growth preserve gross margin?

        Our fictional retailer supplies one row per order, with one category per order. All money
        is USD; dates are calendar dates. This dataset covers January–June 2025 and is synthetic.
        Returns are outside this example's contract. A few bad rows and exact duplicates are deliberate.

        Run all three notebooks with `uv run run-pipeline`, or run this notebook top to bottom and
        copy its printed `run_id` into the parameters cell of notebooks 02 and 03.
        Start Jupyter with `uv run jupyter lab` from the project root and choose its Python kernel.

        One setup call installs automatic per-cell logging. Optional `# step:` comments give log
        entries meaningful names. Errors keep their normal Jupyter traceback and fail this run.
    """),
        parameters(),
        setup(
            "01_ingest",
            """from pathlib import Path
import pandas as pd
from notebook_pipeline.quality import read_sales, validate_columns, valid_sales_rows, enforce_quality
from notebook_pipeline.errors import DataContractError
from notebook_pipeline.runtime import write_json""",
        ),
        markdown(
            "## Read the CSV\nMissing/unreadable input is fatal. The default dataset is bundled locally."
        ),
        code("""
        # step: read_csv
        source = Path(input_csv) if input_csv else session.run.root / "data/raw/sales.csv"
        if demo_failure == "missing-file":
            source = session.run.path / "deliberately_missing.csv"
        sales = read_sales(source)
        sales.head()
    """),
        code("""
        # step: validate_schema
        if demo_failure == "schema":
            sales = sales.drop(columns="unit_price")
        validate_columns(sales)
        raw_rows = len(sales)
        sales.shape
    """),
        markdown("""
        ## Normalize and enforce quality
        Exact duplicate records can be removed safely. Conflicting records with the same order ID
        are fatal. Invalid rows can be excluded only if they are at most **2%** of nonduplicate rows;
        this is an explicit example business policy. No financial values are imputed.
    """),
        code("""
        # step: normalize_types
        sales = sales.copy()
        for column in ["order_id", "region", "category"]:
            sales[column] = sales[column].astype("string").str.strip()
        sales["region"] = sales["region"].str.title()
        sales["category"] = sales["category"].str.title()
        sales["order_date"] = pd.to_datetime(sales["order_date"], format="%Y-%m-%d", errors="coerce")
        for column in ["quantity", "unit_price", "unit_cost"]:
            sales[column] = pd.to_numeric(sales[column], errors="coerce")
    """),
        code("""
        # step: deduplicate
        duplicate_rows = int(sales.duplicated().sum())
        sales = sales.drop_duplicates().copy()
        if sales["order_id"].dropna().duplicated().any():
            raise DataContractError("Conflicting records share an order_id; resolve them at the source.")
        if duplicate_rows:
            session.log.warning("data.exact_duplicates_removed", extra={"fields": {"rows": duplicate_rows}})
    """),
        code("""
        # step: enforce_row_quality
        valid = valid_sales_rows(sales)
        rejected_rows = int((~valid).sum())
        enforce_quality(len(sales), rejected_rows)
        clean = sales.loc[valid].copy()
        clean["quantity"] = clean["quantity"].astype("int64")
        if rejected_rows:
            session.log.warning("data.invalid_rows_excluded", extra={"fields": {"rows": rejected_rows}})
        quality_report = {
            "raw_rows": raw_rows, "duplicates_removed": duplicate_rows,
            "invalid_rows_excluded": rejected_rows, "clean_rows": len(clean),
            "max_rejected_fraction": 0.02,
        }
        quality_report
    """),
        code("""
        # step: publish_clean_data
        with session.output("clean_sales.parquet") as path:
            clean.to_parquet(path, index=False)
        with session.output("quality_report.json") as path:
            write_json(path, quality_report)
    """),
        markdown("""
        ## Complete the handoff
        Outputs become consumable only when this entire final cell succeeds. A failed stage cannot
        be resumed in this example: fix the cause, then start a fresh run from notebook 01.
        In interactive Jupyter, later cells can still be clicked after an error, but publishing and
        downstream handoff are blocked. The automated runner stops immediately.
    """),
        code("""
        # step: finish_ingestion
        print(f"Use run_id = {run_id!r} in notebook 02.")
        session.finish()
    """),
    ],
)

save(
    "02_analyze",
    [
        markdown("""
        # 02 · Analyze sales growth and gross profit
        Consume only the successful, checksum-verified output of notebook 01 from this exact run.
        Revenue = quantity × unit price; gross profit = revenue − quantity × unit cost.
        Gross margin uses total profit / total revenue, rather than averaging order-level margins.
        This is descriptive analysis, not a causal claim or forecast.
    """),
        parameters(),
        setup(
            "02_analyze",
            """import pandas as pd
from notebook_pipeline.runtime import write_json""",
        ),
        code("""
        # step: load_clean_sales
        sales = pd.read_parquet(session.input("01_ingest", "clean_sales.parquet"))
        sales.head()
    """),
        code("""
        # step: calculate_order_economics
        if demo_failure == "unexpected":
            raise RuntimeError("Deliberate analysis failure: testing automatic cell traceback logging.")
        sales = sales.assign(
            revenue=lambda df: df["quantity"] * df["unit_price"],
            cost=lambda df: df["quantity"] * df["unit_cost"],
            month=lambda df: df["order_date"].dt.to_period("M").dt.to_timestamp(),
        )
        sales["gross_profit"] = sales["revenue"] - sales["cost"]
    """),
        markdown("""
        ## Monthly performance
        Reindex the full month sequence so a missing month is explicit. Growth is undefined for
        the first month and after a zero-revenue month; preserve those values as missing.
    """),
        code("""
        # step: monthly_summary
        monthly = sales.groupby("month").agg(
            revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"),
            orders=("order_id", "nunique"), units=("quantity", "sum"),
        )
        months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS", name="month")
        monthly = monthly.reindex(months, fill_value=0)
        monthly["gross_margin"] = monthly["gross_profit"].div(monthly["revenue"].replace(0, float("nan")))
        previous_revenue = monthly["revenue"].shift(1).replace(0, float("nan"))
        monthly["revenue_growth"] = monthly["revenue"].div(previous_revenue).sub(1)
        monthly = monthly.reset_index()
        monthly.round(3)
    """),
        code("""
        # step: category_summary
        categories = sales.groupby("category", as_index=False).agg(
            revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"),
            orders=("order_id", "nunique"),
        )
        categories["gross_margin"] = categories["gross_profit"] / categories["revenue"]
        categories = categories.sort_values("revenue", ascending=False).reset_index(drop=True)
        categories.round(3)
    """),
        code("""
        # step: summarize_kpis
        kpis = {
            "orders": int(sales["order_id"].nunique()),
            "revenue": round(float(sales["revenue"].sum()), 2),
            "gross_profit": round(float(sales["gross_profit"].sum()), 2),
            "gross_margin": float(sales["gross_profit"].sum() / sales["revenue"].sum()),
            "top_category": str(categories.iloc[0]["category"]),
        }
        session.log.info("analysis.summarized", extra={"fields": {"orders": kpis["orders"]}})
        kpis
    """),
        code("""
        # step: publish_analysis
        with session.output("monthly_sales.parquet") as path:
            monthly.to_parquet(path, index=False)
        with session.output("category_sales.parquet") as path:
            categories.to_parquet(path, index=False)
        with session.output("kpis.json") as path:
            write_json(path, kpis)
    """),
        code("""
        # step: finish_analysis
        print(f"Use run_id = {run_id!r} in notebook 03.")
        session.finish()
    """),
    ],
)

save(
    "03_visualize",
    [
        markdown("""
        # 03 · Visualize sales and margin
        Three complementary views: revenue and gross profit over time, category revenue,
        and monthly gross margin. The saved figure and executed notebook belong to this run.
    """),
        parameters(),
        setup(
            "03_visualize",
            """import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter""",
        ),
        code("""
        # step: load_analysis
        monthly = pd.read_parquet(session.input("02_analyze", "monthly_sales.parquet"))
        categories = pd.read_parquet(session.input("02_analyze", "category_sales.parquet"))
        kpis = json.loads(session.input("02_analyze", "kpis.json").read_text(encoding="utf-8"))
    """),
        code("""
        # step: configure_figure
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.titleweight": "bold"})
        # Disable automatic display until all panels are ready.
        plt.ioff()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        fig.patch.set_facecolor("#f8fafc")
        fig.suptitle("Retail performance | January–June 2025", fontsize=21, fontweight="bold")
        money = FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k")
    """),
        code("""
        # step: plot_monthly_revenue
        labels = monthly["month"].dt.strftime("%b")
        axes[0].plot(labels, monthly["revenue"], color="#2563eb", marker="o", linewidth=2.5, label="Revenue")
        axes[0].plot(labels, monthly["gross_profit"], color="#0d9488", marker="o", linewidth=2.5, label="Gross profit")
        axes[0].set(title="Growth over time", ylabel="USD", ylim=(0, None))
        axes[0].yaxis.set_major_formatter(money)
        axes[0].legend(frameon=False)
    """),
        code("""
        # step: plot_category_revenue
        ordered = categories.sort_values("revenue")
        axes[1].barh(ordered["category"], ordered["revenue"], color="#2563eb", height=0.55)
        axes[1].set(title="Revenue by category", xlabel="USD")
        axes[1].xaxis.set_major_formatter(money)
    """),
        code("""
        # step: plot_gross_margin
        axes[2].plot(labels, monthly["gross_margin"], color="#7c3aed", marker="o", linewidth=2.5)
        axes[2].axhline(kpis["gross_margin"], linestyle="--", color="#64748b", label="Period margin")
        axes[2].yaxis.set_major_formatter(PercentFormatter(1))
        axes[2].set(title="Gross margin", ylim=(0, max(0.5, float(monthly["gross_margin"].max()) + 0.05)))
        axes[2].legend(frameon=False)
        for axis in axes:
            axis.grid(axis="y", alpha=0.15)
        fig
    """),
        code("""
        # step: save_dashboard
        with session.output("sales_dashboard.png") as path:
            fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Top category: {kpis['top_category']}; period gross margin: {kpis['gross_margin']:.1%}")
    """),
        code("""
        # step: finish_visualization
        print(f"Dashboard: {session.run.path / 'artifacts/03_visualize/sales_dashboard.png'}")
        session.finish()
    """),
    ],
)

print("Generated three notebooks.")
