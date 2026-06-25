import os
import pandas as pd

"""
ModalAI Shopify SKU-Sales Dataset (Experiment 3)
================================================
Grain: one row per (sku, ISO week). Target: units_sold (Shopify line quantity).
Single clean source (shopify_order_line + shopify_order) over the full ~70 weeks /
15.8 months - the most model-ready demand signal in the extract (vs the component panel's
late-starting internal history). Built by forecasting/scripts/build_shopify_sku_panel.py.

TASK: REGRESSION - predict a SKU's weekly units sold from features known BEFORE that week.
Heavy-tailed (median line qty ~2, max ~922k bulk/entry) -> winsorize + log1p like Exp 1.
LEAKAGE: fulfilled/unfulfilled/status are realized after the order; same-week price and
line-count and the raw sku string are excluded (see clean_sku_sales.py).
"""


def load_sku_sales_data(curr_dir):
    data_path = os.path.join(
        curr_dir, "../../datasets/ForecastDemandSets/sku_week_panel.csv"
    )
    data = pd.read_csv(data_path, parse_dates=["week_start"], low_memory=False)
    print(f"[load_sku_sales] Loaded {len(data)} sku-weeks "
          f"({data['sku'].nunique()} SKUs x {data['week_start'].nunique()} weeks)")
    print(f"[load_sku_sales] Week range: {data['week_start'].min().date()} -> "
          f"{data['week_start'].max().date()}")
    return data
