import pandas as pd

def recommend_factory(df, product_name):
    """
    Recommend the best factory for the selected product.
    Ranking is based on:
    1. Lowest Lead Time
    2. Highest Profit Margin
    """

    product_df = df[df["Product Name"] == product_name]

    if product_df.empty:
        return pd.DataFrame()

    recommendation = (
        product_df.groupby("Factory")
        .agg({
            "Lead Time": "mean",
            "Profit Margin (%)": "mean",
            "Sales": "sum"
        })
        .reset_index()
    )

    recommendation = recommendation.sort_values(
        by=["Lead Time", "Profit Margin (%)"],
        ascending=[True, False]
    )

    return recommendation