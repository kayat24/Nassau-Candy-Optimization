import joblib
from pathlib import Path
import pandas as pd

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Model
MODEL_PATH = BASE_DIR / "Models" / "random_forest_model.pkl"
model = joblib.load(MODEL_PATH)


def predict_lead_time(
    sales,
    units,
    cost,
    gross_profit,
    ship_mode,
    region,
    factory
):
    """
    Predict Lead Time using Random Forest Model
    """

    input_df = pd.DataFrame({
        "Sales": [sales],
        "Units": [units],
        "Cost": [cost],
        "Gross Profit": [gross_profit],
        "Ship Mode": [ship_mode],
        "Region": [region],
        "Factory": [factory]
    })

    prediction = model.predict(input_df)

    return round(float(prediction[0]), 2)