from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO

# Initialize FastAPI
app = FastAPI()

# Example model and dataset (replace with your actual implementation)
model = None  # Placeholder for your trained model
pivoted_data = None  # Placeholder for your dataset

# Route to train model (optional, for testing)
@app.post("/train")
async def train_model():
    global model, pivoted_data
    # Mock training for demo (replace with your logic)
    pivoted_data = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=12, freq="M"),
        "Disease1": np.random.randint(50, 100, 12),
        "Disease2": np.random.randint(30, 60, 12),
    })
    pivoted_data.set_index("Date", inplace=True)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = pivoted_data.shift(1).dropna()
    y = pivoted_data.iloc[1:]
    model.fit(X, y)
    return {"status": "Model trained successfully"}

# Route to make predictions
@app.post("/predict")
async def predict_diseases(month_name: str = Form(...)):
    global model, pivoted_data
    if model is None or pivoted_data is None:
        return JSONResponse(content={"error": "Model not trained"}, status_code=400)

    # Map month to integer
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if month_name not in month_map:
        return JSONResponse(content={"error": "Invalid month name"}, status_code=400)

    user_month = month_map[month_name]
    future_index = pd.date_range(start=pivoted_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    future_data = pd.DataFrame(index=future_index, columns=pivoted_data.columns).fillna(0)

    # Rolling predictions
    for i in range(len(future_data)):
        if i == 0:
            features = pivoted_data.iloc[-1:].copy()
        else:
            features = future_data.iloc[i - 1:i].copy()
        future_data.iloc[i] = model.predict(features)[0]

    specified_month_prediction = future_data.loc[future_data.index.month == user_month]
    return specified_month_prediction.to_dict()

# Route to fetch generated report or image
@app.get("/report")
async def fetch_report():
    plt.figure(figsize=(10, 6))
    plt.title("Sample Disease Prediction")
    plt.bar(["Disease1", "Disease2"], [80, 40])  # Replace with actual predictions
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return JSONResponse(content={"message": "Report created successfully"})
