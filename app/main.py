from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from app.model import predict_solar_output

app = FastAPI(
    title="GridCast Solar Forecaster",
    description="Microservice predicting solar energy generation based on hourly weather features."
)

class HourlyWeatherFeature(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    cloud_cover_pct: float = Field(..., ge=0.0, le=100.0, description="Cloud cover percentage (0-100)")
    irradiance_wm2: float = Field(..., ge=0.0, description="Solar irradiance in W/m2")

class ForecastRequest(BaseModel):
    forecast_date: str = Field(..., description="Date for the forecast in YYYY-MM-DD format")
    hourly_features: List[HourlyWeatherFeature] = Field(..., description="List of weather features for exactly 24 hours")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/forecast")
def forecast(payload: ForecastRequest):
    # 1. Validate that exactly 24 hours of data are provided
    if len(payload.hourly_features) != 24:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exactly 24 hours of data must be provided. Got {len(payload.hourly_features)} hours."
        )
    
    # 2. Convert the Pydantic payload into a Pandas DataFrame
    # Ensure correct columns order matching model training: ["hour", "temperature_c", "cloud_cover_pct", "irradiance_wm2"]
    data = []
    for item in payload.hourly_features:
        data.append({
            "hour": item.hour,
            "temperature_c": item.temperature_c,
            "cloud_cover_pct": item.cloud_cover_pct,
            "irradiance_wm2": item.irradiance_wm2
        })
    df = pd.DataFrame(data)
    
    # 3. Pass the DataFrame to predict_solar_output
    predictions = predict_solar_output(df)
    
    # 4. Return JSON mapping each hour to its predicted_generation_mw (rounded to 2 decimal places)
    forecast_results = []
    for i, item in enumerate(payload.hourly_features):
        forecast_results.append({
            "hour": item.hour,
            "predicted_generation_mw": round(float(predictions[i]), 2)
        })
        
    return {
        "forecast_date": payload.forecast_date,
        "forecast": forecast_results
    }
