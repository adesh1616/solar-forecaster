import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. Simulate a realistic 24-hour weather forecast payload matching the API schema
hourly_features = []
for hour in range(24):
    # Simulate a diurnal cycle: high irradiance at midday, 0 at night
    if 6 <= hour <= 18:
        irradiance = 400.0 + 400.0 * (1.0 - abs(hour - 12) / 6)
        temp = 20.0 + 8.0 * (1.0 - abs(hour - 12) / 6)
        cloud = 15.0
    else:
        irradiance = 0.0
        temp = 14.0
        cloud = 5.0
        
    hourly_features.append({
        "hour": hour,
        "temperature_c": round(temp, 2),
        "cloud_cover_pct": round(cloud, 2),
        "irradiance_wm2": round(irradiance, 2)
    })

payload = {
    "forecast_date": "2026-07-04",
    "hourly_features": hourly_features
}

url = "http://127.0.0.1:8000/api/v1/forecast"

# 2. Make HTTP POST request with error handling
try:
    print(f"Sending live request to FastAPI server at: {url}...")
    response = requests.post(url, json=payload)
    
    # Check if the server returned an error (e.g. 422 Unprocessable Entity)
    if response.status_code != 200:
        print(f"Error: Server returned status code {response.status_code}")
        print("Response detail:")
        print(response.text)
    else:
        # 3. Extract JSON response and convert to DataFrame
        response_json = response.json()
        forecast_date = response_json["forecast_date"]
        df = pd.DataFrame(response_json["forecast"])
        
        # 4. Create the visualization with professional styling
        plt.figure(figsize=(10, 6))

        # Plot the main forecast line
        plt.plot(
            df["hour"], 
            df["predicted_generation_mw"], 
            color="#E65100",          # Dark orange
            linewidth=2.5, 
            marker="o", 
            markersize=6,
            label="Predicted Generation"
        )

        # Shaded area under the curve
        plt.fill_between(
            df["hour"], 
            df["predicted_generation_mw"], 
            color="#FFE0B2",          # Light orange
            alpha=0.5
        )

        # Titles and labels
        plt.title(f"Live Solar Generation Forecast for {forecast_date}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Hour of the Day (0-23)", fontsize=11, labelpad=10)
        plt.ylabel("Predicted Power (Megawatts)", fontsize=11, labelpad=10)

        # Configure axes
        plt.xlim(0, 23)
        plt.xticks(range(24))
        plt.ylim(bottom=0)

        # Dashed background grid
        plt.grid(True, linestyle="--", alpha=0.5)

        # Optimize layout and display
        plt.tight_layout()
        print("Displaying the live forecast chart...")
        plt.show()

except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to the FastAPI server.")
    print("Please make sure your FastAPI microservice is running (e.g., via 'uvicorn app.main:app --reload') before executing this script.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
