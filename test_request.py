import time
import json
import urllib.request
import urllib.error

# We will use urllib.request to avoid dependency on 'requests' library,
# but we will also import requests to be compliant if they run it.
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False

URL = "http://127.0.0.1:8000/api/v1/forecast"
HEALTH_URL = "http://127.0.0.1:8000/health"

# Generate 24 hours of weather features
# hour, temperature_c, cloud_cover_pct, irradiance_wm2
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
    "forecast_date": "2026-06-25",
    "hourly_features": hourly_features
}

def check_health():
    print(f"Checking health endpoint: {HEALTH_URL}")
    if USE_REQUESTS:
        try:
            response = requests.get(HEALTH_URL)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}\n")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}\n")
            return False
    else:
        try:
            with urllib.request.urlopen(HEALTH_URL) as response:
                status_code = response.getcode()
                body = json.loads(response.read().decode())
                print(f"Status: {status_code}")
                print(f"Response: {body}\n")
                return status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}\n")
            return False

def test_forecast():
    print(f"Sending 24-hour forecast request to: {URL}")
    if USE_REQUESTS:
        try:
            response = requests.post(URL, json=payload)
            print(f"Status: {response.status_code}")
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"Forecast request failed: {e}")
    else:
        try:
            req = urllib.request.Request(
                URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as response:
                status_code = response.getcode()
                body = json.loads(response.read().decode())
                print(f"Status: {status_code}")
                print("Response:")
                print(json.dumps(body, indent=2))
        except urllib.error.HTTPError as e:
            print(f"Forecast request failed: {e.code} - {e.read().decode()}")
        except Exception as e:
            print(f"Forecast request failed: {e}")

if __name__ == "__main__":
    # Wait for the server to spin up
    print("Waiting 3 seconds for server to start...")
    time.sleep(3)
    if check_health():
        test_forecast()
    else:
        print("FastAPI server health check failed or server is not running.")
