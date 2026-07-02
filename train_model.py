import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("Step 1: Loading datasets...")

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
weather_path = os.path.join(base_dir, "data", "weather_data.csv")
pv_path = os.path.join(base_dir, "data", "miris_pv.csv")

# Read datasets
weather_df = pd.read_csv(weather_path)
pv_df = pd.read_csv(pv_path)

# Automatically identify the datetime columns
print("Step 2: Auto-detecting Datetime/Time columns...")
weather_time_cols = [c for c in weather_df.columns if 'time' in c.lower() or 'date' in c.lower()]
pv_time_cols = [c for c in pv_df.columns if 'time' in c.lower() or 'date' in c.lower()]

if not weather_time_cols:
    raise ValueError("Could not find a Datetime/Time column in weather_data.csv")
if not pv_time_cols:
    raise ValueError("Could not find a Datetime/Time column in miris_pv.csv")

weather_time_col = weather_time_cols[0]
pv_time_col = pv_time_cols[0]

print(f"   - Identified weather datetime column: '{weather_time_col}'")
print(f"   - Identified PV datetime column: '{pv_time_col}'")

# Convert timestamps to datetime (with UTC alignment to handle different timezone offsets)
weather_df[weather_time_col] = pd.to_datetime(weather_df[weather_time_col], utc=True)
pv_df[pv_time_col] = pd.to_datetime(pv_df[pv_time_col], utc=True)

# Rename to match and merge
weather_df = weather_df.rename(columns={weather_time_col: 'timestamp'})
pv_df = pv_df.rename(columns={pv_time_col: 'timestamp'})

print("Merging datasets using inner join on timestamp...")
merged_df = pd.merge(weather_df, pv_df, on='timestamp', how='inner')
print(f"   - Merged dataset contains {len(merged_df)} rows.")

# Extract hour (0-23)
print("Step 3: Extracting hour of day...")
merged_df['hour'] = merged_df['timestamp'].dt.hour

# Map/Rename features intelligently
print("Step 4: Mapping Liege microgrid column abbreviations...")

# Surface Temperature mapping
temp_col = None
for col in merged_df.columns:
    if col.upper() == 'ST':
        temp_col = col
        break
if temp_col is None:
    for col in merged_df.columns:
        if 'temp' in col.lower() and col != 'temperature_c':
            temp_col = col
            break

if temp_col:
    print(f"   - Mapping '{temp_col}' to 'temperature_c'")
    merged_df['temperature_c'] = merged_df[temp_col]
else:
    raise ValueError("Could not identify temperature/surface temperature column (ST/temp).")

# Cloud Cover mapping (combine CD, CM, CU using random overlap, converting to 0-100 scale)
cloud_cols = [c for c in ['CD', 'CM', 'CU'] if c in merged_df.columns]
if len(cloud_cols) == 3:
    print(f"   - Combining cloud cover columns {cloud_cols} using standard random overlap formula...")
    # Determine if values are 0-1 or 0-100
    max_cloud_val = merged_df[cloud_cols].max().max()
    is_zero_to_one = max_cloud_val <= 1.05
    
    cd = merged_df['CD'] if is_zero_to_one else merged_df['CD'] / 100.0
    cm = merged_df['CM'] if is_zero_to_one else merged_df['CM'] / 100.0
    cu = merged_df['CU'] if is_zero_to_one else merged_df['CU'] / 100.0
    
    # Random overlap formula: 1 - (1 - CD) * (1 - CM) * (1 - CU)
    combined = 1.0 - (1.0 - cd) * (1.0 - cm) * (1.0 - cu)
    merged_df['cloud_cover_pct'] = combined * 100.0
else:
    # Fallback to any generic cloud column
    fallback_cloud = [c for c in merged_df.columns if 'cloud' in c.lower()]
    if fallback_cloud:
        print(f"   - Mapping fallback cloud column '{fallback_cloud[0]}' to 'cloud_cover_pct'")
        merged_df['cloud_cover_pct'] = merged_df[fallback_cloud[0]]
    else:
        raise ValueError("Could not identify cloud cover columns (CD/CM/CU/cloud).")

# Irradiance mapping (prefer SWD for downward shortwave radiation)
irrad_col = None
if 'SWD' in merged_df.columns:
    irrad_col = 'SWD'
else:
    for col in merged_df.columns:
        if 'irradiance' in col.lower() or 'radiation' in col.lower() or 'solar' in col.lower():
            irrad_col = col
            break

if irrad_col:
    print(f"   - Mapping '{irrad_col}' to 'irradiance_wm2'")
    merged_df['irradiance_wm2'] = merged_df[irrad_col]
else:
    raise ValueError("Could not identify solar radiation/irradiance column.")

# PV Power target mapping (prefer PV)
target_col = None
if 'PV' in merged_df.columns:
    target_col = 'PV'
else:
    for col in merged_df.columns:
        if 'pv' in col.lower() or 'power' in col.lower() or 'generation' in col.lower():
            target_col = col
            break

if target_col:
    print(f"   - Mapping '{target_col}' to 'target_mw'")
    merged_df['target_mw'] = merged_df[target_col]
else:
    raise ValueError("Could not identify PV power generation column.")

# Drop rows with NaNs in required columns
required_features = ['hour', 'temperature_c', 'cloud_cover_pct', 'irradiance_wm2', 'target_mw']
print(f"Step 5: Dropping rows with NaNs in: {required_features}")
initial_len = len(merged_df)
merged_df = merged_df.dropna(subset=required_features)
print(f"   - Dropped {initial_len - len(merged_df)} rows containing NaNs. {len(merged_df)} rows remaining.")

# Split data (80/20)
print("Step 6: Splitting data (80/20 train/test)...")
X = merged_df[['hour', 'temperature_c', 'cloud_cover_pct', 'irradiance_wm2']]
y = merged_df['target_mw']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
print("Training RandomForestRegressor(n_estimators=100, random_state=42)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("Step 7: Evaluating model on test set...")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"   - Mean Absolute Error (MAE): {mae:.4f} MW")

# Save model
print("Step 8: Saving model...")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "solar_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Success! Trained model saved to: {model_path}")