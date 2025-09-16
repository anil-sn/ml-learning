# Project 21: Indoor Localization using Wi-Fi Signal Strength (RSSI)

## Objective
Build a multi-class classification model that can predict a user's specific location (a unique room or space) within a building by using the Received Signal Strength Indicator (RSSI) from numerous nearby Wi-Fi Access Points.

## Business Value
- **Indoor Navigation Systems**: Enable precise navigation in shopping malls, airports, hospitals, and large office buildings
- **Location-Based Services**: Provide targeted advertising, personalized recommendations, and context-aware mobile applications
- **Asset Tracking**: Track valuable equipment, inventory, and personnel within warehouses and industrial facilities
- **Emergency Response**: Quickly locate personnel during emergencies in large buildings
- **Network Optimization**: Identify critical Access Points for location services and optimize their placement

## Core Libraries
- **pandas & numpy**: Data manipulation and numerical operations
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **matplotlib**: Data visualization and plotting
- **RandomForestClassifier**: Ensemble learning for multi-class location prediction

## Dataset
**Source**: Kaggle - "UJIIndoorLoc Data Set"
- **Size**: Wi-Fi fingerprints from 520 different Access Points
- **Features**: RSSI values from 520 WAPs (Wireless Access Points)
- **Target**: Building, floor, and room location combinations
- **Scope**: Multiple buildings with hundreds of unique indoor locations
- **Quality**: Comprehensive dataset with training and validation splits

## Step-by-Step Guide

### 1. Environment Setup
```python
# Install required packages
pip install pandas numpy scikit-learn matplotlib kaggle

# Set up Kaggle API for dataset download
# Upload kaggle.json file when prompted
```

### 2. Data Collection and Preprocessing
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Download and load UJIIndoorLoc dataset
df_train = pd.read_csv('trainingData.csv')
df_val = pd.read_csv('validationData.csv') 
df = pd.concat([df_train, df_val], ignore_index=True)

# Handle missing signal values (100 -> -105 dBm)
wap_cols = [f'WAP{str(i).zfill(3)}' for i in range(1, 521)]
df[wap_cols] = df[wap_cols].replace(100, -105)

# Create unique location identifier
df['location'] = (df['BUILDINGID'].astype(str) + '-' + 
                 df['FLOOR'].astype(str) + '-' + 
                 df['SPACEID'].astype(str))
```

### 3. Feature Engineering
```python
# Extract Wi-Fi signal strength features
X = df[wap_cols]  # 520 Access Point RSSI values
y = df['location']  # Unique location identifier

# Encode location labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split with stratification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
```

### 4. Model Training
```python
from sklearn.ensemble import RandomForestClassifier

# Configure RandomForest for high-dimensional problem
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train the model
model.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy:.2%}")

# Detailed performance analysis
print(classification_report(y_test, y_pred))
```

### 6. Feature Importance Analysis
```python
import matplotlib.pyplot as plt

# Identify most important Access Points
importances = model.feature_importances_
top_20_indices = np.argsort(importances)[-20:]

# Visualize critical APs for localization
plt.figure(figsize=(12, 10))
plt.barh(range(20), importances[top_20_indices])
plt.yticks(range(20), [f'WAP{i+1:03d}' for i in top_20_indices])
plt.title('Top 20 Most Important Access Points for Localization')
plt.xlabel('Feature Importance')
plt.show()
```

### 7. Real-World Deployment
```python
# Function for real-time location prediction
def predict_location(rssi_readings):
    \"\"\"
    Predict indoor location from real-time RSSI readings
    
    Args:
        rssi_readings: Dict of AP_ID -> RSSI_value
    
    Returns:
        Predicted location string
    \"\"\"
    # Prepare feature vector
    feature_vector = np.full(520, -105)  # Default to no signal
    
    for ap_id, rssi in rssi_readings.items():
        if ap_id in wap_mapping:
            feature_vector[wap_mapping[ap_id]] = rssi
    
    # Predict location
    prediction = model.predict([feature_vector])[0]
    return le.inverse_transform([prediction])[0]
```

## Success Criteria
- **Primary Metric**: Classification accuracy > 85% for room-level prediction
- **Coverage**: Successfully predict locations across multiple buildings and floors
- **Robustness**: Handle missing Access Point data gracefully
- **Interpretability**: Identify critical Access Points for network planning
- **Speed**: Real-time prediction capability for mobile applications

## Next Steps & Extensions

### Technical Enhancements
1. **Advanced Algorithms**: Experiment with XGBoost, neural networks, or ensemble methods
2. **Feature Engineering**: Include temporal patterns, device-specific calibration, and signal quality metrics
3. **Dimensionality Reduction**: Apply PCA or feature selection to reduce computational requirements
4. **Calibration**: Implement device-specific RSSI calibration for improved accuracy

### Business Applications
1. **Mobile App Integration**: Develop SDKs for iOS/Android applications
2. **Real-Time Systems**: Build streaming prediction pipeline for live location services
3. **Multi-Building Scaling**: Extend to campus-wide or city-wide indoor positioning
4. **Hybrid Positioning**: Combine with BLE beacons, magnetometer, and accelerometer data

### Research Directions
1. **Transfer Learning**: Adapt models trained on one building to work in another
2. **Uncertainty Quantification**: Provide confidence intervals for location predictions
3. **Privacy Preservation**: Implement federated learning approaches for sensitive locations
4. **Energy Optimization**: Balance prediction accuracy with mobile device battery consumption

## Files in this Project
- `README.md` - Project documentation and implementation guide
- `indoor_localization_wifi_rssi.ipynb` - Complete Jupyter notebook implementation
- `requirements.txt` - Python package dependencies

## Key Insights
- Wi-Fi RSSI fingerprints provide highly distinctive signatures for indoor locations
- RandomForest effectively handles the high-dimensional nature of 520 Access Point features
- Feature importance analysis reveals critical Access Points for network infrastructure planning
- Proper handling of missing signals (no-signal values) significantly improves model performance
- The approach scales well to hundreds of unique indoor locations with high accuracy