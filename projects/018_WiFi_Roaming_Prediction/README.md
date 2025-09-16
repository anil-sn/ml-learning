# Project 18: Predicting Wi-Fi Roaming Events for Mobile Devices

## Objective

Build a machine learning model that predicts if a wireless client will roam to a new Access Point within the next few seconds, based on the changing signal strength (RSSI) from surrounding APs. This predictive capability enables proactive network optimization and seamless handover experiences.

## Business Value

**For Network Engineers transitioning to ML/AI:**
- **Proactive Network Management**: Predict and prepare for client roaming before it happens
- **Quality of Experience**: Improve voice and video call quality by reducing handover latency
- **Network Optimization**: Optimize AP placement and power settings based on roaming patterns
- **Resource Planning**: Anticipate capacity needs across different access points
- **Troubleshooting**: Identify problematic areas where clients frequently struggle to roam

**Real-World Applications:**
- Enterprise Wi-Fi network optimization
- Fast roaming (802.11r) implementation
- Hospital and industrial IoT device connectivity
- Campus and large facility Wi-Fi management
- Venue-based customer experience enhancement

## Core Libraries

- **pandas & numpy**: Data manipulation and time-series analysis of RSSI values
- **scikit-learn**: Random Forest classifier for roaming prediction
- **matplotlib & seaborn**: Visualization of signal patterns and prediction results

## Dataset

**Synthetically Generated Wi-Fi Roaming Data**
- **Source**: Simulated mobile device movement through multi-AP environment
- **Size**: 300 time steps representing device mobility patterns
- **Features**: RSSI values from multiple APs, rate of change (delta) features
- **Target**: Binary prediction of roaming events within prediction window

**Key Features:**
- `rssi_AP_1`, `rssi_AP_2`, `rssi_AP_3`: Signal strength from each access point
- `rssi_AP_X_delta`: Rate of change in signal strength for each AP
- `will_roam_soon`: Target variable indicating if roaming will occur soon

## Project Structure

```
018_WiFi_Roaming_Prediction/
├── README.md
├── wifi_roaming_prediction.ipynb
└── requirements.txt
```

## Step-by-Step Guide

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv wifi_roaming_env
source wifi_roaming_env/bin/activate  # On Windows: wifi_roaming_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Synthetic Data Generation

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulation parameters
time_steps = 300  # 300 seconds of data
num_aps = 3
aps = [f'AP_{i+1}' for i in range(num_aps)]

# Simulate mobile device walking through building
for t in range(time_steps):
    # Simulate RSSI values (in dBm, higher is better)
    rssi_ap1 = -30 - (t * 0.2) + np.random.normal(0, 2)
    rssi_ap2 = -90 + (abs(t - 150) * -0.4) + 50 + np.random.normal(0, 2)
    rssi_ap3 = -90 + (abs(t - 250) * -0.4) + 40 + np.random.normal(0, 2)
```

### 3. Feature Engineering

```python
# Create predictive target variable
prediction_window = 5  # seconds
df['will_roam_soon'] = 0

# Shift connected_ap to see future connections
df['next_ap'] = df['connected_ap'].shift(-prediction_window)

# Identify roaming events
roam_indices = df[df['connected_ap'] != df['next_ap']].index

# Set flag for time steps leading up to roam
for idx in roam_indices:
    df.loc[max(0, idx - prediction_window):idx, 'will_roam_soon'] = 1

# Engineer rate-of-change features
for ap in aps:
    df[f'rssi_{ap}_delta'] = df[f'rssi_{ap}'].diff().fillna(0)
```

### 4. Model Training

```python
# Prepare features and target
feature_cols = [col for col in df.columns if 'rssi' in col]
X = df[feature_cols]
y = df['will_roam_soon']

# Time-series split for realistic evaluation
split_point = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Train Random Forest with balanced classes
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
```

### 5. Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
y_pred = model.predict(X_test)

# Focus on recall for roaming events
print(classification_report(y_test, y_pred, target_names=['Will Not Roam', 'Will Roam']))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
```

### 6. Visualization and Analysis

```python
import matplotlib.pyplot as plt

# Plot RSSI patterns and predictions
plt.figure(figsize=(16, 8))

# Plot RSSI values from all APs
for ap in aps:
    plt.plot(df_test_results['time'], df_test_results[f'rssi_{ap}'], label=f'RSSI {ap}')

# Highlight prediction windows
plt.fill_between(df_test_results['time'], -90, -20, 
                 where=df_test_results['will_roam_soon']==1,
                 facecolor='orange', alpha=0.5, label='Actual Pre-Roam Window')

# Show model predictions
plt.scatter(df_test_results['time'][df_test_results['prediction']==1],
           df_test_results['rssi_AP_3'][df_test_results['prediction']==1] + 2,
           color='red', marker='v', s=50, label='Predicted Roam')
```

## Success Criteria

1. **High Recall**: Achieve >80% recall for roaming events (minimize missed roaming predictions)
2. **Acceptable Precision**: Maintain >60% precision to avoid excessive false alarms
3. **Timing Accuracy**: Predict roaming events 3-7 seconds in advance
4. **Feature Importance**: Signal strength delta should be among top features
5. **Real-time Capability**: Model prediction time <100ms for production deployment

## Key Insights

**Technical Learnings:**
- **Time-Series Prediction**: Handle temporal patterns in wireless signal data
- **Feature Engineering**: Rate of change features often more predictive than absolute values
- **Class Imbalance**: Roaming events are rare, requiring balanced training approaches
- **Temporal Validation**: Use time-aware splitting for realistic performance assessment

**Network Engineering Applications:**
- **Fast Roaming**: Pre-authenticate clients with likely destination APs
- **Load Balancing**: Proactively manage client distribution across APs
- **Coverage Optimization**: Identify areas where roaming is problematic
- **Capacity Planning**: Predict load patterns for different access points

## Next Steps & Extensions

### Immediate Improvements
1. **Multi-Device Tracking**: Extend to predict roaming for multiple concurrent clients
2. **Environmental Factors**: Include additional features like device type, application usage
3. **Real-time Integration**: Connect to live Wi-Fi controller APIs for real data
4. **Adaptive Windows**: Dynamic prediction windows based on device mobility patterns

### Advanced Features
1. **Deep Learning Models**: Implement LSTM/GRU networks for complex temporal patterns
2. **Location-Aware Predictions**: Incorporate floor plans and physical layout information
3. **Quality Metrics**: Predict not just roaming, but roaming success probability
4. **Multi-Band Prediction**: Handle 2.4GHz/5GHz/6GHz band steering decisions

### Production Deployment
1. **Controller Integration**: Deploy models on wireless LAN controllers
2. **Edge Computing**: Implement lightweight models on access points themselves
3. **API Development**: Create REST APIs for network management system integration
4. **Monitoring Dashboard**: Real-time visualization of roaming predictions and patterns

### Research Extensions
1. **Federated Learning**: Train models across multiple sites while preserving privacy
2. **Reinforcement Learning**: Optimize AP power and channel assignments based on roaming patterns
3. **Cross-Protocol Analysis**: Combine Wi-Fi with cellular handover prediction models
4. **Anomaly Detection**: Identify unusual roaming patterns that may indicate security issues

### Industry-Specific Applications
1. **Healthcare**: Ensure critical medical device connectivity during roaming
2. **Manufacturing**: Maintain industrial IoT device connections on factory floors
3. **Education**: Optimize campus-wide Wi-Fi for student and faculty mobility
4. **Retail**: Enhance customer Wi-Fi experience in large shopping centers

## Implementation Considerations

### Performance Optimization
- **Feature Selection**: Use only the most predictive RSSI and delta features
- **Model Complexity**: Balance accuracy with inference speed for real-time deployment
- **Memory Usage**: Optimize for deployment on resource-constrained network equipment
- **Batch Processing**: Handle multiple client predictions efficiently

### Network Integration
- **SNMP Integration**: Pull RSSI data from existing network monitoring systems
- **Controller APIs**: Integrate with vendor-specific wireless controller platforms
- **Standards Compliance**: Ensure compatibility with 802.11 standards and amendments
- **Vendor Agnostic**: Design for compatibility across different Wi-Fi equipment vendors

This project demonstrates how predictive analytics can transform reactive network management into proactive optimization, providing network engineers with powerful tools to enhance wireless network performance and user experience.