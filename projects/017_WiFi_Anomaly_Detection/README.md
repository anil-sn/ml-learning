# Project 17: Wi-Fi Anomaly Detection (Deauthentication Flood)

## Objective

Build an unsupervised anomaly detection system that can identify a deauthentication flood attack in real-time by analyzing the rate and type of Wi-Fi management frames. This project demonstrates how machine learning can detect wireless network attacks without requiring labeled attack data.

## Business Value

**For Network Engineers transitioning to ML/AI:**
- **Network Security Enhancement**: Learn to detect sophisticated Wi-Fi attacks that bypass traditional signature-based systems
- **Proactive Threat Detection**: Identify attacks in real-time before they can fully compromise network security
- **Cost Reduction**: Reduce manual security monitoring overhead through automated anomaly detection
- **Compliance**: Meet security requirements for wireless network monitoring and threat detection
- **Scalability**: Deploy across multiple access points for organization-wide wireless security

**Real-World Applications:**
- Enterprise wireless security monitoring
- Public Wi-Fi hotspot protection
- Critical infrastructure wireless network security
- IoT device protection in wireless networks
- Regulatory compliance for wireless security standards

## Core Libraries

- **pandas & numpy**: Data manipulation and numerical operations for Wi-Fi frame analysis
- **scikit-learn**: Isolation Forest algorithm for unsupervised anomaly detection
- **matplotlib & seaborn**: Visualization of attack patterns and detection results

## Dataset

**Synthetically Generated Wi-Fi Management Frame Data**
- **Source**: Custom simulation of Wi-Fi management frames
- **Size**: 120 seconds of simulated Wi-Fi traffic data
- **Features**: Frame types, timestamps, deauthentication ratios
- **Attack Simulation**: Deauthentication flood attack injected at specific time intervals

**Key Features:**
- `total_frames`: Number of frames per time window
- `deauth_ratio`: Ratio of deauthentication frames to total frames
- `Beacon`: Number of beacon frames (access point advertisements)
- `Probe Request`: Number of probe request frames (device scanning)

## Project Structure

```
017_WiFi_Anomaly_Detection/
├── README.md
├── wifi_anomaly_detection.ipynb
└── requirements.txt
```

## Step-by-Step Guide

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv wifi_anomaly_env
source wifi_anomaly_env/bin/activate  # On Windows: wifi_anomaly_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Synthetic Data Generation

```python
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Simulation parameters
total_duration_seconds = 120
attack_start_time = 80
attack_duration = 20
normal_frames_per_second = 50
attack_frames_per_second = 500

# Normal vs attack frame distributions
normal_subtypes = ['Beacon', 'Probe Request', 'Probe Response', 'Association Request', 'Deauthentication']
normal_subtype_weights = [0.5, 0.2, 0.2, 0.09, 0.01]  # Deauth very rare in normal traffic
```

### 3. Feature Engineering

```python
# Aggregate raw frames into time windows
df_agg = df_raw.groupby('timestamp')['subtype'].value_counts().unstack(fill_value=0)

# Calculate critical features
df_agg['total_frames'] = df_agg.sum(axis=1)
df_agg['deauth_ratio'] = df_agg['Deauthentication'] / df_agg['total_frames']

# Select features for anomaly detection
features = ['total_frames', 'deauth_ratio', 'Beacon', 'Probe Request']
```

### 4. Unsupervised Model Training

```python
# Train only on benign data (before attack)
X_train_benign = df_model[df_model.index < attack_start_time]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)

# Train Isolation Forest
model = IsolationForest(contamination='auto', random_state=42)
model.fit(X_train_scaled)
```

### 5. Anomaly Detection

```python
# Detect anomalies across entire timeline
X_all_scaled = scaler.transform(df_model)
df_model['anomaly_score'] = model.decision_function(X_all_scaled)
df_model['is_anomaly'] = model.predict(X_all_scaled)  # -1 for anomaly, 1 for normal

# Evaluate performance
ground_truth = np.where((df_model.index >= attack_start_time) & 
                       (df_model.index < attack_start_time + attack_duration), -1, 1)
accuracy = np.mean(df_model['is_anomaly'] == ground_truth)
```

### 6. Visualization and Analysis

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot deauthentication ratio over time
ax1.plot(df_model.index, df_model['deauth_ratio'], label='Deauthentication Ratio', color='orange')
ax1.axvspan(attack_start_time, attack_start_time + attack_duration, color='red', alpha=0.2, label='Attack Period')

# Plot anomaly scores
ax2.plot(df_model.index, df_model['anomaly_score'], label='Anomaly Score', color='blue')
ax2.fill_between(df_model.index, plt.ylim()[0], plt.ylim()[1], 
                 where=df_model['is_anomaly']==-1, facecolor='red', alpha=0.3, label='Detected Anomaly')
```

## Success Criteria

1. **Detection Accuracy**: Achieve >90% accuracy in distinguishing between normal and attack periods
2. **Real-time Detection**: Model responds within 1-2 seconds of attack onset
3. **Low False Positives**: <5% false positive rate during normal operation
4. **Feature Importance**: Deauthentication ratio should be the primary detection feature
5. **Scalability**: Model can process high-frequency Wi-Fi frame data efficiently

## Key Insights

**Technical Learnings:**
- **Unsupervised Detection**: Learn patterns of normal behavior without labeled attack data
- **Time-Series Analysis**: Process sequential Wi-Fi management frames effectively
- **Feature Engineering**: Create meaningful metrics from raw network protocol data
- **Real-time Processing**: Design system for continuous monitoring and immediate response

**Network Engineering Applications:**
- **Baseline Establishment**: Understand normal Wi-Fi traffic patterns for your network
- **Attack Signatures**: Recognize behavioral patterns of deauthentication attacks
- **Response Automation**: Trigger automated responses when attacks are detected
- **Performance Monitoring**: Track wireless network health and security continuously

## Next Steps & Extensions

### Immediate Improvements
1. **Multi-Attack Detection**: Extend to detect other Wi-Fi attacks (evil twin, rogue AP, WPS attacks)
2. **Adaptive Learning**: Implement online learning to adapt to changing network patterns
3. **Real-time Integration**: Connect to live Wi-Fi packet capture tools (tcpdump, Wireshark)
4. **Alert System**: Add email/SMS notifications for detected attacks

### Advanced Features
1. **Deep Learning Models**: Implement LSTM networks for temporal pattern recognition
2. **Multi-AP Correlation**: Correlate attacks across multiple access points
3. **Geolocation Integration**: Map attack origins using signal strength and AP positioning
4. **Client Behavior Analysis**: Track individual device patterns for targeted attack detection

### Production Deployment
1. **Edge Computing**: Deploy models on wireless controllers or dedicated security appliances
2. **SIEM Integration**: Connect with Security Information and Event Management systems
3. **Network Automation**: Integrate with network orchestration tools for automated response
4. **Compliance Reporting**: Generate security reports for audit and compliance requirements

### Research Extensions
1. **Adversarial Attack Resilience**: Test model robustness against sophisticated evasion attempts
2. **Cross-Protocol Analysis**: Combine with other network protocols for comprehensive threat detection
3. **Federated Learning**: Share threat intelligence across organizations while preserving privacy
4. **Explainable AI**: Develop interpretable models for security analyst understanding

This project provides a foundation for wireless network security using machine learning, demonstrating how network engineers can leverage AI to enhance their security monitoring capabilities without requiring extensive cybersecurity expertise.