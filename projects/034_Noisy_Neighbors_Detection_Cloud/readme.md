# Project 034: Detecting Noisy Neighbors in a Multi-tenant Cloud Environment

## Objective
Build an unsupervised anomaly detection model using Isolation Forest to identify "noisy neighbors" (high-resource-consuming tenants) in a shared cloud environment by analyzing network traffic patterns and flagging outliers that degrade performance for other tenants.

## Business Value
- **Performance Isolation**: Proactively identify tenants causing performance degradation before they impact other customers
- **SLA Protection**: Prevent noisy neighbors from violating service level agreements of co-located tenants
- **Resource Management**: Enable targeted resource throttling, migration, or workload balancing
- **Cost Optimization**: Optimize resource allocation and prevent over-provisioning due to performance issues
- **Customer Experience**: Maintain consistent performance and reliability across multi-tenant environments

## Core Libraries
- **scikit-learn**: Isolation Forest for unsupervised anomaly detection and data preprocessing
- **pandas**: Multi-tenant traffic data manipulation and time-series analysis
- **numpy**: Numerical computations and statistical operations
- **matplotlib/seaborn**: Traffic pattern visualization and anomaly detection results
- **time**: Performance measurement and monitoring

## Dataset
- **Source**: Synthetically Generated (realistic multi-tenant network traffic patterns)
- **Size**: 20,000 samples (20 tenants × 1,000 time steps)
- **Features**: Packets per second, bytes per second, average packet size, network utilization
- **Anomalies**: Primary and secondary noisy neighbor events with varying intensity and duration
- **Type**: Unsupervised anomaly detection with ground truth for evaluation

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv noisy_neighbors_env
source noisy_neighbors_env/bin/activate  # On Windows: noisy_neighbors_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Multi-tenant Traffic Data Generation
```python
# Generate realistic multi-tenant network traffic patterns
import pandas as pd
import numpy as np
import random

# Simulation parameters
num_tenants = 20
time_steps = 1000
tenants = [f'tenant_{i+1}' for i in range(num_tenants)]
noisy_neighbor_tenant = 'tenant_5'
secondary_noisy_tenant = 'tenant_15'

data = []
for t in range(time_steps):
    for tenant in tenants:
        is_noisy = False
        
        # Define normal behavior patterns with tenant-specific baselines
        if 'tenant_1' in tenant or 'tenant_2' in tenant:
            base_pps = max(0, np.random.normal(2000, 500))  # Low activity
            base_bps = base_pps * np.random.normal(250, 30)
        elif 'tenant_19' in tenant or 'tenant_20' in tenant:
            base_pps = max(0, np.random.normal(8000, 1200))  # High activity
            base_bps = base_pps * np.random.normal(400, 60)
        else:
            base_pps = max(0, np.random.normal(5000, 1000))  # Normal activity
            base_bps = base_pps * np.random.normal(300, 50)
        
        # Add time-based patterns (daily cycles)
        time_factor = 1 + 0.3 * np.sin(2 * np.pi * t / 100)
        base_pps *= time_factor
        base_bps *= time_factor
        
        # Simulate noisy neighbor events
        if tenant == noisy_neighbor_tenant and 400 <= t < 600:
            base_pps *= np.random.uniform(5, 10)  # 5-10x spike
            base_bps *= np.random.uniform(5, 10)
            is_noisy = True
        
        if tenant == secondary_noisy_tenant and 750 <= t < 800:
            base_pps *= np.random.uniform(8, 15)  # Intense spike
            base_bps *= np.random.uniform(8, 15)
            is_noisy = True
        
        # Calculate derived metrics
        avg_packet_size = base_bps / max(base_pps, 1)
        network_utilization = min(base_bps / 1000000, 100)
        
        data.append([t, tenant, base_pps, base_bps, avg_packet_size, 
                    network_utilization, is_noisy])

df = pd.DataFrame(data, columns=['timestamp', 'tenant_id', 'packets_per_second', 
                                'bytes_per_second', 'avg_packet_size', 
                                'network_utilization', 'is_truly_noisy'])
```

### 3. Data Exploration and Pattern Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize traffic patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Time series visualization
pivot_data = df.pivot(index='timestamp', columns='tenant_id', values='packets_per_second')
axes[0,0].plot(pivot_data.index, pivot_data[noisy_neighbor_tenant], 
               color='red', linewidth=2, label='Primary Noisy Neighbor')
axes[0,0].plot(pivot_data.index, pivot_data[secondary_noisy_tenant], 
               color='orange', linewidth=2, label='Secondary Noisy Neighbor')
axes[0,0].set_title('Packets per Second Over Time')
axes[0,0].legend()

# Distribution comparison
normal_data = df[df['is_truly_noisy'] == False]
noisy_data = df[df['is_truly_noisy'] == True]

axes[0,1].hist(normal_data['packets_per_second'], bins=50, alpha=0.7, 
               label='Normal', density=True)
axes[0,1].hist(noisy_data['packets_per_second'], bins=30, alpha=0.7, 
               label='Noisy Neighbor', density=True)
axes[0,1].set_title('Traffic Distribution Comparison')
axes[0,1].legend()

# Traffic correlation analysis
axes[1,0].scatter(normal_data['packets_per_second'], normal_data['bytes_per_second'], 
                  alpha=0.6, label='Normal', s=20)
axes[1,0].scatter(noisy_data['packets_per_second'], noisy_data['bytes_per_second'], 
                  alpha=0.8, label='Noisy Neighbor', s=30, color='red')
axes[1,0].set_title('Packets vs Bytes Correlation')
axes[1,0].legend()

plt.tight_layout()
plt.show()

# Tenant behavior summary
tenant_summary = df.groupby('tenant_id').agg({
    'packets_per_second': ['mean', 'max', 'std'],
    'is_truly_noisy': 'sum'
}).round(2)
print("Tenant Behavior Summary:")
print(tenant_summary.head(10))
```

### 4. Feature Engineering and Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler

# Prepare features for anomaly detection
feature_cols = ['packets_per_second', 'bytes_per_second', 'avg_packet_size', 'network_utilization']
X = df[feature_cols].copy()

# Handle any NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store ground truth labels for evaluation
y_true = df['is_truly_noisy'].values

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Ground truth anomalies: {np.sum(y_true)} ({np.mean(y_true)*100:.2f}%)")
```

### 5. Isolation Forest Model Training
```python
from sklearn.ensemble import IsolationForest

# Calculate expected contamination rate
expected_contamination = np.mean(y_true)
contamination_rate = 0.015  # Slightly conservative estimate

# Configure and train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)

print(f"Training Isolation Forest with {contamination_rate*100:.1f}% expected contamination...")
model.fit(X_scaled)

# Make predictions and calculate anomaly scores
y_pred_raw = model.predict(X_scaled)
y_pred = (y_pred_raw == -1).astype(int)  # Convert to binary
anomaly_scores = model.decision_function(X_scaled)

print(f"Predicted anomalies: {np.sum(y_pred)} ({np.mean(y_pred)*100:.2f}%)")
print(f"Actual anomalies: {np.sum(y_true)} ({np.mean(y_true)*100:.2f}%)")
```

### 6. Model Evaluation and Performance Analysis
```python
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Calculate performance metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average='binary', pos_label=1
)

print(f"Performance Metrics:")
print(f"• Precision: {precision:.3f}")
print(f"• Recall: {recall:.3f}")
print(f"• F1-Score: {f1:.3f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Noisy Neighbor']))

# Confusion matrix analysis
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)

print(f"\nAdditional Metrics:")
print(f"• Accuracy: {accuracy:.3f}")
print(f"• Specificity: {specificity:.3f}")
print(f"• False Positive Rate: {false_positive_rate:.3f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Noisy Neighbor'], 
            yticklabels=['Normal', 'Noisy Neighbor'])
plt.title('Confusion Matrix for Noisy Neighbor Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### 7. Tenant-wise Analysis and Investigation
```python
# Analyze detection performance by tenant
df_analysis = df.copy()
df_analysis['predicted_anomaly'] = y_pred
df_analysis['anomaly_score'] = anomaly_scores

tenant_analysis = []
for tenant in tenants:
    tenant_data = df_analysis[df_analysis['tenant_id'] == tenant]
    
    true_anomalies = tenant_data['is_truly_noisy'].sum()
    detected_anomalies = tenant_data['predicted_anomaly'].sum()
    
    if true_anomalies > 0:
        true_positives = ((tenant_data['is_truly_noisy'] == 1) & 
                         (tenant_data['predicted_anomaly'] == 1)).sum()
        tenant_recall = true_positives / true_anomalies
        tenant_precision = true_positives / detected_anomalies if detected_anomalies > 0 else 0
    else:
        tenant_recall = 0
        tenant_precision = 0 if detected_anomalies == 0 else np.nan
    
    tenant_analysis.append({
        'tenant_id': tenant,
        'true_anomalies': true_anomalies,
        'detected_anomalies': detected_anomalies,
        'recall': tenant_recall,
        'precision': tenant_precision
    })

tenant_df = pd.DataFrame(tenant_analysis)
print("Tenant Detection Performance:")
print(tenant_df[tenant_df['true_anomalies'] > 0])  # Focus on noisy tenants

# Visualize anomaly timeline for noisy neighbors
noisy_tenant_data = df_analysis[df_analysis['tenant_id'] == noisy_neighbor_tenant]
plt.figure(figsize=(14, 6))
plt.plot(noisy_tenant_data['timestamp'], noisy_tenant_data['packets_per_second'], 
         color='blue', alpha=0.7, label='Traffic')

# Highlight detections
true_anomalies = noisy_tenant_data[noisy_tenant_data['is_truly_noisy'] == 1]
plt.scatter(true_anomalies['timestamp'], true_anomalies['packets_per_second'], 
           color='red', s=50, label='True Anomalies', zorder=5)

detected_anomalies = noisy_tenant_data[noisy_tenant_data['predicted_anomaly'] == 1]
plt.scatter(detected_anomalies['timestamp'], detected_anomalies['packets_per_second'], 
           color='orange', s=30, marker='x', label='Detected Anomalies', zorder=5)

plt.title(f'Anomaly Detection Timeline for {noisy_neighbor_tenant}')
plt.xlabel('Time Step')
plt.ylabel('Packets per Second')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Success Criteria
- **High Recall (>80%)**: Detect most noisy neighbor events to prevent performance degradation
- **Balanced Precision (>70%)**: Minimize false alarms to avoid alert fatigue
- **Low False Positive Rate (<10%)**: Maintain operational efficiency with reliable alerts
- **Tenant Isolation**: Successfully identify specific problematic tenants

## Next Steps & Extensions
1. **Real-time Deployment**: Integrate with cloud monitoring platforms for live anomaly detection
2. **Multi-dimensional Analysis**: Include CPU, memory, disk I/O, and network bandwidth metrics
3. **Automated Response**: Implement automatic resource throttling or tenant migration
4. **Adaptive Thresholds**: Use dynamic contamination rates based on historical patterns
5. **Root Cause Analysis**: Identify specific applications or processes causing noisy behavior
6. **Predictive Alerts**: Forecast potential noisy neighbor events before they impact performance

## Files Structure
```
034_Noisy_Neighbors_Detection_Cloud/
├── readme.md
├── noisy_neighbors_detection_cloud.ipynb
├── requirements.txt
└── data/
    └── (Generated multi-tenant traffic data)
```

## Running the Project
1. Install required dependencies from requirements.txt
2. Execute the Jupyter notebook step by step
3. Analyze multi-tenant traffic patterns and baseline behavior
4. Train Isolation Forest model for unsupervised anomaly detection
5. Evaluate detection performance and investigate tenant-specific results
6. Deploy model for real-time noisy neighbor monitoring

This project demonstrates how unsupervised machine learning can solve critical multi-tenancy challenges in cloud environments, providing automatic detection of performance-impacting tenants while maintaining high operational efficiency and minimal false alarms.