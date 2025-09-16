# Project 028: Predicting Optimal MTU Size for a Network Path

## Objective
Build a regression model that can predict the optimal MTU (Maximum Transmission Unit) size for a given network path and application type, aiming to maximize throughput and minimize fragmentation.

## Business Value
- **Performance Optimization**: Maximize network throughput by selecting optimal packet sizes
- **Reduced Fragmentation**: Minimize packet fragmentation that causes performance degradation
- **Application-Aware Networking**: Tailor MTU settings to specific application requirements
- **Automated Optimization**: Remove manual MTU tuning and reduce network engineering effort
- **SLA Compliance**: Ensure optimal performance for latency-sensitive applications

## Core Libraries
- **scikit-learn**: Gradient Boosting Regressor for MTU prediction and model evaluation
- **pandas**: Dataset manipulation and feature engineering
- **numpy**: Numerical computations and synthetic data generation
- **matplotlib/seaborn**: Data visualization and model performance analysis

## Dataset
- **Source**: Synthetically Generated
- **Size**: 2000+ samples of network path characteristics and optimal MTU measurements
- **Features**: Application type, base latency, VPN presence, network path characteristics
- **Target**: Optimal MTU size (in bytes)
- **Type**: Regression dataset with realistic network performance relationships

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv mtu_prediction_env
source mtu_prediction_env/bin/activate  # On Windows: mtu_prediction_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Synthetic Data Generation
```python
import pandas as pd
import numpy as np
import random

# Define application types with different MTU requirements
application_types = ['VOIP', 'Video_Streaming', 'Bulk_Data_Transfer', 
                    'Web_Browsing', 'Database_Replication']

# Generate realistic network scenarios
data = []
for _ in range(2000):
    app_type = random.choice(application_types)
    base_latency_ms = np.random.uniform(5, 100)
    has_vpn_tunnel = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # Calculate optimal MTU based on application and network conditions
    optimal_mtu = calculate_optimal_mtu(app_type, has_vpn_tunnel, base_latency_ms)
    data.append([app_type, base_latency_ms, has_vpn_tunnel, optimal_mtu])
```

### 3. Feature Engineering
```python
from sklearn.preprocessing import LabelEncoder

# Prepare features for machine learning
X = df.drop(columns=['optimal_mtu'])
y = df['optimal_mtu']

# One-hot encode categorical application types
X_encoded = pd.get_dummies(X, columns=['application_type'], drop_first=True)
```

### 4. Model Training
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
from sklearn.metrics import mean_absolute_error, r2_score

# Evaluate model performance
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} bytes")
print(f"R-squared Score: {r2:.2%}")
```

### 6. Results Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot actual vs predicted MTU values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Optimal MTU (bytes)')
plt.ylabel('Predicted Optimal MTU (bytes)')
plt.title('MTU Prediction Accuracy')
plt.legend()
plt.show()

# Feature importance analysis
importances = model.feature_importances_
feature_names = X_train.columns
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

sns.barplot(data=feature_df, x='Importance', y='Feature')
plt.title('Feature Importance in MTU Prediction')
plt.show()
```

## Success Criteria
- **High Prediction Accuracy (R² > 85%)**: Accurate MTU recommendations for network optimization
- **Low Mean Absolute Error (<50 bytes)**: Precise MTU predictions within acceptable tolerance
- **Application Awareness**: Model should differentiate MTU needs across application types
- **Network Condition Sensitivity**: Account for VPN tunnels and latency impacts

## Next Steps & Extensions
1. **Real-world Integration**: Deploy with SDN controllers for dynamic MTU adjustment
2. **Path MTU Discovery**: Integrate with PMTU discovery protocols for validation
3. **Multi-hop Analysis**: Extend to complex network topologies with multiple hops
4. **Performance Monitoring**: Add feedback loop to continuously improve predictions
5. **Protocol-Specific**: Customize for different protocols (TCP, UDP, etc.)
6. **Cloud Integration**: Adapt for cloud networking environments and container networks

## Files Structure
```
028_Optimal_MTU_Size_Prediction/
├── README.md
├── mtu_size_prediction.ipynb
├── requirements.txt
└── models/
    └── (trained model artifacts)
```

## Running the Project
1. Execute the Jupyter notebook step by step
2. Review synthetic data generation logic
3. Analyze model performance and feature importance
4. Test predictions with different network scenarios

This project demonstrates how machine learning can optimize network performance by intelligently selecting MTU sizes based on application requirements and network conditions, leading to improved throughput and reduced fragmentation.