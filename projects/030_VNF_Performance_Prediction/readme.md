# Project 030: Virtual Network Function (VNF) Performance Prediction

## Objective
Build a regression model that can predict the maximum achievable throughput (in Gbps) of a Virtual Network Function (VNF) based on its type, allocated resources (vCPUs, RAM), and configuration complexity.

## Business Value
- **Resource Optimization**: Right-size VNF deployments to prevent over-provisioning and reduce cloud costs
- **SLA Compliance**: Ensure adequate resources are allocated to meet performance requirements
- **Automated Orchestration**: Enable NFV orchestrators to make intelligent resource allocation decisions
- **Performance Planning**: Predict VNF performance before deployment for capacity planning
- **Cost Management**: Balance performance requirements with resource costs in NFV environments

## Core Libraries
- **xgboost**: XGBoost Regressor for capturing complex non-linear relationships between resources and performance
- **pandas**: Dataset manipulation and synthetic data generation
- **numpy**: Numerical computations and statistical operations
- **scikit-learn**: Model evaluation metrics and data splitting
- **matplotlib/seaborn**: Data visualization and performance analysis

## Dataset
- **Source**: Synthetically Generated (realistic VNF performance data)
- **Size**: 3,000 VNF performance samples
- **Features**: VNF type (Firewall, Router, LoadBalancer, IDS), vCPUs, RAM (GB), configuration complexity
- **Target**: Throughput performance in Gbps
- **Type**: Regression dataset with both categorical and numerical features

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv vnf_performance_env
source vnf_performance_env/bin/activate  # On Windows: vnf_performance_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### 2. Data Generation and Preparation
```python
# Generate synthetic VNF performance dataset
import pandas as pd
import numpy as np
import random

# Create realistic VNF performance data
vnf_types = ['Firewall', 'Router', 'LoadBalancer', 'IDS']
data = []

for _ in range(3000):
    vnf_type = random.choice(vnf_types)
    vcpus = random.randint(2, 16)
    ram_gb = random.choice([4, 8, 16, 32])
    config_complexity = np.random.randint(10, 10000)  # Varies by VNF type
    
    # Calculate throughput based on resources and complexity
    base_throughput = (vcpus * 1.5) + (ram_gb * 0.2)
    complexity_penalty = np.log1p(config_complexity) * 0.5
    throughput_gbps = max(1, base_throughput - complexity_penalty + np.random.normal(0, 0.5))
    
    data.append([vnf_type, vcpus, ram_gb, config_complexity, throughput_gbps])

df = pd.DataFrame(data, columns=['vnf_type', 'vcpus', 'ram_gb', 'config_complexity', 'throughput_gbps'])
```

### 3. Feature Engineering
```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=['throughput_gbps'])
y = df['throughput_gbps']

# One-hot encode VNF type
X_encoded = pd.get_dummies(X, columns=['vnf_type'], drop_first=True)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
```

### 4. Model Training
```python
import xgboost as xgb

# Train XGBoost Regressor optimized for VNF performance prediction
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Mean Absolute Error: {mae:.3f} Gbps")
print(f"R-squared Score: {r2:.3f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
```

### 6. Performance Prediction Examples
```python
# Predict performance for different VNF configurations
example_configs = [
    {'vnf_type': 'Firewall', 'vcpus': 8, 'ram_gb': 16, 'config_complexity': 1000},
    {'vnf_type': 'Router', 'vcpus': 4, 'ram_gb': 8, 'config_complexity': 50},
    {'vnf_type': 'LoadBalancer', 'vcpus': 6, 'ram_gb': 16, 'config_complexity': 20}
]

# Make predictions for resource planning
for config in example_configs:
    predicted_throughput = model.predict([encoded_config])
    print(f"VNF: {config['vnf_type']}, Resources: {config['vcpus']} vCPUs, {config['ram_gb']} GB RAM")
    print(f"Predicted Throughput: {predicted_throughput[0]:.2f} Gbps")
```

## Success Criteria
- **High R² Score (>0.85)**: Model explains most variance in VNF performance
- **Low MAE (<1.0 Gbps)**: Predictions accurate within 1 Gbps for resource planning
- **Feature Importance**: vCPUs identified as primary performance driver
- **Realistic Complexity Impact**: Configuration complexity correctly penalizes performance

## Next Steps & Extensions
1. **Real-world Validation**: Integrate with actual VNF telemetry data from production environments
2. **Multi-dimensional Performance**: Predict latency, CPU utilization, and memory usage alongside throughput
3. **Dynamic Scaling**: Implement auto-scaling recommendations based on performance predictions
4. **Cost Optimization**: Combine with cloud pricing data for cost-performance optimization
5. **Vendor-specific Models**: Develop separate models for different VNF vendors and implementations
6. **Real-time Orchestration**: Deploy model as microservice for NFV orchestration platforms

## Files Structure
```
030_VNF_Performance_Prediction/
├── readme.md
├── vnf_performance_prediction.ipynb
├── requirements.txt
└── data/
    └── (Generated synthetic VNF performance data)
```

## Running the Project
1. Install required dependencies from requirements.txt
2. Execute the Jupyter notebook step by step
3. Analyze feature importance to understand VNF performance drivers
4. Use the trained model for VNF resource planning and optimization

This project demonstrates how machine learning can optimize NFV environments by accurately predicting VNF performance, enabling intelligent resource allocation and cost-effective network function virtualization.