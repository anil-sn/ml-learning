# Project 23: Predicting Latency Jitter for a Network Path

## Objective
Build a regression model that can predict the latency (in milliseconds) of a network path based on characteristics like distance, time of day, and current traffic load to enable proactive network performance management.

## Business Value
- **Proactive Monitoring**: Predict network performance issues before they impact users
- **Capacity Planning**: Make data-driven decisions about network infrastructure investments
- **SLA Management**: Ensure service level agreements are met through predictive analytics
- **Route Optimization**: Select optimal network paths based on predicted performance
- **Real-Time Decision Making**: Enable automatic failover and traffic rerouting based on latency predictions

## Core Libraries
- **pandas & numpy**: Data manipulation and numerical operations
- **xgboost**: Gradient boosting framework for high-performance regression
- **scikit-learn**: Model evaluation metrics and data splitting utilities
- **matplotlib & seaborn**: Data visualization and performance analysis

## Dataset
**Source**: Synthetically Generated Network Monitoring Data
- **Size**: 5,000 network path measurements
- **Features**: Distance, hour of day, congestion factors, traffic spikes
- **Target**: Network latency in milliseconds
- **Scope**: Realistic simulation of WAN performance characteristics
- **Quality**: Physics-based modeling with appropriate noise and variability

## Step-by-Step Guide

### 1. Environment Setup
```python
# Install required packages
pip install pandas numpy xgboost scikit-learn matplotlib seaborn

# All data is synthetically generated - no external APIs needed
```

### 2. Synthetic Data Generation
```python
import pandas as pd
import numpy as np

# Generate realistic network latency dataset
def generate_latency_data(num_samples=5000):
    data = []
    
    for _ in range(num_samples):
        distance_km = np.random.randint(50, 5000)
        hour_of_day = np.random.randint(0, 24)
        
        # Business hours congestion (9am to 5pm)
        if 9 <= hour_of_day <= 17:
            congestion_factor = np.random.uniform(1.2, 2.5)
        else:
            congestion_factor = np.random.uniform(0.8, 1.2)
            
        # Random traffic spikes
        traffic_spike = np.random.choice([0, 1], p=[0.9, 0.1]) * np.random.uniform(5, 15)
        
        # Physics-based latency calculation
        base_latency = 5  # Local processing time
        distance_latency = distance_km * 0.05  # Speed of light effect
        congestion_latency = hour_of_day * congestion_factor
        random_noise = np.random.normal(0, 5)  # Jitter simulation
        
        latency = base_latency + distance_latency + congestion_latency + traffic_spike + random_noise
        latency = max(5, latency)  # Ensure realistic minimum
        
        data.append([distance_km, hour_of_day, congestion_factor, traffic_spike, latency])
    
    return pd.DataFrame(data, columns=['distance_km', 'hour_of_day', 'congestion_factor', 'traffic_spike', 'latency_ms'])
```

### 3. Exploratory Data Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize feature relationships
sns.pairplot(df, x_vars=['distance_km', 'hour_of_day', 'congestion_factor'], 
             y_vars=['latency_ms'], height=4, aspect=1)
plt.suptitle('Latency vs. Key Features')
plt.show()

# Correlation analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

### 4. Data Preparation and Splitting
```python
from sklearn.model_selection import train_test_split

# Prepare features and target
feature_cols = ['distance_km', 'hour_of_day', 'congestion_factor', 'traffic_spike']
X = df[feature_cols]
y = df['latency_ms']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5. Model Training with XGBoost
```python
import xgboost as xgb

# Configure XGBoost Regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression objective
    n_estimators=100,              # Number of boosting rounds
    learning_rate=0.1,             # Step size shrinkage
    max_depth=6,                   # Maximum tree depth
    random_state=42,
    n_jobs=-1                      # Use all CPU cores
)

# Train the model
model.fit(X_train, y_train)
```

### 6. Model Evaluation
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} ms")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} ms")
print(f"R-squared (R²): {r2:.2%}")
```

### 7. Visualization and Analysis
```python
# Actual vs. Predicted scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.title('Model Performance: Actual vs. Predicted')
plt.legend()
plt.show()

# Feature importance analysis
xgb.plot_importance(model, height=0.8)
plt.title('Feature Importance in Latency Prediction')
plt.show()
```

### 8. Real-Time Prediction Function
```python
def predict_network_latency(distance_km, hour_of_day, congestion_factor, traffic_spike):
    """
    Predict network latency for given conditions
    
    Args:
        distance_km: Physical distance of network path
        hour_of_day: Current hour (0-23)
        congestion_factor: Network congestion multiplier
        traffic_spike: Presence of traffic spike (0 or spike value)
    
    Returns:
        Predicted latency in milliseconds
    """
    features = np.array([[distance_km, hour_of_day, congestion_factor, traffic_spike]])
    prediction = model.predict(features)[0]
    return round(prediction, 2)

# Example usage
predicted_latency = predict_network_latency(
    distance_km=1500,
    hour_of_day=14,  # 2 PM
    congestion_factor=1.8,
    traffic_spike=0
)
print(f"Predicted latency: {predicted_latency} ms")
```

## Success Criteria
- **Primary Metric**: R-squared > 0.85 for variance explanation
- **Accuracy**: Mean Absolute Error < 10ms for practical network management
- **Robustness**: Model handles various network conditions and distances
- **Speed**: Real-time prediction capability for operational use
- **Interpretability**: Clear feature importance for network engineering insights

## Next Steps & Extensions

### Technical Enhancements
1. **Time Series Analysis**: Add temporal patterns and seasonal effects
2. **Advanced Features**: Include network topology, link utilization, and protocol overhead
3. **Ensemble Methods**: Combine XGBoost with other algorithms for improved accuracy
4. **Real-Time Learning**: Implement online learning for continuous model adaptation

### Business Applications
1. **NOC Dashboard**: Real-time latency prediction for network operations centers
2. **Traffic Engineering**: Optimize routing decisions based on predicted performance
3. **SLA Monitoring**: Proactive alerts before performance degrades below thresholds
4. **Capacity Planning**: Predict when network upgrades will be needed

### Advanced Analytics
1. **Confidence Intervals**: Provide prediction uncertainty for better decision making
2. **Anomaly Detection**: Identify unusual latency patterns that may indicate issues
3. **What-If Analysis**: Simulate different network scenarios for planning purposes
4. **Multi-Path Optimization**: Extend to predict performance of alternative routes

## Files in this Project
- `README.md` - Project documentation and implementation guide
- `latency_jitter_prediction.ipynb` - Complete Jupyter notebook implementation
- `requirements.txt` - Python package dependencies

## Key Insights
- Distance is the dominant factor in network latency prediction due to speed of light limitations
- Business hours significantly impact network performance through congestion patterns
- XGBoost effectively captures non-linear relationships between network variables
- Feature importance analysis provides actionable insights for network engineering decisions
- The model enables "what-if" analysis for network planning and optimization scenarios

## Physics-Based Modeling
- **Base Latency**: Local processing and switching delays
- **Distance Effect**: Speed of light propagation delays (≈5ms per 1000km)
- **Congestion Modeling**: Business hours traffic patterns with realistic multipliers
- **Traffic Spikes**: Random events that simulate network anomalies
- **Jitter Simulation**: Gaussian noise to represent natural network variability