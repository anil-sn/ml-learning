# Project 032: Container Network Traffic Pattern Analysis

## Objective
Build a machine learning model that can classify the type of application running inside a container (WebApp, Database, Cache, MessageQueue, APIGateway) by analyzing the statistical features of its network traffic patterns.

## Business Value
- **Container Visibility**: Automatically identify application types without deep packet inspection or container introspection
- **Resource Optimization**: Right-size containers based on application-specific network behavior patterns
- **Security Monitoring**: Detect anomalous applications or configuration drift by monitoring network patterns
- **Network Planning**: Optimize network policies and resource allocation based on application traffic characteristics
- **Compliance**: Ensure containers match expected application profiles for regulatory requirements

## Core Libraries
- **scikit-learn**: RandomForestClassifier for pattern recognition and model evaluation metrics
- **pandas**: Network traffic data manipulation and statistical analysis
- **numpy**: Numerical computations and data generation
- **matplotlib/seaborn**: Network pattern visualization and classification results analysis
- **time**: Performance measurement and training time analysis

## Dataset
- **Source**: Synthetically Generated (realistic network flow data from containerized applications)
- **Size**: 5,000 network flow samples across 5 application types
- **Features**: Average packet size, server port, flow duration, client/server packets, total bytes, throughput
- **Applications**: WebApp, Database, Cache, MessageQueue, APIGateway
- **Type**: Multi-class classification with engineered network features

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv container_traffic_env
source container_traffic_env/bin/activate  # On Windows: container_traffic_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Data Generation and Network Profiles
```python
# Generate realistic container network traffic data
import pandas as pd
import numpy as np
import random

# Define application-specific network "personalities"
app_profiles = {
    'WebApp':       {'avg_pkt_size': 500,  'server_port': 443,  'flow_duration_ms': 500, 'client_server_ratio': 0.8},
    'Database':     {'avg_pkt_size': 1000, 'server_port': 5432, 'flow_duration_ms': 100, 'client_server_ratio': 0.5},
    'Cache':        {'avg_pkt_size': 150,  'server_port': 6379, 'flow_duration_ms': 20,  'client_server_ratio': 0.5},
    'MessageQueue': {'avg_pkt_size': 300,  'server_port': 5672, 'flow_duration_ms': 10000,'client_server_ratio': 0.5},
    'APIGateway':   {'avg_pkt_size': 800,  'server_port': 8080, 'flow_duration_ms': 200, 'client_server_ratio': 0.7}
}

# Generate network flows based on application profiles
data = []
for _ in range(5000):
    app_type = random.choice(['WebApp', 'Database', 'Cache', 'MessageQueue', 'APIGateway'])
    profile = app_profiles[app_type]
    
    # Generate features with realistic variations
    avg_pkt_size = max(50, np.random.normal(profile['avg_pkt_size'], 50))
    server_port = profile['server_port']
    flow_duration_ms = max(1, np.random.normal(profile['flow_duration_ms'], 100))
    client_server_ratio = np.clip(np.random.normal(profile['client_server_ratio'], 0.1), 0.1, 0.9)
    
    # Calculate derived features
    total_packets = np.random.randint(5, 100)
    client_packets = int(total_packets * client_server_ratio)
    server_packets = total_packets - client_packets
    total_bytes = avg_pkt_size * total_packets
    throughput_kbps = (total_bytes * 8) / (flow_duration_ms / 1000) / 1000
    
    data.append([avg_pkt_size, server_port, flow_duration_ms, client_packets, 
                server_packets, total_bytes, throughput_kbps, app_type])

df = pd.DataFrame(data, columns=['avg_pkt_size', 'server_port', 'flow_duration_ms', 
                                'client_packets', 'server_packets', 'total_bytes', 
                                'throughput_kbps', 'app_type'])
```

### 3. Data Exploration and Pattern Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize application network patterns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Analyze packet size patterns by application
sns.boxplot(data=df, x='app_type', y='avg_pkt_size', ax=axes[0,0])
axes[0,0].set_title('Average Packet Size by Application')

# Flow duration patterns
sns.boxplot(data=df, x='app_type', y='flow_duration_ms', ax=axes[0,1])
axes[0,1].set_yscale('log')
axes[0,1].set_title('Flow Duration by Application')

# Throughput characteristics
sns.boxplot(data=df, x='app_type', y='throughput_kbps', ax=axes[0,2])
axes[0,2].set_yscale('log')
axes[0,2].set_title('Throughput by Application')

# Client vs Server packet relationships
for app in df['app_type'].unique():
    app_data = df[df['app_type'] == app]
    axes[1,0].scatter(app_data['client_packets'], app_data['server_packets'], 
                     alpha=0.6, label=app)
axes[1,0].set_title('Client vs Server Packet Distribution')
axes[1,0].legend()

plt.tight_layout()
plt.show()

# Statistical analysis by application type
profile_stats = df.groupby('app_type').agg({
    'avg_pkt_size': ['mean', 'std'],
    'flow_duration_ms': ['mean', 'std'],
    'throughput_kbps': ['mean', 'std']
}).round(2)
print("Application Network Profiles:")
print(profile_stats)
```

### 4. Feature Engineering and Data Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features and target
X = df.drop(columns=['app_type'])
y = df['app_type']

# Encode application labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split to ensure balanced representation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Features: {list(X.columns)}")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### 5. Model Training with RandomForest
```python
from sklearn.ensemble import RandomForestClassifier
import time

# Train RandomForest optimized for network pattern classification
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Model trained in {training_time:.2f} seconds")
```

### 6. Model Evaluation and Performance Analysis
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix analysis
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Container Application Classification Confusion Matrix')
plt.ylabel('Actual Application')
plt.xlabel('Predicted Application')
plt.show()
```

### 7. Feature Importance Analysis
```python
# Analyze which network features are most predictive
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importance in Container Traffic Classification')
plt.xlabel('Importance')
plt.show()

print("Most Important Network Characteristics:")
print(feature_importance_df.round(4))
```

### 8. Application Prediction Examples
```python
# Test model with example network flows
example_flows = [
    # WebApp: Small packets, HTTPS, moderate duration
    {'avg_pkt_size': 480, 'server_port': 443, 'flow_duration_ms': 450, 
     'client_packets': 15, 'server_packets': 8, 'total_bytes': 11040, 'throughput_kbps': 196},
    
    # Database: Large packets, PostgreSQL, short duration
    {'avg_pkt_size': 980, 'server_port': 5432, 'flow_duration_ms': 95, 
     'client_packets': 8, 'server_packets': 12, 'total_bytes': 19600, 'throughput_kbps': 1651},
    
    # Cache: Small packets, Redis, very short duration
    {'avg_pkt_size': 140, 'server_port': 6379, 'flow_duration_ms': 18, 
     'client_packets': 5, 'server_packets': 5, 'total_bytes': 1400, 'throughput_kbps': 622}
]

examples_df = pd.DataFrame(example_flows)
predictions = model.predict(examples_df)
probabilities = model.predict_proba(examples_df)

for i, flow in enumerate(example_flows):
    predicted_app = le.classes_[predictions[i]]
    confidence = np.max(probabilities[i])
    print(f"Flow {i+1}: Predicted as {predicted_app} (confidence: {confidence:.3f})")
```

## Success Criteria
- **High Accuracy (>90%)**: Model correctly classifies application types from network patterns
- **Balanced Performance**: Good precision and recall across all application types
- **Feature Interpretability**: Network characteristics align with domain knowledge
- **Fast Training**: Model trains quickly for real-time deployment scenarios

## Next Steps & Extensions
1. **Real-time Integration**: Deploy with container orchestration platforms (Kubernetes, Docker Swarm)
2. **Deep Packet Inspection**: Enhance with application layer protocol analysis
3. **Anomaly Detection**: Flag containers with unexpected network behavior patterns
4. **Multi-cluster Analysis**: Extend classification across different container environments
5. **Performance Monitoring**: Baseline normal behavior for automated alerting
6. **Security Enhancement**: Detect potential malicious containers based on traffic anomalies

## Files Structure
```
032_Container_Network_Traffic_Analysis/
├── readme.md
├── container_network_traffic_analysis.ipynb
├── requirements.txt
└── data/
    └── (Generated container network flow data)
```

## Running the Project
1. Install required dependencies from requirements.txt
2. Execute the Jupyter notebook step by step
3. Analyze application network profiles and classification patterns
4. Test model predictions with custom network flow examples
5. Deploy model for container visibility and management

This project demonstrates how machine learning can provide automatic container application discovery through network traffic analysis, enabling better visibility, security, and resource management in containerized environments without requiring application instrumentation or deep packet inspection.