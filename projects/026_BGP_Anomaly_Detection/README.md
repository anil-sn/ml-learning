# Project 026: BGP Anomaly Detection (Route Leaks, Hijacks)

## Objective
Build an unsupervised anomaly detection model that can identify anomalous BGP update messages, such as those indicative of a route leak or prefix hijack, by analyzing features of the BGP AS-path.

## Business Value
- **Network Security**: Protect against BGP hijacking attacks and route leaks that can disrupt internet connectivity
- **Early Warning System**: Provide rapid detection of routing anomalies before they cause widespread outages
- **SLA Protection**: Maintain service availability by detecting and mitigating routing attacks quickly
- **Compliance**: Meet security requirements for internet service providers and large enterprises

## Core Libraries
- **scikit-learn**: Isolation Forest for anomaly detection, data preprocessing, and model evaluation
- **pandas**: Dataset manipulation and feature engineering 
- **numpy**: Numerical computations and data array operations
- **matplotlib/seaborn**: Data visualization and results analysis
- **kaggle**: API for accessing the BGP Hijacking Detection Dataset

## Dataset
- **Source**: Kaggle - "BGP Hijacking Detection Dataset"
- **Size**: Real-world BGP update messages with extracted features
- **Features**: AS-path characteristics including length statistics, edit distances, prefix information
- **Labels**: Normal vs. anomalous BGP updates (used only for evaluation)
- **Type**: Time-series network routing data

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv bgp_anomaly_env
source bgp_anomaly_env/bin/activate  # On Windows: bgp_anomaly_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
```

### 2. Data Collection and Preparation
```python
# Download dataset using Kaggle API
import kaggle
kaggle.api.dataset_download_files('dprembath/bgp-hijacking-detection-dataset', 
                                 path='.', unzip=True)

# Load and preprocess data
import pandas as pd
df = pd.read_csv('bgp_data.csv')

# Key BGP features for anomaly detection
feature_cols = [
    'AS_PATH_LEN', 'AS_PATH_AVG_LEN', 'AS_PATH_MAX_LEN', 
    'EDIT_DIST_AS_PATH', 'PREFIX_LEN', 'UNIQUE_AS_COUNT'
]
```

### 3. Feature Engineering
```python
from sklearn.preprocessing import StandardScaler

# Separate normal traffic for training (unsupervised approach)
X_normal = df[df['Label'] == 'normal'][feature_cols]

# Scale features for better model performance  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normal)
```

### 4. Model Training
```python
from sklearn.ensemble import IsolationForest

# Configure Isolation Forest for BGP anomaly detection
model = IsolationForest(
    n_estimators=100,
    contamination=0.15,  # Expected anomaly rate
    random_state=42,
    n_jobs=-1
)

# Train on normal BGP updates only
model.fit(X_scaled)
```

### 5. Anomaly Detection and Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict anomalies on full dataset
X_all_scaled = scaler.transform(df[feature_cols])
predictions = model.predict(X_all_scaled)

# Evaluate performance
print(classification_report(y_true, predictions, 
                          target_names=['Anomaly', 'Normal']))
```

### 6. Results Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze detected anomalies
df['prediction'] = predictions
anomalies = df[df['prediction'] == -1]

# Compare AS-path length distributions
plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['prediction'] == 1]['AS_PATH_LEN'], 
           label='Normal', fill=True)
sns.kdeplot(anomalies['AS_PATH_LEN'], 
           label='Detected Anomaly', fill=True, color='red')
plt.title('AS-Path Length Distribution: Normal vs Anomalous BGP Updates')
plt.show()
```

## Success Criteria
- **High Recall (>90%)**: Successfully detect most BGP anomalies to prevent security breaches
- **Low False Positive Rate (<5%)**: Minimize false alarms that could overwhelm network operators
- **Fast Detection**: Process BGP updates in near real-time for rapid threat response
- **Interpretable Results**: Provide clear insights into what makes BGP updates anomalous

## Next Steps & Extensions
1. **Real-time Integration**: Deploy model with BGP routing daemons for live monitoring
2. **Multi-vendor Support**: Adapt for different router vendors and BGP implementations
3. **Automated Response**: Integrate with network automation to auto-block suspicious routes
4. **Advanced Features**: Include geographical and temporal patterns in anomaly detection
5. **Ensemble Methods**: Combine multiple anomaly detection algorithms for better accuracy

## Files Structure
```
026_BGP_Anomaly_Detection/
├── README.md
├── bgp_anomaly_detection.ipynb
├── requirements.txt
└── data/
    └── (Kaggle dataset files)
```

## Running the Project
1. Set up Kaggle API credentials
2. Execute the Jupyter notebook step by step
3. Review anomaly detection results and visualizations
4. Analyze feature importance for BGP security insights

This project demonstrates how unsupervised machine learning can enhance network security by detecting routing anomalies that traditional rule-based systems might miss.