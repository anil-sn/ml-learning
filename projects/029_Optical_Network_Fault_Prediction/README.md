# Project 029: Optical Network Fault Prediction

## Objective
Build a machine learning model that can predict an impending fault in an optical network device (like an amplifier or transceiver) by analyzing its real-time performance metrics.

## Business Value
- **Predictive Maintenance**: Prevent costly optical network outages through early fault detection
- **Service Continuity**: Maintain high availability for critical communication services
- **Cost Reduction**: Reduce emergency repairs and minimize truck rolls for maintenance
- **SLA Compliance**: Ensure service level agreements by proactive fault management
- **Resource Optimization**: Schedule maintenance during planned windows instead of emergency situations

## Core Libraries
- **scikit-learn**: Random Forest Classifier for fault prediction and model evaluation
- **pandas**: Dataset manipulation and time-series analysis
- **numpy**: Numerical computations and signal processing
- **matplotlib/seaborn**: Data visualization and optical signal analysis
- **kaggle**: API for accessing the Optical Network Intrusion Dataset

## Dataset
- **Source**: Kaggle - "Optical Network Intrusion Dataset"
- **Size**: Time-series optical signal properties and performance metrics
- **Features**: Optical power levels, OSNR measurements, signal quality indicators
- **Labels**: Stable vs. unstable/pre-fault conditions
- **Type**: Time-series optical network telemetry data

## Step-by-Step Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv optical_fault_env
source optical_fault_env/bin/activate  # On Windows: optical_fault_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
```

### 2. Data Collection and Preparation
```python
# Download dataset using Kaggle API
import kaggle
kaggle.api.dataset_download_files('561616/optical-network-intrusion-dataset', 
                                 path='.', unzip=True)

# Load and preprocess optical telemetry data
import pandas as pd
df = pd.read_csv('optical_data/Optical_Intrusion_Dataset.csv')

# Rename target column for fault prediction context
df.rename(columns={'Intrusion': 'Fault_Condition'}, inplace=True)
df = df.drop(columns=['Unnamed: 0'])  # Remove ID column
```

### 3. Feature Engineering
```python
from sklearn.preprocessing import LabelEncoder

# Encode target variable: 'No' -> 0 (Stable), 'Yes' -> 1 (Unstable/Fault)
le = LabelEncoder()
df['Fault_Condition'] = le.fit_transform(df['Fault_Condition'])

# Key optical features for fault prediction
optical_features = [col for col in df.columns if col != 'Fault_Condition']
X = df[optical_features]
y = df['Fault_Condition']
```

### 4. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest with class balancing
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1, 
    class_weight='balanced'
)

model.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

# Evaluate with focus on recall for fault detection
print("Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Stable', 'Unstable/Fault']))

# Confusion matrix for fault prediction assessment
cm = confusion_matrix(y_test, y_pred)
```

### 6. Feature Importance Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze which optical metrics predict faults
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Visualize top predictive optical metrics
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_df.head(15), x='Importance', y='Feature')
plt.title('Top 15 Optical Metrics for Fault Prediction')
plt.show()
```

## Success Criteria
- **High Recall (>90%)**: Catch most impending faults to prevent outages
- **Balanced Precision**: Minimize false alarms that could overwhelm operations teams
- **Early Warning**: Detect pre-fault conditions with sufficient lead time for action
- **Interpretable Results**: Clear understanding of which optical metrics indicate problems

## Next Steps & Extensions
1. **Real-time Integration**: Deploy with optical line system telemetry for live monitoring
2. **Fault Severity Classification**: Predict severity levels and expected time to failure
3. **Automated Response**: Integrate with network automation for automatic rerouting
4. **Multi-vendor Support**: Adapt for different optical equipment manufacturers
5. **Predictive Scheduling**: Optimize maintenance windows based on fault predictions
6. **Environmental Factors**: Include temperature, humidity, and other external conditions

## Files Structure
```
029_Optical_Network_Fault_Prediction/
├── README.md
├── optical_fault_prediction.ipynb
├── requirements.txt
└── data/
    └── (Kaggle dataset files)
```

## Running the Project
1. Set up Kaggle API credentials
2. Execute the Jupyter notebook step by step
3. Analyze optical signal patterns and fault indicators
4. Review feature importance for optical engineering insights

This project demonstrates how machine learning can enhance optical network reliability by predicting equipment failures before they cause service disruptions, enabling proactive maintenance strategies.