# Project 24: Quality of Experience (QoE) Prediction for Video Streaming

## Objective
Build a machine learning model that predicts the Quality of Experience (QoE) score for video streaming sessions based on network performance metrics like throughput, packet loss, and jitter to enable proactive quality management.

## Business Value
- **Proactive Quality Management**: Identify and prevent video quality issues before users experience buffering
- **Network Optimization**: Prioritize traffic and allocate resources based on predicted quality impact
- **Customer Satisfaction**: Maintain high-quality streaming experiences to reduce churn
- **Revenue Protection**: Prevent revenue loss from poor streaming experiences
- **Operational Intelligence**: Provide network teams with actionable insights for infrastructure improvements

## Core Libraries
- **pandas & numpy**: Data manipulation and numerical operations
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **matplotlib & seaborn**: Data visualization and performance analysis
- **RandomForestClassifier**: Ensemble learning for robust QoE classification

## Dataset
**Source**: Kaggle - "YouTube UGC Video Quality & Network Dataset"
- **Size**: Real-world measurements from thousands of YouTube streaming sessions
- **Features**: Network performance metrics (throughput, packet loss, latency, jitter)
- **Target**: QoE categories derived from buffering events and resolution changes
- **Scope**: Comprehensive network conditions and video quality indicators
- **Quality**: Production-grade data from actual video streaming sessions

## Step-by-Step Guide

### 1. Environment Setup
```python
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn kaggle

# Set up Kaggle API for dataset download
# Upload kaggle.json file when prompted
```

### 2. Data Collection and Preprocessing
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Download and merge YouTube UGC dataset
df_net = pd.read_csv('network_features.csv')
df_vid = pd.read_csv('video_features.csv')
df = pd.merge(df_net, df_vid, on='session_id')

# Clean and prepare data
df = df.drop(columns=['session_id', 'vmaf'])  # Remove identifiers and direct quality metrics
df = df.loc[:, (df != df.iloc[0]).any()]  # Remove zero-variance columns
df.dropna(inplace=True)
```

### 3. QoE Target Engineering
```python
# Create rule-based QoE categories
def get_qoe_label(row):
    """
    Classify QoE based on streaming experience indicators
    
    Poor: Any stalling/buffering events
    Fair: Multiple resolution changes (adaptive streaming active)
    Good: Smooth streaming with minimal adaptations
    """
    if row['stalls'] > 0:
        return 'Poor'
    elif row['resolution_changes'] > 2:
        return 'Fair'
    else:
        return 'Good'

df['qoe_label'] = df.apply(get_qoe_label, axis=1)

# Remove columns used for label creation
df = df.drop(columns=['stalls', 'resolution_changes'])
```

### 4. Data Preparation and Splitting
```python
from sklearn.model_selection import train_test_split

# Prepare features and target
X = df.drop(columns=['qoe_label'])
y = df['qoe_label']

# Encode categorical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
```

### 5. Model Training with RandomForest
```python
from sklearn.ensemble import RandomForestClassifier

# Configure RandomForest with balanced class weights
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

# Train the model
model.fit(X_train, y_train)
```

### 6. Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Make predictions
y_pred = model.predict(X_test)

# Detailed performance analysis
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('QoE Prediction Confusion Matrix')
plt.show()
```

### 7. Feature Importance Analysis
```python
import matplotlib.pyplot as plt

# Identify most important network metrics
importances = model.feature_importances_
top_indices = np.argsort(importances)[-15:]

# Visualize key QoE drivers
plt.figure(figsize=(12, 8))
plt.barh(range(15), importances[top_indices])
plt.yticks(range(15), [X.columns[i] for i in top_indices])
plt.title('Top Network Metrics for QoE Prediction')
plt.xlabel('Feature Importance')
plt.show()
```

### 8. Real-Time QoE Prediction
```python
def predict_qoe_risk(network_metrics):
    """
    Predict QoE category for real-time streaming session
    
    Args:
        network_metrics: Dict of network performance indicators
    
    Returns:
        Predicted QoE category and probability
    """
    # Convert metrics to feature vector
    feature_vector = np.array([list(network_metrics.values())])
    
    # Predict QoE category and probability
    prediction = model.predict(feature_vector)[0]
    probability = model.predict_proba(feature_vector)[0].max()
    
    qoe_category = le.inverse_transform([prediction])[0]
    return qoe_category, probability

# Example usage for monitoring
network_status = {
    'throughput_avg': 5.2,      # Mbps
    'throughput_std': 0.8,      # Variability
    'packet_loss': 0.02,        # 2% loss
    'rtt_avg': 45,              # ms latency
    'jitter': 15                # ms jitter
}

qoe_prediction, confidence = predict_qoe_risk(network_status)
print(f"Predicted QoE: {qoe_prediction} (confidence: {confidence:.2%})")
```

## Success Criteria
- **Primary Metric**: High recall for 'Poor' QoE category (minimize missed quality issues)
- **Precision**: Accurate identification of quality problems to avoid false alarms
- **Overall Accuracy**: >85% classification accuracy across all QoE categories
- **Real-Time Performance**: Sub-second prediction time for operational deployment
- **Interpretability**: Clear feature importance rankings for network optimization

## Next Steps & Extensions

### Technical Enhancements
1. **Advanced Models**: Experiment with XGBoost, neural networks, or ensemble methods
2. **Time Series Analysis**: Include temporal patterns and user behavior trends
3. **Multi-Modal Features**: Incorporate device capabilities, content metadata, and user preferences
4. **Online Learning**: Continuous model updates based on real-time feedback

### Business Applications
1. **Adaptive Streaming**: Dynamic quality adjustments based on predicted QoE
2. **Network Operations**: Automated traffic prioritization and resource allocation
3. **Customer Support**: Proactive outreach for users experiencing quality issues
4. **Infrastructure Planning**: Data-driven network capacity and upgrade decisions

### Advanced Analytics
1. **Personalization**: User-specific QoE models based on viewing patterns
2. **Content-Aware QoE**: Different quality thresholds for different content types
3. **Geographic Analysis**: Location-based QoE patterns and infrastructure optimization
4. **A/B Testing**: Quantify impact of network improvements on user experience

## Files in this Project
- `README.md` - Project documentation and implementation guide
- `qoe_prediction_video_streaming.ipynb` - Complete Jupyter notebook implementation
- `requirements.txt` - Python package dependencies

## Key Insights
- Throughput consistency (average and variability) is the primary driver of streaming QoE
- Packet-level metrics like reordering and latency significantly impact user experience
- RandomForest with balanced class weights effectively handles QoE category imbalance
- High recall for 'Poor' QoE detection enables proactive quality management
- Feature importance analysis provides actionable network optimization guidance

## QoE Classification Framework
- **Good QoE**: Smooth streaming with minimal adaptive bitrate changes
- **Fair QoE**: Noticeable quality adaptations but no interruptions
- **Poor QoE**: Buffering events that directly impact user experience
- **Rule-Based Approach**: Transparent, interpretable quality categorization
- **Network-Centric**: Focus on measurable infrastructure performance indicators