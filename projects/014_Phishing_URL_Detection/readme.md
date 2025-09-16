# Project 14: Phishing & Malicious URL Detection from Web Proxy Logs

## Objective

Build a fast, interpretable machine learning model that can classify a URL as 'benign' or 'malicious' by engineering features directly from the URL string itself, providing actionable intelligence for network security teams.

## Business Value

**For Network Security Teams:**
- **Real-time Protection**: Deploy lightweight URL filtering at web proxies and DNS servers
- **Interpretable Results**: Clear feature importance rankings show exactly which URL characteristics indicate threats
- **Rapid Response**: Fast inference enables real-time blocking with minimal latency impact
- **Cost Efficiency**: Reduce reliance on expensive threat intelligence feeds through self-contained detection

**For Enterprise IT:**
- **Proactive Defense**: Block malicious URLs before users can access them
- **Compliance Support**: Meet regulatory requirements for web content filtering
- **User Productivity**: Minimize false positives to avoid blocking legitimate business websites
- **Integration Ready**: Lightweight model suitable for existing security infrastructure

## Core Libraries

- **pandas & numpy**: Data processing and numerical computations for URL feature extraction
- **scikit-learn**: LogisticRegression with balanced class weights for interpretable classification
- **matplotlib & seaborn**: Feature importance visualization and confusion matrix analysis
- **urllib.parse**: URL parsing for extracting hostname, path, and query components
- **kaggle**: Access to "Malicious and Benign Websites" dataset with labeled URLs

## Dataset

**Source**: Kaggle - "Malicious and Benign Websites" dataset
- **URL Samples**: Large collection of labeled benign and malicious URLs
- **Features**: Lexical characteristics engineered directly from URL strings
- **Balance**: Slightly imbalanced dataset requiring class weight adjustment
- **Real-world Relevance**: URLs from actual phishing campaigns, malware distribution sites, and legitimate websites

**Key Engineered Features:**
- URL structure: length, hostname length, path length, directory count
- Special characters: dash, at-symbol, question mark, percent encoding counts
- Protocol indicators: HTTP vs HTTPS usage patterns
- Content indicators: digit/letter ratios, www presence

## Step-by-Step Guide

### 1. Environment Setup and Data Acquisition
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle urllib3
```

Set up Kaggle API credentials and download the "Malicious and Benign Websites" dataset containing labeled URL samples.

### 2. URL Feature Engineering
```python
# Engineer lexical features from raw URL strings
df['url_length'] = df['Url'].apply(len)
df['hostname_length'] = df['Url'].apply(lambda x: len(urlparse(x).netloc))
df['path_length'] = df['Url'].apply(lambda x: len(urlparse(x).path))
df['count_dash'] = df['Url'].apply(lambda x: x.count('-'))
df['count_at'] = df['Url'].apply(lambda x: x.count('@'))
df['count_question'] = df['Url'].apply(lambda x: x.count('?'))
```

### 3. Data Preprocessing and Label Encoding
```python
# Binary encoding: malicious -> 1, benign -> 0
df['Label'] = df['Label'].apply(lambda x: 1 if x == 'malicious' else 0)

# Feature scaling for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### 4. Model Training with Class Balance
```python
# Logistic Regression for interpretability
model = LogisticRegression(
    random_state=42, 
    class_weight='balanced',  # Handle class imbalance
    max_iter=200
)
model.fit(X_train_scaled, y_train)
```

### 5. Security-Focused Evaluation
```python
# Classification performance
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

# Focus on minimizing false negatives (missed threats)
cm = confusion_matrix(y_test, y_pred)
```

### 6. Model Interpretability Analysis
```python
# Feature coefficient analysis for threat intelligence
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
```

## Success Criteria

**Primary Metrics:**
- **Precision for Malicious Class**: >90% (minimize false alarms in production)
- **Recall for Malicious Class**: >90% (catch actual threats)
- **F1-Score**: >0.90 for balanced performance

**Secondary Metrics:**
- **Model Interpretability**: Clear coefficient rankings for security analysis
- **Inference Speed**: <1ms per URL for real-time deployment
- **Feature Stability**: Consistent feature importance across different data samples

**Business Impact:**
- Deploy in web proxies for real-time URL filtering
- Integrate with email security gateways
- Provide threat intelligence insights through feature analysis

## Next Steps & Extensions

### Immediate Improvements
- **Feature Enhancement**: Add entropy-based features and subdomain analysis
- **Temporal Features**: Incorporate domain age and registration patterns
- **Context Features**: Include referring page and user behavior patterns

### Advanced Techniques
- **Ensemble Methods**: Combine with neural networks for improved accuracy
- **Active Learning**: Continuously update model with new threat samples
- **Multi-class Classification**: Distinguish between phishing, malware, and other threat types

### Production Deployment
- **Real-time Pipeline**: Stream processing integration with network appliances
- **Model Monitoring**: Performance tracking and drift detection
- **Feedback Loop**: Incorporate security analyst feedback for continuous improvement

### Domain Expansion
- **Cross-Language Support**: Handle internationalized domain names (IDN)
- **Mobile Integration**: Adapt for mobile app URL filtering
- **API Development**: REST API for integration with existing security tools

This project demonstrates practical application of interpretable machine learning for cybersecurity, providing both high accuracy threat detection and actionable insights for security teams.