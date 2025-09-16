# Project 13: Identifying Lateral Movement in a Network

## Objective

Build an unsupervised anomaly detection model that can identify hosts exhibiting behavior indicative of lateral movement, such as internal port scanning or connecting to an unusually high number of other hosts.

## Business Value

**For Security Operations Centers (SOC):**
- **Early Breach Detection**: Identify compromised hosts before they cause significant damage
- **Automated Threat Hunting**: Replace manual analysis with scalable machine learning detection
- **Behavioral Baselining**: Establish normal network behavior patterns for anomaly detection
- **Incident Response**: Provide forensic insights into attacker movement patterns

**For Network Security Teams:**
- **Zero-Day Protection**: Detect novel attack techniques without signature-based rules
- **Insider Threat Detection**: Identify malicious or compromised internal users
- **Compliance Support**: Meet regulatory requirements for advanced persistent threat detection
- **Risk Assessment**: Quantify and visualize network security posture

## Core Libraries

- **pandas & numpy**: Network flow data processing and host behavior profiling
- **scikit-learn**: Isolation Forest for unsupervised anomaly detection
- **matplotlib & seaborn**: Network behavior visualization and anomaly analysis
- **kaggle**: CIC-IDS2017 dataset containing port scan and lateral movement examples

## Dataset

**Source**: CIC-IDS2017 from Kaggle - Tuesday working hours traffic
- **Normal Traffic**: Legitimate internal network communications between hosts
- **Attack Traffic**: Port scans and lateral movement activities from compromised hosts
- **Host Profiling**: Aggregate statistics per source IP for behavioral analysis
- **Features**: Unique destination IPs, ports, flow counts, and timing patterns

**Key Behavioral Indicators:**
- **Unique Destination IPs**: Scanning behavior targets many hosts
- **Unique Destination Ports**: Port scanning attempts across services
- **Total Flow Count**: Volume of network connections initiated
- **Average Flow Duration**: Connection pattern analysis

## Step-by-Step Guide

### 1. Environment Setup and Data Loading
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
```

Download CIC-IDS2017 dataset and load Tuesday traffic containing port scanning activities.

### 2. Host Behavior Profiling
```python
# Aggregate network flows by source IP to create host profiles
host_profiles = df.groupby('Source IP').agg({
    'Destination IP': 'nunique',      # Scanning indicator
    'Destination Port': 'nunique',    # Port scan indicator  
    'Flow ID': 'count',              # Connection volume
    'Flow Duration': 'mean'          # Timing patterns
}).reset_index()
```

### 3. Ground Truth Labeling
```python
# Label hosts based on their traffic mix
def get_host_label(group):
    if (group != 'BENIGN').any():
        return 'Attack'
    return 'BENIGN'

host_labels = df.groupby('Source IP')['Label'].apply(get_host_label)
```

### 4. Unsupervised Model Training
```python
# Train Isolation Forest only on known benign host behavior
X_train_benign = X[host_profiles['Label'] == 'BENIGN']
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(scaler.fit_transform(X_train_benign))
```

### 5. Anomaly Detection and Evaluation
```python
# Apply trained model to all hosts (benign and malicious)
predictions = model.predict(scaler.transform(X))
# -1 indicates anomaly (potential lateral movement)
# +1 indicates normal behavior
```

### 6. Forensic Analysis of Detected Anomalies
```python
# Analyze characteristics of flagged hosts
detected_anomalies = host_profiles[host_profiles['Prediction'] == -1]
# Focus on hosts with high unique destination counts
```

## Success Criteria

**Primary Metrics:**
- **Recall for Attack Hosts**: >85% (catch lateral movement attempts)
- **Precision Balance**: Reasonable false positive rate for analyst review
- **Unsupervised Performance**: Effective learning from benign data only

**Secondary Metrics:**
- **Feature Interpretability**: Clear understanding of anomalous behavior patterns
- **Scalability**: Processing capability for large network environments
- **Temporal Stability**: Consistent performance across different time periods

**Operational Requirements:**
- Real-time host behavior scoring for continuous monitoring
- Integration with SIEM platforms for automated alerting
- Forensic reporting capabilities for incident investigation

## Next Steps & Extensions

### Immediate Improvements
- **Temporal Analysis**: Include time-based patterns in host behavior profiles
- **Network Topology**: Incorporate subnet and VLAN information for context
- **Protocol Analysis**: Extend beyond TCP/UDP to include application-layer protocols

### Advanced Techniques
- **Graph Neural Networks**: Model network topology and host relationships
- **Time Series Clustering**: Identify temporal patterns in lateral movement
- **Multi-Scale Analysis**: Combine individual flow and host-level behavioral analysis

### Production Deployment
- **Stream Processing**: Real-time host behavior profiling and anomaly detection
- **Threat Intelligence**: Integration with external IOCs and threat feeds
- **Automated Response**: Quarantine or investigate flagged hosts automatically

### Specialized Applications
- **Insider Threat Programs**: Adapt for detecting malicious employee activities
- **Cloud Security**: Extend to container and serverless environments
- **IoT Networks**: Specialized profiling for IoT device behavioral patterns
- **Critical Infrastructure**: Enhanced monitoring for SCADA and industrial networks

This project provides essential capabilities for detecting advanced persistent threats that have evaded perimeter defenses, focusing on the critical lateral movement phase where attackers spread through internal networks.