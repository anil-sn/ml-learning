### **Project 2: Basic Anomaly Detection in Network Logs**

#### **1. Objective**
To identify anomalous sequences of events in system logs using unsupervised machine learning. This project demonstrates how to detect faults, misconfigurations, or security incidents without needing prior examples of every possible bad event, making it crucial for proactive network monitoring.

#### **2. Business Value**
By automatically detecting log anomalies, we can:
*   **Proactive Issue Detection:** Identify problems before they impact customers
*   **Security Monitoring:** Detect unusual patterns that may indicate security breaches
*   **System Health Monitoring:** Catch configuration errors and system faults early
*   **Operational Efficiency:** Reduce manual log analysis and accelerate incident response
This capability enhances overall network reliability and security posture.

#### **3. Core Libraries**
*   `pandas`: For data manipulation and log processing
*   `numpy`: For numerical operations and data handling
*   `scikit-learn`: For the IsolationForest algorithm and TF-IDF vectorization
*   `matplotlib` & `seaborn`: For visualization of results
*   `re`: For regular expression-based log parsing
*   `kaggle`: For dataset acquisition

#### **4. Dataset**
*   **Primary Dataset:** **HDFS Log Anomaly Detection Dataset** ([Available on Kaggle](https://www.kaggle.com/datasets/logpai/hdfs-log-anomaly-detection))
*   **Why it's suitable:** The HDFS dataset tracks events related to data block operations in a large computing cluster, making it an excellent proxy for complex network system logs. It contains both normal operational logs and anomalous sequences, with ground truth labels for evaluation. The logs represent realistic system behavior patterns that are similar to what network engineers encounter in production environments.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create project environment and install dependencies
    ```bash
    mkdir log-anomaly-detection
    cd log-anomaly-detection
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install required libraries
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn kaggle jupyterlab
    ```
3.  Configure Kaggle API for data download
    ```bash
    # Place kaggle.json in ~/.kaggle/
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

**Step 2: Data Acquisition and Loading**
1.  Download the HDFS log dataset
    ```python
    import kaggle
    kaggle.api.dataset_download_files('logpai/hdfs-log-anomaly-detection', unzip=True)
    ```
2.  Load and examine the log data
    ```python
    import pandas as pd
    import re
    
    # Load the ground truth labels
    labels_df = pd.read_csv('anomaly_label.csv')
    print("Labels loaded:", labels_df.shape)
    
    # Load raw log file
    with open('HDFS.log', 'r') as f:
        logs = f.readlines()
    print(f"Total log lines: {len(logs)}")
    ```

**Step 3: Log Parsing and Preprocessing**
1.  **Parse log entries to extract block IDs and content**
    ```python
    def parse_log_line(line):
        # Extract block ID using regex
        match = re.search(r'(blk_[-]?\\d+)', line)
        block_id = match.group(1) if match else None
        content = line.strip()
        return block_id, content
    
    # Parse all logs
    parsed_logs = [parse_log_line(line) for line in logs]
    log_df = pd.DataFrame(parsed_logs, columns=['BlockId', 'Content'])
    log_df.dropna(inplace=True)
    
    print(f"Parsed logs: {log_df.shape}")
    ```
2.  **Group logs by session (BlockId)**
    ```python
    # Each BlockId represents a session - aggregate log content
    print("Grouping logs by BlockId (session)...")
    session_df = log_df.groupby('BlockId')['Content'].apply(
        lambda x: ' '.join(x)
    ).reset_index()
    
    # Merge with ground truth labels
    session_df = pd.merge(session_df, labels_df, on='BlockId', how='left')
    session_df['Label'].fillna('Normal', inplace=True)
    
    print(f"Session data shape: {session_df.shape}")
    print("Label distribution:", session_df['Label'].value_counts())
    ```

**Step 4: Feature Engineering with TF-IDF**
1.  **Convert log text to numerical vectors**
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("Converting log messages to TF-IDF vectors...")
    # TF-IDF gives more weight to terms frequent in a document but rare across all documents
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(session_df['Content'])
    
    print(f"Feature matrix shape: {X.shape}")
    ```

**Step 5: Unsupervised Anomaly Detection**
1.  **Train Isolation Forest model**
    ```python
    from sklearn.ensemble import IsolationForest
    
    # Calculate expected anomaly proportion from labels
    anomaly_proportion = len(labels_df[labels_df['Label'] == 'Anomaly']) / len(session_df)
    print(f"Estimated anomaly proportion: {anomaly_proportion:.4f}")
    
    # Initialize and train the model
    model = IsolationForest(
        n_estimators=100,
        contamination=anomaly_proportion,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Isolation Forest model...")
    model.fit(X)  # Unsupervised - no labels used in training
    print("Training complete.")
    ```

**Step 6: Model Evaluation**
1.  **Generate predictions and evaluate performance**
    ```python
    from sklearn.metrics import classification_report, accuracy_score
    
    # Predict anomalies (1 for normal, -1 for anomaly)
    predictions = model.predict(X)
    
    # Convert ground truth to same format
    y_true = session_df['Label'].apply(lambda x: 1 if x == 'Normal' else -1)
    y_pred = predictions
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Anomaly (-1)', 'Normal (1)']))
    ```

**Step 7: Results Analysis and Visualization**
1.  **Analyze detected anomalies**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Add predictions to dataframe for analysis
    session_df['Prediction'] = predictions
    
    # Show examples of detected anomalies
    detected_anomalies = session_df[session_df['Prediction'] == -1]
    print(f"Detected {len(detected_anomalies)} anomalies")
    
    # Visualization of results
    plt.figure(figsize=(10, 6))
    confusion_data = pd.crosstab(session_df['Label'], session_df['Prediction'])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues')
    plt.title('Log Anomaly Detection Results')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    ```

#### **6. Success Criteria**
*   Successfully parse and process log files into structured format
*   TF-IDF vectorization creates meaningful numerical features from log text
*   Isolation Forest model trains without errors and generates predictions
*   Model achieves reasonable precision/recall balance for anomaly detection
*   Clear visualization of results showing model performance
*   Ability to identify and analyze specific anomalous log sequences

#### **7. Next Steps & Extensions**
*   **Advanced Text Processing:** Implement log templating to extract structured patterns
*   **Sequential Analysis:** Use LSTM or other sequence models for temporal patterns
*   **Real-time Processing:** Implement streaming anomaly detection for live logs
*   **Multi-source Integration:** Combine logs from multiple system components
*   **Alert System:** Build automated alerting based on anomaly scores
*   **Parameter Tuning:** Optimize contamination parameter and TF-IDF settings