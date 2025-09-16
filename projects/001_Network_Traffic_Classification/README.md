### **Project 1: Network Traffic Classification**

#### **1. Objective**
To build and train a supervised machine learning model capable of classifying network traffic into different categories (Normal vs. Attack) using the UNSW-NB15 dataset. This project demonstrates the application of RandomForest classifier for network intrusion detection, providing a robust baseline for security monitoring systems.

#### **2. Business Value**
By accurately classifying network traffic, we can:
*   **Enhanced Security:** Proactively identify malicious traffic and potential security threats
*   **Network Monitoring:** Establish real-time classification of network activities
*   **Incident Response:** Provide automated first-line detection to reduce response times
*   **Compliance:** Meet security monitoring requirements for network infrastructure
This capability directly improves network security posture and operational efficiency.

#### **3. Core Libraries**
*   `pandas`: For data loading, manipulation, and analysis
*   `numpy`: For numerical operations and array handling
*   `scikit-learn`: For machine learning model (RandomForestClassifier) and preprocessing
*   `matplotlib` & `seaborn`: For data visualization and results presentation
*   `kaggle`: For dataset acquisition from Kaggle platform

#### **4. Dataset**
*   **Primary Dataset:** **UNSW-NB15 Dataset** ([Available on Kaggle](https://www.kaggle.com/datasets/rawadahmed/unsw-nb15))
*   **Why it's suitable:** The UNSW-NB15 dataset is a comprehensive network intrusion detection dataset containing realistic modern network traffic. It includes both normal activities and contemporary attack behaviors, with 49 features extracted from network flows. The dataset provides ground truth labels for both binary classification (Normal/Attack) and multi-class classification (specific attack categories), making it ideal for supervised learning approaches.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and set up dependencies
    ```bash
    mkdir network-traffic-classification
    cd network-traffic-classification
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  Install the necessary libraries
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn kaggle jupyterlab
    ```
3.  Set up Kaggle API credentials for data download
    ```bash
    # Place your kaggle.json file in ~/.kaggle/
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

**Step 2: Data Acquisition and Loading**
1.  Download the UNSW-NB15 dataset using Kaggle API
    ```python
    import kaggle
    kaggle.api.dataset_download_files('rawadahmed/unsw-nb15', unzip=True)
    ```
2.  Load and combine the training and testing datasets
    ```python
    import pandas as pd
    
    # Load the pre-split datasets
    df_train = pd.read_csv('UNSW_NB15_training-set.csv')
    df_test = pd.read_csv('UNSW_NB15_testing-set.csv')
    
    # Combine for consistent preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Dataset shape: {df.shape}")
    ```

**Step 3: Data Preprocessing**
1.  **Clean and prepare the data**
    ```python
    # Remove unnecessary ID column
    df = df.drop(columns=['id'])
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Examine target distribution
    print(df['attack_cat'].value_counts())
    ```
2.  **Handle categorical features**
    ```python
    from sklearn.preprocessing import LabelEncoder
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    feature_cols = [col for col in categorical_cols if col != 'attack_cat']
    
    # Apply one-hot encoding to features
    df_encoded = pd.get_dummies(df, columns=feature_cols, drop_first=True)
    
    # Label encode the target variable
    y_encoder = LabelEncoder()
    df_encoded['attack_cat'] = y_encoder.fit_transform(df_encoded['attack_cat'])
    ```
3.  **Prepare features and target**
    ```python
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df_encoded.drop(columns=['attack_cat', 'label'])
    y = df_encoded['attack_cat']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    ```

**Step 4: Model Training**
1.  Initialize and train the RandomForest classifier
    ```python
    from sklearn.ensemble import RandomForestClassifier
    import time
    
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Train the model
    print("Training RandomForestClassifier...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    ```

**Step 5: Model Evaluation**
1.  Make predictions and evaluate performance
    ```python
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Display detailed classification report
    target_names = y_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    ```
2.  Visualize results with confusion matrix
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Network Traffic Classification - Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.show()
    ```

#### **6. Success Criteria**
*   Model achieves accuracy > 85% on test set
*   Successfully classifies both normal and attack traffic with good precision/recall
*   Confusion matrix shows clear separation between classes
*   Model training completes without errors and in reasonable time
*   Feature importance analysis provides insights into key network characteristics

#### **7. Next Steps & Extensions**
*   **Feature Engineering:** Explore additional feature combinations and transformations
*   **Model Comparison:** Implement and compare with other algorithms (SVM, Neural Networks, XGBoost)
*   **Hyperparameter Tuning:** Use GridSearch or RandomSearch for optimal parameters
*   **Real-time Implementation:** Deploy model for live network traffic classification
*   **Multi-class Analysis:** Deep dive into specific attack category classification performance
*   **Ensemble Methods:** Combine multiple models for improved accuracy