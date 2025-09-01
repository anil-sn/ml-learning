### **Project 1: Network Traffic Anomaly Detection**

#### **1. Objective**
To build, train, and evaluate an unsupervised machine learning model capable of identifying anomalous patterns in network traffic data. This project will establish a foundational understanding of data preprocessing, feature engineering, and the application of algorithms that can find outliers without pre-labeled data.

#### **2. Business Value**
By detecting anomalies in real-time, we can proactively identify:
*   **Security Threats:** Such as Denial-of-Service (DoS) attacks or network scanning.
*   **Equipment Malfunctions:** A faulty device might generate unusual traffic patterns.
*   **Network Misconfigurations:** An improperly configured router or switch could lead to traffic spikes.
This capability enhances network reliability and security, directly benefiting our customers.

#### **3. Core Libraries**
*   `pandas`: For data loading, manipulation, and analysis.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For data preprocessing (scaling) and the machine learning model (`IsolationForest`).
*   `matplotlib` & `seaborn`: For data visualization.

#### **4. Dataset**
*   **Primary Dataset:** **NSL-KDD Dataset** ([Verified Link on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd))
*   **Why it's suitable:** The NSL-KDD dataset is a refined version of the classic KDD'99 dataset, specifically designed for evaluating network intrusion detection systems. It contains a wide variety of simulated network connections, each described by 41 features (e.g., duration, protocol type, service, byte counts). Crucially, it includes both normal traffic and various types of attacks, which we will treat as our anomalies. We will use the model in an *unsupervised* way, meaning we won't show it the labels during training.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment to keep dependencies isolated.
    ```bash
    mkdir anomaly-detection
    cd anomaly-detection
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  Install the necessary libraries.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
    ```
3.  Start a Jupyter Lab session to run the code interactively.
    ```bash
    jupyter lab
    ```

**Step 2: Load and Explore the Data**
1.  Download the `KDDTrain+.txt` file from the Kaggle dataset.
2.  In your Jupyter Notebook, load the data using Pandas. The dataset does not have a header row, so you'll need to define the column names manually. A list of column names is available on the Kaggle dataset page.
    ```python
    import pandas as pd

    # Define column names (a subset is shown for brevity)
    column_names = [ 'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', ... 'label', 'difficulty' ]
    # You will need to get all 43 column names from the dataset description

    df = pd.read_csv('KDDTrain+.txt', header=None, names=column_names)

    print(df.head())
    print(df.info())
    ```

**Step 3: Preprocess the Data**
1.  **Select Numerical Features:** For this first pass, we will focus on the numerical features, as they can be used directly by the model.
    ```python
    numerical_features = df.select_dtypes(include=['number'])
    print(numerical_features.columns)
    ```
2.  **Handle Categorical Data:** The `IsolationForest` algorithm requires all inputs to be numerical. You must convert categorical columns like `protocol_type`, `service`, and `flag` into numbers. The easiest way is using one-hot encoding.
    ```python
    df_processed = pd.get_dummies(df.drop(['label', 'difficulty'], axis=1))
    ```
3.  **Scale the Features:** Many ML algorithms are sensitive to the scale of the input data. We will scale all features to be between 0 and 1. This prevents features with large values (like `src_bytes`) from dominating the model.
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_processed)
    ```

**Step 4: Train the Anomaly Detection Model**
1.  We will use the `IsolationForest` algorithm. It works by "isolating" outliers—anomalies are easier to separate from the rest of the data points.
2.  The most important parameter is `contamination`, which is the expected proportion of outliers in the dataset. Let's start with a value of `0.02` (or 2%).
    ```python
    from sklearn.ensemble import IsolationForest

    # The 'auto' contamination is a good starting point, or you can specify a float
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X_scaled)
    ```

**Step 5: Predict and Analyze Anomalies**
1.  Use the trained model to predict which data points are anomalies. The model will return `1` for normal points (inliers) and `-1` for anomalies (outliers).
    ```python
    # Add the predictions back to the original DataFrame for analysis
    df['anomaly'] = model.predict(X_scaled)

    print(df['anomaly'].value_counts())
    ```
2.  Inspect the data points that were flagged as anomalies.
    ```python
    anomalies = df[df['anomaly'] == -1]
    print("Anomalies Found:")
    print(anomalies.head())

    print("\nDescription of Anomalies:")
    print(anomalies[numerical_features.columns].describe())

    print("\nDescription of Normal Traffic:")
    print(df[df['anomaly'] == 1][numerical_features.columns].describe())
    ```
    *Compare the descriptions. Do anomalies have significantly higher `src_bytes` or `duration`?*

**Step 6: Visualize the Results**
1.  Create a scatter plot to visually separate the normal points from the anomalies. Let's use two key features like `src_bytes` and `dst_bytes`.
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    # Using log scale can help visualize wide-ranging data
    sns.scatterplot(data=df, x='src_bytes', y='dst_bytes', hue='anomaly', palette={1: 'blue', -1: 'red'})
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Anomaly Detection in Network Traffic')
    plt.show()
    ```

#### **6. Success Criteria**
*   The model successfully runs and flags a subset of the data as anomalies.
*   The team can describe the statistical characteristics of the detected anomalies (e.g., "anomalies tend to have much higher `src_bytes` than normal traffic").
*   The team can produce a visualization that clearly distinguishes between normal and anomalous data points.

#### **7. Next Steps & Extensions**
*   **Evaluate Performance:** Compare the model's predictions (`anomaly` column) with the actual `label` column from the dataset. How many of the real attacks did the model find? This turns it into a semi-supervised evaluation problem.
*   **Tune Hyperparameters:** Experiment with the `contamination` parameter in `IsolationForest` to see how it affects the number of detected anomalies.
*   **Try Other Algorithms:** Implement another unsupervised algorithm like `DBSCAN` or a One-Class SVM and compare the results.