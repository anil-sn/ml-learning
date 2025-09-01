### **Project 2: Predictive Maintenance for Network Equipment**

#### **1. Objective**
To build a supervised machine learning model that predicts impending equipment failure based on real-time telemetry data. This project will teach the team how to handle imbalanced datasets—a common and critical challenge in real-world applications where failure events are rare.

#### **2. Business Value**
Proactively identifying at-risk equipment allows us to:
*   **Reduce Network Downtime:** Schedule maintenance before a failure occurs, minimizing service disruption for customers.
*   **Lower Operational Costs:** Reduce the need for emergency truck rolls and expensive, unplanned hardware replacements.
*   **Improve Customer Satisfaction:** Enhance network reliability and build trust in our service quality.

#### **3. Core Libraries**
*   `pandas` & `numpy`: For data loading and manipulation.
*   `scikit-learn`: For data splitting and model evaluation.
*   `imbalanced-learn`: For handling the imbalanced dataset using SMOTE.
*   `xgboost`: For training a high-performance gradient boosting model.
*   `matplotlib` & `seaborn`: For data visualization and interpreting results.

#### **4. Dataset**
*   **Primary Dataset:** **Machine Predictive Maintenance Classification** ([Verified Link on Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification))
*   **Why it's suitable:** This dataset is a perfect proxy for our use case. It contains sensor readings from machines (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`) and, most importantly, a binary label indicating if a failure occurred (`Target`). We can map these features conceptually to telemetry from our network devices (e.g., ONT temperature, error counters, processing load). The dataset is also realistically imbalanced, with far fewer failure events than normal operations.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment.
    ```bash
    mkdir predictive-maintenance
    cd predictive-maintenance
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. Note the inclusion of `imbalanced-learn`.
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Load and Explore the Data**
1.  Download the `predictive_maintenance.csv` file from the Kaggle dataset.
2.  In your Jupyter Notebook, load the data and perform an initial exploration.
    ```python
    import pandas as pd

    df = pd.read_csv('predictive_maintenance.csv')

    print(df.head())
    print(df.info())

    # This is the most important step in exploration for this project
    print("\nClass Distribution:")
    print(df['Target'].value_counts())
    ```
    You will notice that `0` (No Failure) vastly outnumbers `1` (Failure). This confirms the class imbalance.

**Step 3: Preprocess the Data**
1.  **Feature Selection:** Drop columns that are just identifiers and not useful for prediction.
    ```python
    df_processed = df.drop(['UDI', 'Product ID'], axis=1)
    ```
2.  **Categorical Encoding:** Convert the categorical `Type` column into numerical format using one-hot encoding.
    ```python
    df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)
    ```
3.  **Split Data into Features and Target:** Separate your dataset into `X` (the features) and `y` (the target variable you want to predict).
    ```python
    X = df_processed.drop('Target', axis=1)
    y = df_processed['Target']
    
    # Also drop the specific failure type, as we are only predicting if a failure occurs
    X = X.drop('Failure Type', axis=1)
    ```
4.  **Create Training and Testing Sets:** Split the data before doing anything else to ensure the test set remains a true, unseen evaluation set.
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ```
    *Using `stratify=y` is crucial here—it ensures that the proportion of failures is the same in both the train and test sets.*

**Step 4: Handle Class Imbalance with SMOTE**
1.  We will apply the **S**ynthetic **M**inority **O**ver-sampling **TE**chnique (SMOTE) to generate new, synthetic examples of the minority class (failures). **This must only be done on the training data.**
    ```python
    from imblearn.over_sampling import SMOTE

    print("Before SMOTE:", y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", y_train_smote.value_counts())
    ```
    You will see that the number of `0`s and `1`s in the training set is now equal.

**Step 5: Train the Predictive Model**
1.  We will use `XGBoost`, a powerful and efficient algorithm that performs very well on tabular data like ours.
    ```python
    import xgboost as xgb

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

    # Train the model on the balanced (SMOTE) training data
    model.fit(X_train_smote, y_train_smote)
    ```

**Step 6: Evaluate Model Performance**
1.  Make predictions on the original, unseen test set (`X_test`).
    ```python
    y_pred = model.predict(X_test)
    ```
2.  Evaluate the predictions. **Accuracy is a misleading metric here.** Instead, we will use a `classification_report` and `confusion_matrix`, which give us a better view of performance on the minority class.
    ```python
    from sklearn.metrics import classification_report, confusion_matrix

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    ```
    *Focus on the **Recall** for class `1`. This tells you: "Of all the actual failures, what percentage did our model correctly identify?" This is often the most important metric for this problem.*

**Step 7: Explain the Model's Predictions**
1.  A key part of machine learning is understanding *why* a model makes its predictions. We can inspect the model's `feature_importances_` to see which sensor readings were most influential.
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importance for Predicting Equipment Failure')
    plt.show()
    ```

#### **6. Success Criteria**
*   The team correctly applies SMOTE to the training data only, not the test data.
*   The trained model achieves a **Recall score for the failure class (1) that is significantly higher** than what would be achieved by a naive model.
*   The team can interpret the `classification_report` and explain the business meaning of Precision vs. Recall for this use case.
*   The team can generate and interpret a feature importance plot, identifying the top 3 telemetry signals that predict failure.

#### **7. Next Steps & Extensions**
*   **Hyperparameter Tuning:** Use `GridSearchCV` to find the optimal parameters for the `XGBClassifier` to further improve performance.
*   **Explore Different Resampling Techniques:** Compare SMOTE to other techniques from `imbalanced-learn`, such as Random Undersampling.
*   **Cost-Benefit Analysis:** Assign a business cost to False Positives (unnecessary maintenance) and False Negatives (missed failures) to find a model threshold that maximizes business value.
