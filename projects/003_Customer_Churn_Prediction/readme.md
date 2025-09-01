
### **Project 3: Customer Churn Prediction**

#### **1. Objective**
To build a classification model that predicts whether a customer is likely to churn and, more importantly, to understand *why* the model makes its predictions. This project's primary focus is on moving beyond just prediction accuracy to generating actionable business insights through model explainability.

#### **2. Business Value**
Identifying customers at risk of churning before they leave is immensely valuable. It allows the business to:
*   **Implement Proactive Retention Strategies:** Target at-risk customers with special offers, service check-ins, or loyalty programs.
*   **Reduce Revenue Loss:** It is significantly more expensive to acquire a new customer than to retain an existing one.
*   **Improve Products and Services:** By understanding the common drivers of churn (e.g., poor performance on a specific service), we can identify areas for improvement.

#### **3. Core Libraries**
*   `pandas` & `numpy`: For data loading and manipulation.
*   `scikit-learn`: For data preprocessing, model training, and evaluation.
*   `matplotlib` & `seaborn`: For exploratory data analysis.
*   `shap`: A powerful library for explaining the output of any machine learning model. **This is the key new library for this project.**

#### **4. Dataset**
*   **Primary Dataset:** **Telco Customer Churn** ([Verified Link on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
*   **Why it's suitable:** This is the quintessential dataset for this task. It is clean, well-documented, and contains a perfect mix of features:
    *   **Customer Demographics:** `gender`, `SeniorCitizen`, etc.
    *   **Subscribed Services:** `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
    *   **Account Information:** `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`.
    *   **Target Variable:** A binary `Churn` label.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment.
    ```bash
    mkdir customer-churn
    cd customer-churn
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. `shap` is the key addition.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn shap jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Load and Explore the Data**
1.  Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file from Kaggle.
2.  In your Jupyter Notebook, load the data and perform Exploratory Data Analysis (EDA).
    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Explore the relationship between Contract type and Churn
    sns.countplot(x='Contract', hue='Churn', data=df)
    plt.title('Churn by Contract Type')
    plt.show()
    ```
    *This EDA step is crucial for building intuition. You should quickly see that customers on "Month-to-month" contracts are far more likely to churn.*

**Step 3: Preprocess the Data**
1.  **Data Cleaning:** The `TotalCharges` column has some missing values represented as spaces. We need to convert it to a numeric type and fill the missing values (e.g., with the median).
    ```python
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    ```
2.  **Feature Encoding:** Convert all categorical columns to numerical format.
    ```python
    # Drop customerID as it's just an identifier
    df_processed = df.drop('customerID', axis=1)
    
    # Use one-hot encoding for categorical variables
    df_processed = pd.get_dummies(df_processed, drop_first=True) # drop_first=True avoids multicollinearity
    ```
3.  **Split Data:** Separate the features (`X`) from the target (`y`) and create training and testing sets.
    ```python
    from sklearn.model_selection import train_test_split

    X = df_processed.drop('Churn_Yes', axis=1)
    y = df_processed['Churn_Yes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

**Step 4: Train a Classification Model**
1.  For this problem, a `RandomForestClassifier` is a great choice as it's powerful and `shap` can explain it efficiently.
    ```python
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```

**Step 5: Evaluate Model Performance**
1.  Check the model's performance on the unseen test set.
    ```python
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    ```

**Step 6: Explain the Model's Predictions with SHAP**
This is the core of the project. We will now uncover the "why" behind the predictions.
1.  **Initialize the SHAP Explainer:**
    ```python
    import shap

    # For tree-based models like RandomForest, TreeExplainer is efficient
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    ```
2.  **Global Feature Importance (Summary Plot):** This plot shows the most important features for the model overall.
    ```python
    # The plot shows the impact of each feature on the model's output
    # We are interested in the shap_values for the "Churn" class (class 1)
    shap.summary_plot(shap_values[1], X_test, plot_type="dot")
    ```
    *   **How to read this plot:**
        *   Each point is a customer from the test set.
        *   **Feature Importance:** Features are ranked from most important (top) to least important (bottom).
        *   **Impact:** The x-axis shows the SHAP value. A positive value means that feature pushed the prediction towards "Churn." A negative value pushed it away from "Churn."
        *   **Color:** The color shows the feature's value (e.g., red for high `tenure`, blue for low `tenure`).
    *   You will likely see that low tenure (blue dots on the `tenure` row with positive SHAP values) is a strong predictor of churn.

3.  **Individual Prediction Explanation (Force Plot):** Let's explain the prediction for a single customer.
    ```python
    # Explain the first customer in the test set
    shap.initjs() # required for javascript-based plots in notebooks
    shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])
    ```
    *   **How to read this plot:**
        *   The **base value** is the average prediction for all customers.
        *   **Red arrows** show features that are pushing the prediction *higher* (towards churn).
        *   **Blue arrows** show features that are pushing the prediction *lower* (towards staying).
        *   The **final prediction** (`f(x)`) is the result after considering all feature impacts.

#### **6. Success Criteria**
*   The model achieves a reasonable accuracy (typically ~80% for this dataset).
*   The team can successfully generate and interpret the **SHAP summary plot**, identifying the top 3 global drivers of churn.
*   The team can select a single customer predicted to churn and use a **SHAP force plot** to explain exactly which features contributed to that specific prediction.

#### **7. Next Steps & Extensions**
*   **Segmented Analysis:** Create SHAP summary plots for different customer segments (e.g., customers with `InternetService_Fiber optic`) to see if churn drivers differ.
*   **Develop a "Churn Risk Score":** Instead of a binary prediction, use the model's `predict_proba()` method to assign each customer a risk score from 0 to 1.
*   **Actionable Insights:** Translate the SHAP findings into a business recommendation (e.g., "We should offer a loyalty discount to month-to-month customers with high monthly charges after their first three months of service").