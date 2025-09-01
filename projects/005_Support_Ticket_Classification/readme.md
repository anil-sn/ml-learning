### **Project 5: Customer Support Ticket Classification with LLMs**

#### **1. Objective**
To build and compare two systems for automatically classifying customer support tickets based on their text content. We will first build a traditional machine learning baseline and then use a powerful, pre-trained Large Language Model (LLM) to perform the same task, demonstrating the advantages of modern NLP.

#### **2. Business Value**
Automating ticket classification is a high-impact task for operational efficiency:
*   **Faster Response Times:** Tickets are instantly routed to the correct team (e.g., Networking, Billing, Outage Support) without manual triage.
*   **Improved Prioritization:** High-priority issues (like a network outage) can be automatically flagged and escalated.
*   **Workload Reduction:** Reduces the manual, repetitive workload on support staff, allowing them to focus on solving customer problems.

#### **3. Core Libraries**
*   `scikit-learn`: To build the traditional baseline model (`TfidfVectorizer` and `LogisticRegression`).
*   `pandas`: For data handling and analysis.
*   `transformers` & `datasets` (from Hugging Face): The industry-standard libraries for working with state-of-the-art NLP models.
*   `torch`: The deep learning framework that powers the LLMs.

#### **4. Dataset**
*   **Primary Dataset:** **Customer Support on Twitter** ([Verified Link on Hugging Face](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints))
*   **Why it's suitable:** This dataset is part of the RAFT (Retrieval Augmented Fine-Tuning) benchmark and contains real-world customer complaints from Twitter. It has a `Tweet text` column and a `Label` column, making it perfectly suited for a multi-class text classification task. The text is informal and challenging, providing a realistic test for our models.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment.
    ```bash
    mkdir ticket-classifier
    cd ticket-classifier
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. `transformers` and `torch` are substantial packages.
    ```bash
    pip install pandas scikit-learn transformers datasets torch jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Load and Prepare the Data**
1.  Use the `datasets` library to load the data directly from the Hugging Face Hub. This is a very efficient workflow.
    ```python
    from datasets import load_dataset
    import pandas as pd

    # Load the training split of the dataset
    dataset = load_dataset("ought/raft", "twitter_complaints", split="train")
    
    # Convert to a pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # We only need the text and the label columns
    df = df[['Tweet text', 'Label']]
    df.rename(columns={'Tweet text': 'text', 'Label': 'label'}, inplace=True)

    print(df.head())
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    ```

**Step 3: Build a Traditional ML Baseline**
This model serves as our benchmark. It doesn't use any deep learning.
1.  **Split the data:**
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    ```
2.  **Create a pipeline:** We will use a pipeline to combine text vectorization (`TfidfVectorizer`) and classification (`LogisticRegression`).
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    baseline_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    ```
3.  **Train and evaluate the baseline:**
    ```python
    from sklearn.metrics import classification_report

    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print("--- Baseline Model Performance ---")
    print(classification_report(y_test, y_pred_baseline))
    ```

**Step 4: Classify with a Zero-Shot LLM**
Now, let's use a powerful, pre-trained LLM. The "zero-shot" technique allows the model to classify text into categories it has never been explicitly trained on.
1.  **Initialize the Zero-Shot Pipeline:**
    ```python
    from transformers import pipeline

    # This will download the model the first time you run it
    classifier_llm = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    ```
2.  **Define the candidate labels:** Get the unique labels directly from our dataset.
    ```python
    candidate_labels = df['label'].unique().tolist()
    print("Candidate Labels:", candidate_labels)
    ```
3.  **Make predictions:** Let's run the model on a small sample of the test set first to see how it works.
    ```python
    # Note: Running this on the full test set can be slow without a GPU.
    # We'll run it on the first 100 samples for this example.
    sample_test_text = X_test.tolist()[:100]
    sample_y_test = y_test.tolist()[:100]

    predictions_llm = classifier_llm(sample_test_text, candidate_labels)
    ```
4.  **Process and evaluate the LLM results:** The output from the pipeline needs to be processed to get the final predicted label.
    ```python
    # The pipeline returns a list of dicts; we just want the highest-scoring label
    y_pred_llm = [pred['labels'][0] for pred in predictions_llm]

    print("\n--- Zero-Shot LLM Performance (on first 100 samples) ---")
    print(classification_report(sample_y_test, y_pred_llm))
    ```

**Step 5: Compare the Results**
1.  Look at the two `classification_report` outputs side-by-side.
2.  **Discussion:**
    *   The **Baseline Model** was trained specifically on our data. Its performance is likely quite good. It is also very fast to run (inference).
    *   The **Zero-Shot LLM** was *never trained* on our specific data or labels. The fact that it can perform the classification task at all is remarkable. While its performance might be slightly lower than the baseline, its major advantage is **flexibility**. You can change the labels or the task entirely without any retraining.

#### **6. Success Criteria**
*   The team can successfully build and train the TF-IDF + Logistic Regression baseline model.
*   The team can successfully use the Hugging Face `pipeline` to perform zero-shot classification on a sample of the data.
*   The team can articulate the key differences between the two approaches, explaining the trade-offs between a specialized, trained model (baseline) and a general-purpose, zero-shot model (LLM).

#### **7. Next Steps & Extensions**
*   **Fine-Tuning:** The ultimate step for performance is to **fine-tune** a smaller pre-trained model (like `distilbert-base-uncased`) on our specific training data (`X_train`, `y_train`). This combines the power of pre-training with specialization on our task and will almost certainly outperform both the baseline and the zero-shot model. The Hugging Face `Trainer` API is the tool for this.
*   **Build a Demo:** Use a simple library like `Gradio` to build an interactive web interface for your best-performing model. This allows anyone on the team to input a ticket text and see the model's predicted category in real-time.

---