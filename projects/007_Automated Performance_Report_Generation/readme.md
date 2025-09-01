### **Project 7: Automated Network Performance Report Generation**

#### **1. Objective**
To build a system that automatically generates human-readable, narrative summaries of network performance from structured, numerical data. This project introduces the team to **prompt engineering** and the use of LLM pipelines (via LangChain) for sophisticated text generation tasks.

#### **2. Business Value**
This capability addresses a significant communication gap in engineering organizations:
*   **Time Savings:** Automates the tedious and time-consuming process of writing weekly or daily performance reports.
*   **Improved Stakeholder Communication:** Provides clear, consistent, and easily understandable updates for managers, executives, and non-technical stakeholders.
*   **Data-Driven Narratives:** Ensures that performance summaries are directly and accurately tied to the underlying metrics, removing subjective interpretation.

#### **3. Core Libraries**
*   `pandas` & `numpy`: To create the structured performance data that will be the input for our system.
*   `langchain`: A powerful framework for developing applications powered by language models. It simplifies the process of creating chains of operations (e.g., get data -> format prompt -> call LLM -> get response).
*   `langchain-community` & `langchain-huggingface`: To integrate open-source LLMs from the Hugging Face Hub.
*   `transformers` & `torch`: The underlying libraries for running the LLM.

#### **4. Dataset**
*   **Approach:** **Synthetic Data Generation**.
*   **Why:** This project is entirely process-oriented. The goal is to learn how to transform structured numbers into unstructured text. Creating our own input data is the most direct and effective way to learn this skill. We will simulate a weekly performance summary table.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment.
    ```bash
    mkdir report-generator
    cd report-generator
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. `langchain` and its ecosystem are key.
    ```bash
    pip install pandas numpy langchain langchain-community langchain-huggingface transformers torch jupyterlab
    ```
3.  **Hugging Face Login (Important):** To use many of the models from the Hugging Face Hub, you need to be logged in.
    *   Go to [huggingface.co](https://huggingface.co), create an account, and get an Access Token from your profile settings.
    *   Run the following command in your terminal and paste your token when prompted:
    ```bash
    huggingface-cli login
    ```
4.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Generate the Input Data**
1.  In your Jupyter Notebook, create a Pandas DataFrame that represents a weekly summary of network metrics for different regions.
    ```python
    import pandas as pd

    data = {
        'Region': ['US-West', 'US-East', 'EU-Central', 'APAC-South'],
        'Avg_Latency_ms': [35.2, 28.5, 45.1, 85.7],
        'Uptime_Percentage': [99.99, 99.95, 99.99, 99.98],
        'Peak_Bandwidth_Gbps': [890, 950, 720, 650],
        'Packet_Loss_Rate': [0.001, 0.005, 0.002, 0.003]
    }
    df_metrics = pd.DataFrame(data)

    print("--- Weekly Network Metrics ---")
    print(df_metrics)
    ```

**Step 3: Define the Prompt Template**
This is the core of prompt engineering. We create a template that defines the structure of our request to the LLM, with placeholders for our data.
1.  Import the necessary `langchain` modules.
2.  Create a `PromptTemplate`.
    ```python
    from langchain.prompts import PromptTemplate

    template_string = """
    As a senior network operations analyst, write a concise, one-paragraph executive summary of the network performance for the {Region} region.
    Your tone should be professional and clear.
    
    Use the following metrics in your summary:
    - Average Latency: {Avg_Latency_ms} ms
    - Network Uptime: {Uptime_Percentage}%
    - Peak Bandwidth Usage: {Peak_Bandwidth_Gbps} Gbps
    - Packet Loss Rate: {Packet_Loss_Rate}%

    Highlight any potential areas of concern. For example, any latency above 50ms or uptime below 99.98% should be mentioned as needing observation.
    """

    prompt_template = PromptTemplate(
        input_variables=['Region', 'Avg_Latency_ms', 'Uptime_Percentage', 'Peak_Bandwidth_Gbps', 'Packet_Loss_Rate'],
        template=template_string
    )
    ```

**Step 4: Set Up the LLM and the LangChain Chain**
1.  **Choose an LLM:** We will use an open-source model from the Hugging Face Hub. `google/flan-t5-large` is a good choice for instruction-following tasks like this.
    ```python
    from langchain_huggingface import HuggingFacePipeline

    # This initializes the connection to a model on the Hugging Face Hub
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 150}, # Limit the length of the output
    )
    ```
2.  **Create the Chain:** An `LLMChain` binds our prompt template to the LLM. It's the object that will execute our request.
    ```python
    from langchain.chains import LLMChain

    chain = LLMChain(llm=llm, prompt=prompt_template)
    ```

**Step 5: Generate the Reports**
1.  Now, we can iterate through our DataFrame and use the chain to generate a report for each region.
    ```python
    # The chain can be run on a dictionary or a list of dictionaries.
    # We will convert our DataFrame to a list of dictionaries.
    list_of_metrics = df_metrics.to_dict(orient='records')

    for metrics in list_of_metrics:
        report = chain.invoke(metrics)
        print(f"--- Performance Report for {metrics['Region']} ---")
        print(report['text'])
        print("-" * 50 + "\n")
    ```

**Step 6: Analyze the Output**
1.  Read the generated summaries for each region.
2.  Notice how the LLM incorporates the specific metrics into a natural-sounding paragraph.
3.  Check if it correctly followed the instruction to flag areas of concern (e.g., for the `EU-Central` and `APAC-South` regions with their higher latency).

#### **6. Success Criteria**
*   The team can successfully create a `PromptTemplate` in LangChain with multiple input variables.
*   The team can successfully initialize an LLM from the Hugging Face Hub and connect it to a `LLMChain`.
*   The system successfully generates a unique, coherent, and contextually aware text summary for each row in the input DataFrame.
*   The generated reports correctly incorporate the numerical data and follow the specific instructions given in the prompt (e.g., highlighting areas of concern).

#### **7. Next Steps & Extensions**
*   **Experiment with Different Models:** Swap out `google/flan-t5-large` for other models (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) to see how the quality and tone of the generated text change.
*   **Chain of Thought Prompting:** For more complex reports, modify the prompt to ask the LLM to "think step-by-step." For example, first ask it to identify key highlights, then ask it to identify key concerns, and finally ask it to write the summary. This often improves the quality of the output.
*   **Full Automation:** Create a Python script that pulls live metrics from a database (or a CSV file for simulation), runs the LangChain process, and emails the generated reports to a list of stakeholders, creating a fully automated reporting pipeline.

---