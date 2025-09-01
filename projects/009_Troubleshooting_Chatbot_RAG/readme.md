
### **Project 9: Chatbot for Network Troubleshooting**

#### **1. Objective**
To build a conversational AI assistant that can answer technical questions about network troubleshooting by consulting a private knowledge base. This project will teach the team how to use **Retrieval-Augmented Generation (RAG)**, a powerful technique that combines the retrieval of relevant documents with the generative power of an LLM.

#### **2. Business Value**
A troubleshooting assistant has a direct and immediate impact on operational efficiency:
*   **Reduced Troubleshooting Time:** Provides instant, accurate answers to common (and uncommon) technical queries for field technicians and NOC staff.
*   **Knowledge Centralization:** Captures and institutionalizes expert knowledge that might otherwise be siloed in documentation, wikis, or individual experts' heads.
*   **Improved Consistency:** Ensures that all team members receive the same, up-to-date information and follow standardized procedures.
*   **24/7 Support:** Acts as an always-on "Tier 1" support expert for internal teams.

#### **3. Core Libraries**
*   `langchain` & its ecosystem: The core framework for building the RAG pipeline.
*   `langchain-huggingface`: To use open-source models and embedding functions.
*   `faiss-cpu`: A library from Facebook AI for efficient similarity search, which will act as our vector store.
*   `pypdf`: A library to load and read text from PDF documents, our knowledge source.
*   `gradio`: A simple and fast way to create a web-based user interface for our chatbot.

#### **4. Dataset**
*   **Approach:** **Using Internal or Public Technical Documents**.
*   **Why:** The goal of RAG is to ground the LLM in your specific data. Therefore, the "dataset" is your knowledge base. For this project, a great public document to use is a well-known, detailed networking guide.
*   **Primary Document:** **"Juniper Networks Security Configuration Guide"** ([Public PDF from Juniper](https://www.juniper.net/documentation/us/en/software/junos/security/junos-sec-config-guide.pdf)). This is a long, dense, and highly technical document, making it a perfect real-world test for our RAG system.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a project folder and a Python virtual environment.
    ```bash
    mkdir rag-chatbot
    cd rag-chatbot
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install all necessary libraries. This is a substantial list.
    ```bash
    pip install langchain langchain-community langchain-huggingface transformers torch faiss-cpu pypdf gradio jupyterlab
    ```
3.  Log in to the Hugging Face CLI as you did in Project 7.
4.  Download the `junos-sec-config-guide.pdf` from the link above and place it in your project folder.
5.  Start a Jupyter Lab session for building and testing the pipeline step-by-step.

**Step 2: Load and Chunk the Knowledge Base**
The first step in RAG is to load our document and split it into smaller, manageable chunks. This is crucial for effective retrieval.
```python
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load the document
loader = PyPDFLoader("junos-sec-config-guide.pdf")
docs = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunked_docs = text_splitter.split_documents(docs)

print(f"Loaded {len(docs)} pages.")
print(f"Split document into {len(chunked_docs)} chunks.")
print("\nExample Chunk:")
print(chunked_docs[50].page_content)
```

**Step 3: Create Embeddings and Store in a Vector Database**
Next, we convert each text chunk into a numerical vector (an "embedding") and store these vectors in a database that allows for fast similarity searches.
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Choose an embedding model
# This model is small, fast, and effective for retrieval tasks
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Create the vector store
# This will download the embedding model and then process all chunks.
# This step can take a few minutes.
vector_store = FAISS.from_documents(chunked_docs, embedding_model)

print("\nVector store created successfully.")```

**Step 4: Build the RAG Chain**
Now we assemble the components using LangChain. The chain will:
1.  Take a user's question.
2.  Find the most relevant document chunks from the `vector_store`.
3.  Format those chunks and the question into a prompt.
4.  Send the prompt to an LLM to generate an answer.
```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# 1. Set up the LLM for generation
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 250},
)

# 2. Create the prompt template
template = """
You are a helpful network troubleshooting assistant. Use the following pieces of context from the knowledge base to answer the user's question.
If you don't know the answer from the context provided, just say that you don't know, don't try to make up an answer.
Keep the answer concise and helpful.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 3. Create the RAG chain (RetrievalQA)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" means we will stuff all retrieved chunks into the prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve the top 3 most relevant chunks
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

print("\nRAG Chain is ready.")
```

**Step 5: Test the Chatbot**
Let's ask a specific question that can only be answered by consulting the Juniper PDF.
```python
question = "What is the purpose of a security zone in Junos OS?"
result = rag_chain.invoke({"query": question})

print("\n--- Query ---")
print(question)
print("\n--- Answer ---")
print(result['result'])
print("\n--- Sources ---")
for doc in result['source_documents']:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:200]}...")
```
*You should see a correct, detailed answer generated by the LLM, grounded in the specific text retrieved from the PDF.*

**Step 6: Create a User Interface with Gradio**
Let's wrap our chain in a simple web interface so anyone can use it.
1.  Create a file named `app.py`.
2.  Copy all the code from steps 2, 3, and 4 into this file.
3.  Add the following Gradio code at the end:
    ```python
    # (All the code from above goes here)
    import gradio as gr

    def get_answer(question):
        result = rag_chain.invoke({"query": question})
        return result['result']

    iface = gr.Interface(fn=get_answer, inputs="text", outputs="text",
                         title="Network Troubleshooting Chatbot",
                         description="Ask a question about the Juniper Security Guide.")

    iface.launch()
    ```
4.  Run the app from your terminal:
    ```bash
    python app.py
    ```
5.  Open the URL provided in your browser to interact with your chatbot.

#### **6. Success Criteria**
*   The team can successfully load a PDF document, split it into chunks, and create a `FAISS` vector store.
*   The RAG chain can correctly answer specific questions by retrieving relevant context from the document and providing it to the LLM.
*   When asked a question whose answer is *not* in the document, the chatbot correctly states that it does not know the answer, demonstrating that it is grounded in the provided context.
*   The team can successfully launch the `Gradio` web interface and interact with the chatbot in real-time.

#### **7. Next Steps & Extensions**
*   **Expand the Knowledge Base:** Add more documents (other PDFs, text files, etc.) to the vector store to create a more comprehensive knowledge base.
*   **Experiment with Retrievers:** Try different retriever settings, such as changing the number of retrieved documents (`k`) or using more advanced retrieval strategies like MMR (Maximal Marginal Relevance) to get more diverse results.
*   **Add Chat History:** Use LangChain's memory components to allow for conversational follow-up questions, making the chatbot feel more natural.

---
