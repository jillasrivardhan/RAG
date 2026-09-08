# 🎓 College FAQ RAG Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project that allows students to ask questions about college admissions, library rules, and examination policies.

The project combines **document retrieval, embeddings, vector search, and an LLM-powered agent** to provide answers based on information stored in a college FAQ document.

---

## 📌 Project Overview

Traditional LLM applications can generate answers using their general knowledge, but they may not know information specific to a particular college.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The application:

1. Loads college FAQ information from a text file.
2. Splits the document into smaller chunks.
3. Converts the chunks into numerical embeddings using Jina Embeddings.
4. Stores the embeddings in a FAISS vector database.
5. Retrieves the most relevant information for a student's question.
6. Provides the retrieved information to an LLM.
7. Uses a LangChain agent and retriever tool to generate the final answer.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with Large Language Models.

Instead of asking an LLM to answer a question entirely from its trained knowledge, RAG first searches a knowledge base for relevant information and then uses that information to generate the answer.

### RAG Pipeline

```text
                 College FAQ
                     │
                     ▼
              Document Loader
                     │
                     ▼
               Text Splitting
                     │
                     ▼
              Jina Embeddings
                     │
                     ▼
               FAISS Vector Store
                     │
                     ▼
              Retriever Tool
                     │
Student Question ───► Agent / LLM
                     │
                     ▼
                Final Answer
```

---

## ✨ Features

* 📄 Loads college FAQ data from a text file
* ✂️ Splits documents into manageable chunks
* 🧮 Generates embeddings using Jina Embeddings
* 🔎 Performs semantic similarity search
* 🗂️ Uses FAISS for vector storage and retrieval
* 🛠️ Exposes the retriever as a LangChain tool
* 🤖 Uses Groq-powered LLM inference
* 🧠 Uses a LangChain agent to decide when to use the FAQ search tool
* 🚫 Designed to avoid making up college-specific information
* 👨‍🎓 Beginner-friendly RAG implementation

---

## 📚 Knowledge Base

The current knowledge base is:

```text
data/
└── college-faq.txt
```

The FAQ contains information about three major areas:

### 📝 College Admissions

* Application opening date
* Application closing date
* Required documents
* Application fee

### 📖 College Library

* Library working days
* Library opening and closing time
* Maximum number of books students can borrow
* Borrowing period
* Late fee

### 📝 College Examinations

* Required arrival time
* College ID requirement
* Prohibited devices

---

## 🛠️ Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Programming language            |
| LangChain        | RAG and agent framework         |
| Jina Embeddings  | Text embeddings                 |
| FAISS            | Vector similarity search        |
| Groq             | LLM inference                   |
| `python-dotenv`  | Environment variable management |
| Jupyter Notebook | Development and experimentation |

---

## 📂 Project Structure

```text
RAG-main/
│
├── data/
│   └── college-faq.txt
│
├── rag.ipynb
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## ⚙️ How It Works

### 1. Load Environment Variables

API keys are loaded from a `.env` file.

```python
from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")
```

---

### 2. Load the FAQ Document

The college FAQ is loaded using LangChain's `TextLoader`.

```python
from langchain_community.document_loaders import TextLoader

data_path = os.path.join("data", "college-faq.txt")

loader = TextLoader(
    data_path,
    encoding="utf-8",
    autodetect_encoding=True
)

docs = loader.load()
```

---

### 3. Split the Document

The document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)
```

### Why Chunking?

Large documents should not always be passed to the LLM as a single piece of text.

Chunking allows the system to:

* Retrieve only relevant information
* Reduce unnecessary context
* Improve retrieval efficiency
* Work with larger documents

The project currently uses:

```text
Chunk Size: 500 characters
Chunk Overlap: 50 characters
```

---

### 4. Generate Embeddings

The project uses **Jina Embeddings** to convert text chunks into numerical vectors.

```python
from langchain_community.embeddings import JinaEmbeddings

vectors = JinaEmbeddings(
    jina_api_key=jina_key,
    model_name="jina-embeddings-v2-base-en"
)
```

Embeddings allow the system to compare the semantic meaning of the student's question with the information stored in the FAQ.

---

### 5. Create the FAISS Vector Store

The document chunks and their embeddings are stored using FAISS.

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunks,
    vectors
)
```

FAISS allows the system to efficiently search for chunks that are semantically similar to a student's question.

---

### 6. Perform Similarity Search

The vector store can directly search for relevant information.

```python
query = "What is the admission process?"

top_match = vector_store.similarity_search(
    query,
    k=2
)
```

The `k=2` parameter means the system retrieves the two most relevant chunks.

---

## 🔧 Retriever Tool

One of the main learning objectives of this project is converting the retriever into a **tool that an LLM agent can use**.

```python
from langchain_core.tools import create_retriever_tool

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

retriever_tool = create_retriever_tool(
    retriever,
    name="college_faq_search",
    description="""
    Search the college FAQ to answer questions about:
    - college admissions
    - application dates
    - required admission documents
    - application fees
    - library hours
    - book borrowing limits
    - borrowing periods
    - library late fees
    - examination rules
    - examination arrival time
    - required college ID
    - prohibited devices during examinations

    Use this tool whenever a student asks a question about
    college policies, admissions, library rules, or examinations.
    """
)
```

### Why use a tool?

A retriever by itself can search the vector database.

A **retriever tool** makes that search functionality available to an agent.

```text
Student Question
       │
       ▼
      Agent
       │
       │ decides to use tool
       ▼
college_faq_search
       │
       ▼
    Retriever
       │
       ▼
   FAISS Search
       │
       ▼
Relevant FAQ Chunks
       │
       ▼
      Agent
       │
       ▼
 Final Answer
```

---

## 🤖 LLM and Agent

The project uses a Groq-powered chat model.

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4
)
```

The retriever tool is then provided to the LangChain agent.

```python
from langchain.agents import create_agent

college_assistant = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt="..."
)
```

The agent can use the `college_faq_search` tool when a student asks a college-specific question.

---

## 🛡️ Hallucination Control

The system prompt instructs the assistant to:

* Use only information retrieved from the FAQ
* Avoid making assumptions
* Avoid inventing college policies
* Preserve important dates, times, fees, and rules
* Ignore irrelevant retrieved information
* Tell the student when the FAQ does not contain the requested information

For example:

```text
Student:
What is the hostel fee?

Assistant:
I'm sorry, I don't have that information in the college FAQ.
```

This is important because the current knowledge base does not contain hostel information.

---

## 💬 Example Questions

You can test the assistant with questions such as:

```text
What is the application fee?

When do college applications open?

What documents are required for admission?

How many books can I borrow?

How long can I keep a borrowed book?

What is the library late fee?

What time does the library close?

How early should I arrive for an examination?

Do I need my college ID for an examination?

Are phones allowed during examinations?
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
```

### Important

Never commit your `.env` file or API keys to GitHub.

The project already includes `.env` in `.gitignore`.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd RAG-main
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv ragenv
```

Activate it:

```powershell
ragenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create:

```text
.env
```

and add:

```text
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
```

### 5. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
rag.ipynb
```

Run the cells sequentially.

---

## 📋 Requirements

The project dependencies are listed in `requirements.txt`.

Main packages include:

```text
langchain
langchain-core
langchain-community
langchain-groq
langchain-text-splitters
faiss-cpu
python-dotenv
jupyter
ipykernel
streamlit
```

---

## 🧪 Testing the Assistant

The project includes a Python function for asking questions:

```python
def answer_student_question(question: str) -> str:

    response = college_assistant.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    answer = response["messages"][-1].content

    return answer
```

Example:

```python
answer_student_question(
    "What is the application fee?"
)
```

---

## 🎯 Learning Objectives

This project is designed as a practical introduction to RAG.

By building this project, you can understand:

* What RAG is
* Document ingestion
* Document loading
* Text chunking
* Chunk size
* Chunk overlap
* Embeddings
* Vector databases
* FAISS similarity search
* Retrievers
* Retriever tools
* LLM integration
* LangChain agents
* Prompt engineering
* Hallucination prevention
* Environment variables
* Basic RAG pipeline design

---

## 🚀 Future Improvements

This project can be extended into a more complete RAG application.

### 🔹 Streamlit UI

Create a chat interface where students can interact with the assistant through a web browser.

```text
Student
   │
   ▼
Streamlit Chat UI
   │
   ▼
RAG Agent
   │
   ▼
FAQ Retriever
   │
   ▼
FAISS
   │
   ▼
LLM
   │
   ▼
Answer
```

### 🔹 Persistent Vector Database

Instead of creating the FAISS index every time the notebook runs, save and reload the vector store.

### 🔹 More Documents

Add additional knowledge sources:

```text
data/
├── college-faq.txt
├── hostel-faq.txt
├── departments.txt
├── scholarships.txt
└── campus-rules.txt
```

### 🔹 Conversation Memory

Allow students to ask follow-up questions such as:

```text
Student: How many books can I borrow?

Assistant: You can borrow up to three books.

Student: How long can I keep them?

Assistant: The borrowing period is 14 days.
```

### 🔹 Source Citations

Display the document/chunk used to generate each answer.

### 🔹 Better Retrieval

Experiment with:

* Different chunk sizes
* Different chunk overlaps
* Different embedding models
* Similarity search
* Maximum Marginal Relevance (MMR)
* Reranking

---

## 📊 Current RAG Configuration

| Component        | Configuration                    |
| ---------------- | -------------------------------- |
| Document         | `college-faq.txt`                |
| Loader           | `TextLoader`                     |
| Chunking         | `RecursiveCharacterTextSplitter` |
| Chunk Size       | 500                              |
| Chunk Overlap    | 50                               |
| Embeddings       | Jina Embeddings                  |
| Vector Store     | FAISS                            |
| Retrieved Chunks | 2                                |
| LLM Provider     | Groq                             |
| Model            | `openai/gpt-oss-120b`            |
| Temperature      | 0.4                              |
| Agent            | LangChain `create_agent`         |

---

## ⚠️ Current Limitations

This is a learning-focused project and currently has some limitations:

* The knowledge base contains only a small college FAQ.
* The project is primarily implemented in a Jupyter Notebook.
* The FAISS vector store is created during execution rather than being maintained as a separate persistent database.
* There is no production authentication system.
* There is no conversation history implementation.
* There is no dedicated evaluation framework for retrieval quality.
* The current project does not include a completed Streamlit chat interface.

---

## 📁 Recommended Next Step

A good next version of this project would separate the notebook into modules:

```text
RAG-College-FAQ/
│
├── data/
│   └── college-faq.txt
│
├── src/
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── tools.py
│   └── chatbot.py
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

This would make the project easier to maintain and would prepare it for deployment.

---
⚠️ Important Note

This is experimental/learning code.

It is NOT production-ready and should not be considered a production-grade RAG system.

There are still many things I want to improve, including:

🔹 Better retrieval
🔹 RAG evaluation
🔹 Persistent vector storage
🔹 Conversation memory
🔹 Source citations
🔹 Better error handling
🔹 Improved prompt safety
🔹 Streamlit UI
🔹 More robust document ingestion

For now, I'm happy with it as another step in my learning journey. 🚀

## 👨‍💻 Author

**Jilla srivardhan**

This project was created as part of a hands-on journey into **Generative AI, Retrieval-Augmented Generation, LangChain, embeddings, vector databases, and AI agents**.

---

## ⭐ If You Found This Project Useful

If this project helped you understand RAG, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and learning purposes.
