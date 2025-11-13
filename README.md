# 📚 PDF-AI Assistant

A **Streamlit-powered AI assistant** that lets you **chat with your PDF**!  
Upload any PDF, and the app extracts, embeds, and queries your content using **ChromaDB**, **Sentence Transformers**, and the **Groq LLM API**.  
It even talks back — thanks to **gTTS** text-to-speech integration.   

---

## 🚀 Features
-  **PDF Upload & Parsing** – Easily extract text from PDF files.  
-  **Vector Database with ChromaDB** – Stores document embeddings for efficient retrieval.  
-  **Smart Querying (RAG)** – Uses *Retrieval-Augmented Generation* to find the most relevant text chunks and generate precise answers.  
-  **Voice Response** – Converts AI-generated answers into speech using **gTTS**.  
-  **Persistent Memory** – Saves your embeddings inside `data/vectordb/` for reuse.  
-  **Streamlit UI** – Interactive, responsive, and browser-based interface.  

---

## 🧠 Model & Architecture

| Component | Description | Source |
|------------|--------------|---------|
| **Embedding Model** | Converts text into dense vector representations | `all-MiniLM-L6-v2` from **Sentence Transformers (Hugging Face)** |
| **Vector Database** | Stores embeddings and performs similarity search | **ChromaDB** |
| **LLM** | Generates intelligent, context-aware answers | **Groq API (LLM Endpoint)** |
| **TTS Engine** | Converts responses into speech | **gTTS (Google Text-to-Speech)** |
| **Framework** | User interface and app deployment | **Streamlit** |

---

## 🏗️ Project Structure

```
MyProject/
│
├── app.py
├── requirements.txt
├── data/
    └── vectordb/    ← Chroma vector database (create this folder manually)

```


---

## ⚙️ Setup & Run

### 🧩 Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

🔑 Add Your API Key: Open app.py and set your Groq API key.

▶️ Run the App
```bash
streamlit run app.py
```
